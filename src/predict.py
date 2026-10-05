"""
End-to-End Inference Pipeline
Author: Savan (Person 3)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Task P3.7: Raw resume text -> preprocessing -> feature extraction -> model -> predicted category
Task P3.8: Load saved artifacts for reproducible inference
"""

import os
import numpy as np
import joblib
from typing import Dict, Any, Union

from src.preprocessing import clean_resume_text, tokenize_resume
from src.data_loader import extract_text_from_pdf


class ResumeClassifierPipeline:
    """
    Full end-to-end prediction pipeline:
    1. Accept raw text or PDF
    2. Clean with identical preprocessing used during training
    3. Generate features (Word2Vec pooling or TF-IDF)
    4. Classify
    5. Return category + confidence
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        self.models_dir = os.path.abspath(models_dir)
        self.w2v_model = None
        self.vocab = None
        self.classifier = None
        self.label_encoder = None
        self.tfidf_vectorizer = None
        self.pipeline_type = None
        self.pipeline = None
        self.ml_metadata = None

        # Try Word2Vec + Dense NN first
        if self._try_load_word2vec_pipeline():
            self.pipeline_type = "word2vec_dense"
            print(f"Loaded Word2Vec + Dense NN pipeline from {self.models_dir}")
        elif self._try_load_tfidf_pipeline():
            self.pipeline_type = "tfidf_ml"
            print(f"Loaded TF-IDF + ML pipeline from {self.models_dir}")
        else:
            print("No trained model artifacts found. Pipeline will return placeholder predictions.")

    def _try_load_word2vec_pipeline(self) -> bool:
        """Load PyTorch Word2Vec + Dense NN artifacts."""
        w2v_path = os.path.join(self.models_dir, "word2vec_pytorch.pt")
        vocab_path = os.path.join(self.models_dir, "word2vec_vocab.json")
        clf_path = os.path.join(self.models_dir, "dense_nn_classifier.joblib")
        le_path = os.path.join(self.models_dir, "label_encoder.joblib")

        if all(os.path.exists(p) for p in [w2v_path, vocab_path, clf_path, le_path]):
            try:
                from src.word2vec import load_word2vec_artifacts
                self.w2v_model, self.vocab = load_word2vec_artifacts(self.models_dir)
                self.classifier = joblib.load(clf_path)
                self.label_encoder = joblib.load(le_path)
                return True
            except Exception as e:
                print(f"Warning: Failed to load Word2Vec pipeline: {e}")
        return False

    def _try_load_tfidf_pipeline(self) -> bool:
        """
        Load TF-IDF + ML artifacts written by src/run_p2_pipeline.py (Person 2 / branch jeel_p2).

        That pipeline writes:
          - models/tfidf_vectorizer.joblib
          - models/best_ml_model_<Name>.joblib   (e.g. best_ml_model_Linear_SVM.joblib)
          - models/final_pipeline.joblib         (sklearn Pipeline: tfidf -> clf)
        """
        import glob
        import json

        tfidf_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")

        # Prefer the self-contained sklearn Pipeline (tfidf + classifier in one artifact).
        pipeline_path = os.path.join(self.models_dir, "final_pipeline.joblib")
        if os.path.exists(pipeline_path):
            try:
                self.pipeline = joblib.load(pipeline_path)
                self.tfidf_vectorizer = self.pipeline
                self.classifier = self.pipeline
                meta_path = os.path.join(self.models_dir, "training_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.ml_metadata = json.load(f)
                self.label_encoder = None  # Pipeline exposes string classes directly
                return True
            except Exception as e:
                print(f"Warning: Failed to load final_pipeline.joblib: {e}")
                self.pipeline = None

        # Fallback: vectorizer + best_ml_model_<Name>.joblib
        if os.path.exists(tfidf_path):
            candidates = sorted(glob.glob(os.path.join(self.models_dir, "best_ml_model_*.joblib")))
            if not candidates:
                # Legacy/explicit name kept working.
                candidates = [os.path.join(self.models_dir, "best_ml_model.joblib")]
            try:
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                self.classifier = joblib.load(candidates[0])
                print(f"  Using classical ML artifact: {os.path.basename(candidates[0])}")
                le_path = os.path.join(self.models_dir, "label_encoder.joblib")
                if os.path.exists(le_path):
                    self.label_encoder = joblib.load(le_path)
                return True
            except Exception as e:
                print(f"Warning: Failed to load TF-IDF pipeline: {e}")
        return False

    def _get_w2v_embedding(self, tokens: list) -> np.ndarray:
        """Get document embedding via mean pooling of Word2Vec vectors."""
        import torch
        self.w2v_model.eval()
        vec_size = self.w2v_model.center_embeddings.embedding_dim

        indices = [self.vocab[t] for t in tokens if t in self.vocab]
        if not indices:
            return np.zeros(vec_size)

        with torch.no_grad():
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            word_vecs = self.w2v_model.center_embeddings(idx_tensor).numpy()
        return np.mean(word_vecs, axis=0)

    def predict(self, input_data: Union[str, os.PathLike], is_pdf: bool = False) -> Dict[str, Any]:
        """
        P3.7: Full prediction pipeline.
        """
        # Extract text
        if is_pdf or str(input_data).lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(str(input_data))
        else:
            raw_text = str(input_data)

        # Preprocess (identical to training)
        cleaned_text = clean_resume_text(raw_text)

        if not cleaned_text or len(cleaned_text.strip()) < 5:
            return {
                "predicted_category": "UNKNOWN",
                "confidence": 0.0,
                "cleaned_preview": cleaned_text[:200] if cleaned_text else "",
                "raw_length": len(raw_text),
                "error": "Resume text is empty or too short.",
            }

        # No model loaded
        if self.pipeline_type is None:
            return {
                "predicted_category": "MODEL_NOT_YET_TRAINED",
                "confidence": 0.0,
                "cleaned_preview": cleaned_text[:200] + ("..." if len(cleaned_text) > 200 else ""),
                "raw_length": len(raw_text),
            }

        # Feature extraction + classification
        confidence = 0.0
        top_categories = []

        if self.pipeline_type == "word2vec_dense":
            tokens = tokenize_resume(cleaned_text)
            embedding = self._get_w2v_embedding(tokens)
            X = embedding.reshape(1, -1)

            pred_encoded = self.classifier.predict(X)[0]
            pred_category = self.label_encoder.inverse_transform([pred_encoded])[0]

            if hasattr(self.classifier, "predict_proba"):
                probs = self.classifier.predict_proba(X)[0]
                confidence = float(np.max(probs))
                top_indices = np.argsort(probs)[::-1][:5]
                top_categories = [
                    {
                        "category": self.label_encoder.inverse_transform([idx])[0],
                        "confidence": float(probs[idx]),
                    }
                    for idx in top_indices
                ]

        elif self.pipeline_type == "tfidf_ml":
            # If Person 2's self-contained sklearn Pipeline is available it takes
            # raw (cleaned) text directly; otherwise transform with the vectorizer.
            if self.pipeline is not None:
                X_in = [cleaned_text]
            else:
                X_in = self.tfidf_vectorizer.transform([cleaned_text])
            pred_category = self.classifier.predict(X_in)[0]

            def _label(idx):
                if self.label_encoder is not None:
                    return self.label_encoder.inverse_transform([idx])[0]
                classes = getattr(self.classifier, "classes_", None)
                if classes is None:
                    return str(idx)
                return str(classes[idx])

            if hasattr(self.classifier, "predict_proba"):
                probs = self.classifier.predict_proba(X_in)[0]
                confidence = float(np.max(probs))
                top_indices = np.argsort(probs)[::-1][:5]
                top_categories = [
                    {"category": _label(idx), "confidence": float(probs[idx])}
                    for idx in top_indices
                ]
            elif hasattr(self.classifier, "decision_function"):
                scores = self.classifier.decision_function(X_in)[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()
                confidence = float(np.max(probs))
                top_indices = np.argsort(probs)[::-1][:5]
                top_categories = [
                    {"category": _label(idx), "confidence": float(probs[idx])}
                    for idx in top_indices
                ]

        return {
            "predicted_category": pred_category,
            "confidence": confidence,
            "top_categories": top_categories,
            "pipeline_type": self.pipeline_type,
            "cleaned_preview": cleaned_text[:200] + ("..." if len(cleaned_text) > 200 else ""),
            "raw_length": len(raw_text),
            "cleaned_word_count": len(cleaned_text.split()),
        }

    def predict_batch(self, texts: list) -> list:
        """Predict categories for multiple resume texts."""
        return [self.predict(t) for t in texts]
