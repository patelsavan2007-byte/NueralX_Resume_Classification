"""
Inference Pipeline Module
Author: Savan (Person 3)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Rule: Pipeline must apply exact same preprocessing and feature extraction used during training.
"""

import os
from typing import Dict, Any, Union
import joblib
from src.preprocessing import clean_resume_text
from src.data_loader import extract_text_from_pdf

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_pipeline.joblib")


class ResumeClassifierPipeline:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        if os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str):
        self.pipeline = joblib.load(model_path)
        print(f"Loaded classification pipeline from {model_path}")

    def predict(self, input_data: Union[str, os.PathLike], is_pdf: bool = False) -> Dict[str, Any]:
        """
        Takes raw string or PDF path, cleans text, and runs inference.
        """
        if is_pdf or str(input_data).lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(str(input_data))
        else:
            raw_text = str(input_data)

        cleaned_text = clean_resume_text(raw_text)

        if not cleaned_text:
            return {
                "predicted_category": "UNKNOWN",
                "confidence": 0.0,
                "cleaned_preview": "",
                "error": "Resume text is empty or could not be parsed.",
            }

        if self.pipeline is None:
            # Fallback placeholder if weights are not trained yet
            return {
                "predicted_category": "MODEL_NOT_YET_TRAINED",
                "confidence": 0.0,
                "cleaned_preview": cleaned_text[:200] + "...",
                "raw_length": len(raw_text),
            }

        pred_class = self.pipeline.predict([cleaned_text])[0]
        confidence = 1.0

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([cleaned_text])[0]
            confidence = float(max(probs))

        return {
            "predicted_category": pred_class,
            "confidence": confidence,
            "cleaned_preview": cleaned_text[:200] + "...",
            "raw_length": len(raw_text),
        }
