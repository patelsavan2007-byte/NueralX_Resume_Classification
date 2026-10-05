"""
P3.5 / P3.6 — Deep Learning Evaluation & Final Model Selection
Author: Savan (Person 3, branch: savan_p3)

Purpose
-------
src/run_dl_pipeline.py trains Word2Vec + Dense NN + BiLSTM and prints metrics to
stdout, but the metrics were never persisted. This module reloads the serialized
artifacts (P3.8) and recomputes metrics on the validation and test splits so that:

  * P3.5 is satisfied  -> identical metric set as the classical ML models
                          (accuracy, macro P/R/F1, weighted F1)
  * P3.8 is verified  -> artifacts demonstrably round-trip from disk
  * P3.6 is satisfied  -> head-to-head DL vs classical ML comparison

The LSTM architecture is reconstructed from the saved ``state_dict`` tensor shapes,
so no configuration side-file is required for the already-trained checkpoint.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(PROJ, "models")
REPORTS = os.path.join(PROJ, "reports")


# ============================================================
#  Metric helpers
# ============================================================

def compute_metrics(y_true, y_pred) -> dict:
    """Same metric set used by the classical ML pipeline (src/run_p2_pipeline.py)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def print_row(name, m):
    print(
        f"{name:<26} {m['accuracy']:>9.4f} {m['macro_precision']:>8.4f} "
        f"{m['macro_recall']:>8.4f} {m['macro_f1']:>8.4f} {m['weighted_f1']:>9.4f}"
    )


def print_header(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"{'Model':<26} {'Accuracy':>9} {'MacroP':>8} {'MacroR':>8} {'MacroF1':>8} {'WtF1':>9}")
    print("-" * 78)


# ============================================================
#  Loading
# ============================================================

def load_splits():
    splits = {}
    for name in ("train", "val", "test"):
        path = os.path.join(PROJ, "data", "processed", f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}. Run scripts/p1_preprocessing_split.py first.")
        splits[name] = pd.read_csv(path)
    return splits


def _tokenize_df(df):
    """Return a list of token-string lists (needed by the Word2Vec pooling path)."""
    from src.preprocessing import clean_resume_text, tokenize_resume

    if "cleaned_text" not in df.columns:
        df = df.copy()
        df["cleaned_text"] = df["Resume_str"].astype(str).apply(clean_resume_text)

    return [tokenize_resume(t) for t in df["cleaned_text"]]


def build_tokenized(df, vocab):
    """
    Map tokens -> vocab index ids, exactly as ResumeDataset does during LSTM training.

    Note: index 0 is the <PAD> token, so OOV words collapse to 0 and are masked by
    the embedding layer's padding_idx.
    """
    return [[vocab.get(t, 0) for t in tokens] for tokens in _tokenize_df(df)]


def rebuild_lstm(state_dict, num_classes=None):
    """
    P3.8 — Reconstruct LSTMClassifier purely from checkpoint tensor shapes.

    state_dict keys/shapes:
        embedding.weight        -> (vocab_size, embed_dim)
        lstm.weight_ih_l0       -> (4*hidden, embed_dim)
        lstm.weight_hh_l0       -> (4*hidden, hidden)
        fc.weight               -> (num_classes, 2*hidden)
    """
    import torch
    from src.train_dl import LSTMClassifier

    emb_w = state_dict["embedding.weight"]
    vocab_size, embed_dim = emb_w.shape

    ih = state_dict["lstm.weight_ih_l0"]
    # hidden_dim comes from the recurrent-hidden weight, not weight_ih (which is embed_dim).
    hidden_dim = state_dict["lstm.weight_hh_l0"].shape[1]

    fc_w = state_dict["fc.weight"]
    inferred_classes = fc_w.shape[0]
    if num_classes is None:
        num_classes = inferred_classes
    if inferred_classes != num_classes:
        print(f"  Warning: fc implies {inferred_classes} classes, label encoder says {num_classes}")

    num_layers = 1 + max(
        int(k.split("weight_ih_l")[1][0]) for k in state_dict if k.startswith("lstm.weight_ih_l")
    )

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=0.0,  # inference: dropout inactive
        pretrained_weights=None,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"  LSTM rebuilt from checkpoint: vocab_size={vocab_size}, embed_dim={embed_dim}, "
        f"hidden_dim={hidden_dim}, num_layers={num_layers}, num_classes={num_classes}"
    )
    return model, max_len_from_state(state_dict)


def max_len_from_state(state_dict, default=250):
    return state_dict.get("_max_len", default)


# ============================================================
#  Main evaluation
# ============================================================

def main():
    print("=" * 78)
    print("  NueralX Resume Classification — DL Evaluation & Model Selection (P3.5 / P3.6)")
    print("  Author: Savan (Person 3, branch: savan_p3)")
    print("=" * 78)

    splits = load_splits()
    print(f"\nSplits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    le = joblib.load(os.path.join(MODELS, "label_encoder.joblib"))
    num_classes = len(le.classes_)

    y = {
        name: le.transform(df["Category"])
        for name, df in splits.items()
    }

    results = {}
    dense_preds = {}
    lstm_preds = {}

    # ---------- Dense NN (Word2Vec mean-pooled features) ----------
    dense_path = os.path.join(MODELS, "dense_nn_classifier.joblib")
    w2v_path = os.path.join(MODELS, "word2vec_pytorch.pt")
    vocab_path = os.path.join(MODELS, "word2vec_vocab.json")

    w2v_model = None
    vocab = None
    if os.path.exists(w2v_path) and os.path.exists(vocab_path):
        from src.word2vec import load_word2vec_artifacts
        w2v_model, vocab = load_word2vec_artifacts(MODELS)
        print(f"\nWord2Vec artifacts loaded (vocab size = {len(vocab)}).")

    if os.path.exists(dense_path):
        from src.word2vec import get_document_vectors_pytorch

        print("\n" + "-" * 78)
        print("  Dense NN (MLP over mean-pooled Word2Vec vectors)")
        print("-" * 78)
        dense = joblib.load(dense_path)
        for split in ("val", "test"):
            tokens = _tokenize_df(splits[split])
            X = get_document_vectors_pytorch(tokens, w2v_model, vocab, pooling="mean")
            m = compute_metrics(y[split], dense.predict(X))
            results.setdefault("dense_nn", {})[split] = m
            dense_preds[split] = dense.predict(X)
            print_row(f"Dense NN [{split}]", m)

    # ---------- BiLSTM ----------
    lstm_path = os.path.join(MODELS, "lstm_classifier.pt")
    if os.path.exists(lstm_path) and vocab is not None:
        import torch

        print("\n" + "-" * 78)
        print("  Bidirectional LSTM (Word2Vec-initialised, frozen embeddings)")
        print("-" * 78)

        state = torch.load(lstm_path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("_")}
        lstm, max_len = rebuild_lstm(state, num_classes)

        for split in ("val", "test"):
            ids = build_tokenized(splits[split], vocab)
            preds = []
            with torch.no_grad():
                for seq in ids:
                    padded = seq[:max_len] + [0] * (max_len - min(len(seq), max_len))
                    logits = lstm(torch.tensor([padded], dtype=torch.long))
                    preds.append(int(logits.argmax(dim=1).item()))
            m = compute_metrics(y[split], np.array(preds))
            results.setdefault("bilstm", {})[split] = m
            lstm_preds[split] = np.array(preds)
            print_row(f"BiLSTM [{split}]", m)

    # ---------- Classical ML baseline (P2 artifacts), for P3.6 ----------
    ml_results = {}
    pipeline_path = os.path.join(MODELS, "final_pipeline.joblib")
    if os.path.exists(pipeline_path):
        from src.preprocessing import clean_resume_text

        print("\n" + "-" * 78)
        print("  Classical ML baseline (Person 2 artifacts, jeel_p2)")
        print("-" * 78)
        pipe = joblib.load(pipeline_path)
        for split in ("val", "test"):
            cleaned = splits[split]["Resume_str"].astype(str).apply(clean_resume_text)
            pred_labels = pipe.predict(cleaned.tolist())
            truth_labels = le.inverse_transform(y[split])
            m = compute_metrics(truth_labels, pred_labels)
            ml_results[split] = m
            print_row(f"Classical ML [{split}]", m)

    # ---------- P3.6 selection ----------
    print("\n" + "=" * 78)
    print("  P3.6 — FINAL MODEL SELECTION (Macro-F1 on the held-out TEST split)")
    print("=" * 78)

    candidates = []
    for name, splits_m in results.items():
        if "test" in splits_m:
            candidates.append((name, splits_m["test"]["macro_f1"]))
    for name, m in ml_results.items():
        if name == "test":
            candidates.append((f"classical_ml_{meta_best_name()}", m["macro_f1"]))

    candidates.sort(key=lambda kv: kv[1], reverse=True)
    for name, score in candidates:
        print(f"  {name:<28} Macro-F1 = {score:.4f}")

    if not candidates:
        print("  (no evaluable models)")
        return results

    winner, winner_score = candidates[0]
    print(f"\n  -> Selected model: {winner}  (test Macro-F1 = {winner_score:.4f})")

    # Per-class report for the winning DL model, on the test split
    winner_preds = {"dense_nn": dense_preds.get("test"), "bilstm": lstm_preds.get("test")}.get(winner)
    if winner_preds is not None:
        per_class = classification_report(
            le.inverse_transform(y["test"]),
            le.inverse_transform(winner_preds),
            zero_division=0,
        )
        print("\n" + "-" * 78)
        print(f"  Per-class report — {winner} (test set)")
        print("-" * 78)
        print(per_class)

    # ---------- persist ----------
    out = {
        "note": "Metrics recomputed from serialized artifacts (P3.8 round-trip verification).",
        "dl_results": results,
        "classical_ml_results": ml_results,
        "selected_model": winner,
        "selected_test_macro_f1": winner_score,
        "candidate_ranking_test_macro_f1": {n: s for n, s in candidates},
    }
    os.makedirs(REPORTS, exist_ok=True)
    out_path = os.path.join(REPORTS, "dl_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")

    return results


def meta_best_name():
    path = os.path.join(MODELS, "training_metadata.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("best_model", "classical_ml")
        except Exception:
            pass
    return "classical_ml"


if __name__ == "__main__":
    main()