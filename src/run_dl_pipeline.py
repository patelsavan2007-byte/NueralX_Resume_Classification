"""
Full DL Training Runner — Savan (Person 3)
Executes: P3.1 (Word2Vec) -> P3.2 (Pooling) -> P3.3 (Dense NN + LSTM) -> P3.4 (Early Stopping) -> P3.5 (Evaluation) -> P3.8 (Save)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import clean_resume_text, tokenize_resume
from src.word2vec import train_word2vec_pytorch, get_document_vectors_pytorch, save_word2vec_artifacts
from src.train_dl import train_dense_nn, train_lstm_classifier, save_dl_artifacts

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("=" * 70)
    print("  NueralX Resume Classification — Deep Learning Pipeline")
    print("  Author: Savan (Person 3, Branch: savan_p3)")
    print("=" * 70)

    # ── Load processed splits ──────────────────────────────────────
    train_df = pd.read_csv(os.path.join(PROJ, "data", "processed", "train.csv"))
    val_df = pd.read_csv(os.path.join(PROJ, "data", "processed", "val.csv"))
    test_df = pd.read_csv(os.path.join(PROJ, "data", "processed", "test.csv"))

    print(f"\nDataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Categories: {train_df['Category'].nunique()}")

    # ── Ensure cleaned_text column exists ──────────────────────────
    for df in [train_df, val_df, test_df]:
        if "cleaned_text" not in df.columns:
            df["cleaned_text"] = df["Resume_str"].apply(clean_resume_text)

    # ── Encode labels ──────────────────────────────────────────────
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Category"])
    y_val = le.transform(val_df["Category"])
    y_test = le.transform(test_df["Category"])
    num_classes = len(le.classes_)
    print(f"Label encoder fitted: {num_classes} classes")

    # ── Tokenize ───────────────────────────────────────────────────
    print("\nTokenizing documents...")
    train_tokens = [tokenize_resume(t) for t in train_df["cleaned_text"]]
    val_tokens = [tokenize_resume(t) for t in val_df["cleaned_text"]]
    test_tokens = [tokenize_resume(t) for t in test_df["cleaned_text"]]

    avg_len = np.mean([len(t) for t in train_tokens])
    print(f"Average document length: {avg_len:.0f} tokens")

    # ── P3.1: Train Word2Vec on training corpus ONLY ───────────────
    print("\n" + "=" * 70)
    print("  P3.1: Training Word2Vec (Skip-Gram + Negative Sampling)")
    print("=" * 70)

    w2v_model, vocab = train_word2vec_pytorch(
        train_tokens,
        vector_size=150,
        window=5,
        min_count=2,
        epochs=5,
        batch_size=2048,
        lr=0.005,
    )

    # ── P3.2: Document-level embeddings via mean pooling ───────────
    print("\n" + "=" * 70)
    print("  P3.2: Creating Document Embeddings (Mean Pooling)")
    print("=" * 70)

    X_train_emb = get_document_vectors_pytorch(train_tokens, w2v_model, vocab, pooling="mean")
    X_val_emb = get_document_vectors_pytorch(val_tokens, w2v_model, vocab, pooling="mean")
    X_test_emb = get_document_vectors_pytorch(test_tokens, w2v_model, vocab, pooling="mean")

    print(f"Embedding shapes: Train={X_train_emb.shape}, Val={X_val_emb.shape}, Test={X_test_emb.shape}")

    # ── P3.3 + P3.4 + P3.5: Train Dense NN ────────────────────────
    print("\n" + "=" * 70)
    print("  P3.3/P3.4/P3.5: Training Dense Neural Network")
    print("=" * 70)

    dense_results = train_dense_nn(
        X_train_emb, y_train,
        X_val_emb, y_val,
        hidden_layers=(256, 128, 64),
        max_iter=100,
        lr=1e-3,
    )

    # ── P3.3 + P3.4 + P3.5: Train LSTM ────────────────────────────
    print("\n" + "=" * 70)
    print("  P3.3/P3.4/P3.5: Training Bidirectional LSTM Classifier")
    print("=" * 70)

    lstm_results = train_lstm_classifier(
        train_tokens, y_train,
        val_tokens, y_val,
        w2v_model, vocab,
        num_classes=num_classes,
        embed_dim=150,
        hidden_dim=128,
        max_len=250,
        epochs=10,
        batch_size=64,
        lr=1e-3,
    )

    # ── Compare DL models ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DL MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12}")
    print("-" * 55)
    print(f"{'Dense NN (MLP)':<20} {dense_results['metrics']['accuracy']:>10.4f} "
          f"{dense_results['metrics']['macro_f1']:>10.4f} "
          f"{dense_results['metrics']['weighted_f1']:>12.4f}")
    print(f"{'BiLSTM':<20} {lstm_results['metrics']['accuracy']:>10.4f} "
          f"{lstm_results['metrics']['macro_f1']:>10.4f} "
          f"{lstm_results['metrics']['weighted_f1']:>12.4f}")

    # Select best DL model
    if dense_results["metrics"]["macro_f1"] >= lstm_results["metrics"]["macro_f1"]:
        best_dl = dense_results
        best_type = "dense_nn"
        print(f"\n-> Best DL Model: Dense NN (Macro-F1 = {best_dl['metrics']['macro_f1']:.4f})")
    else:
        best_dl = lstm_results
        best_type = "lstm"
        print(f"\n-> Best DL Model: BiLSTM (Macro-F1 = {best_dl['metrics']['macro_f1']:.4f})")

    # ── P3.8: Save all artifacts ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  P3.8: Saving All Artifacts")
    print("=" * 70)

    save_dir = os.path.join(PROJ, "models")
    save_dl_artifacts(w2v_model, vocab, best_dl["model"], le,
                      classifier_type=best_type, save_dir=save_dir)

    # Also save Dense NN specifically for the prediction pipeline
    if best_type != "dense_nn":
        import joblib
        joblib.dump(dense_results["model"],
                    os.path.join(save_dir, "dense_nn_classifier.joblib"))
        print("Also saved Dense NN as fallback classifier.")

    print("\n" + "=" * 70)
    print("  ALL DL TASKS COMPLETE")
    print("=" * 70)

    return {
        "dense_results": dense_results,
        "lstm_results": lstm_results,
        "best_type": best_type,
        "label_encoder": le,
    }


if __name__ == "__main__":
    main()
