"""
Deep Learning Training Pipeline — Full Implementation
Author: Savan (Person 3)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Tasks Covered:
  P3.1 — Train Word2Vec on training corpus only (via src/word2vec.py)
  P3.2 — Document representation via mean pooling (via src/word2vec.py)
  P3.3 — Build Dense NN + LSTM classifiers over Word2Vec embeddings
  P3.4 — Validation tracking with early stopping
  P3.5 — Full DL evaluation with same metrics as ML models
  P3.8 — Save all artifacts for reproducible inference
"""

import os
import numpy as np
import joblib
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


# ============================================================
#  P3.3 — Option A: Dense NN (MLP) over pooled Word2Vec
# ============================================================

def train_dense_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: Tuple[int, ...] = (256, 128, 64),
    max_iter: int = 100,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """
    P3.3 + P3.4: Train sklearn MLP with early stopping over pooled W2V embeddings.
    """
    print(f"Training Dense NN: layers={hidden_layers}, max_iter={max_iter}, lr={lr}")

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=lr,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=42,
        verbose=True,
    )

    mlp.fit(X_train, y_train)
    val_preds = mlp.predict(X_val)

    metrics = _compute_metrics(y_val, val_preds)
    report_str = classification_report(y_val, val_preds, zero_division=0)

    print("\n" + "=" * 60)
    print("DENSE NN (MLP) — VALIDATION RESULTS")
    print("=" * 60)
    _print_metrics(metrics)
    print(f"Training iterations: {mlp.n_iter_}")
    print(f"Best validation score: {mlp.best_validation_score_:.4f}")
    print("=" * 60)
    print("\nPer-Class Classification Report:\n" + report_str)

    return {
        "model": mlp,
        "model_type": "dense_nn",
        "metrics": metrics,
        "report_str": report_str,
        "predictions": val_preds,
    }


# ============================================================
#  P3.3 — Option B: LSTM Classifier over Word2Vec sequences
# ============================================================

class ResumeDataset(Dataset):
    """Converts tokenized docs to padded index sequences for LSTM."""

    def __init__(self, tokenized_docs, labels, vocab, max_len=300):
        self.max_len = max_len
        self.sequences = []
        self.labels = labels

        for doc in tokenized_docs:
            indices = [vocab.get(t, 0) for t in doc[:max_len]]
            # Pad to max_len
            padded = indices + [0] * (max_len - len(indices))
            self.sequences.append(padded)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier using pretrained Word2Vec embeddings."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3, pretrained_weights=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
            self.embedding.weight.requires_grad = False  # Freeze embeddings

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        emb = self.embedding(x)                    # (batch, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(emb)     # hidden: (num_layers*2, batch, hidden)
        # Concatenate final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden*2)
        out = self.dropout(hidden_cat)
        return self.fc(out)


def _build_embedding_matrix(w2v_model, vocab, embed_dim):
    """Build embedding matrix from trained Word2Vec model."""
    vocab_size = len(vocab)
    matrix = np.random.normal(0, 0.1, (vocab_size, embed_dim)).astype(np.float32)
    matrix[0] = 0  # PAD token

    w2v_model.eval()
    with torch.no_grad():
        for word, idx in vocab.items():
            if idx < w2v_model.center_embeddings.num_embeddings:
                matrix[idx] = w2v_model.center_embeddings(
                    torch.tensor([idx])
                ).squeeze().cpu().numpy()
    return matrix


def train_lstm_classifier(
    train_tokens, train_labels_encoded,
    val_tokens, val_labels_encoded,
    w2v_model, vocab,
    num_classes: int,
    embed_dim: int = 150,
    hidden_dim: int = 128,
    max_len: int = 250,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """
    P3.3 (Option B): Train a Bidirectional LSTM classifier using Word2Vec embeddings.
    P3.4: Early stopping based on validation macro-F1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LSTM Classifier on: {device}", flush=True)

    # Build datasets
    train_dataset = ResumeDataset(train_tokens, train_labels_encoded, vocab, max_len)
    val_dataset = ResumeDataset(val_tokens, val_labels_encoded, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build embedding matrix from Word2Vec
    embedding_matrix = _build_embedding_matrix(w2v_model, vocab, embed_dim)

    # Build model
    vocab_size = len(vocab)
    model = LSTMClassifier(
        vocab_size, embed_dim, hidden_dim, num_classes,
        pretrained_weights=embedding_matrix,
    ).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    patience = 4
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(device)
                logits = model(seqs)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print(f"  Epoch [{epoch + 1}/{epochs}] — Loss: {avg_loss:.4f}, Val Macro-F1: {val_f1:.4f}", flush=True)

        # P3.4: Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch + 1}. Best Val Macro-F1: {best_f1:.4f}", flush=True)
                break

    # Load best model state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    metrics = _compute_metrics(np.array(all_labels), np.array(all_preds))
    report_str = classification_report(all_labels, all_preds, zero_division=0)

    print("\n" + "=" * 60)
    print("LSTM CLASSIFIER — VALIDATION RESULTS")
    print("=" * 60)
    _print_metrics(metrics)
    print("=" * 60)
    print("\nPer-Class Classification Report:\n" + report_str)

    return {
        "model": model,
        "model_type": "lstm",
        "metrics": metrics,
        "report_str": report_str,
        "predictions": np.array(all_preds),
    }


# ============================================================
#  P3.8 — Save all DL artifacts
# ============================================================

def save_dl_artifacts(
    w2v_model,
    vocab: dict,
    classifier,
    label_encoder: LabelEncoder,
    classifier_type: str = "dense_nn",
    save_dir: str = "models",
):
    """
    P3.8: Save everything needed for reproducible inference:
    - Word2Vec model + vocab
    - Classifier (MLP or LSTM)
    - Label encoder
    """
    from src.word2vec import save_word2vec_artifacts

    os.makedirs(save_dir, exist_ok=True)

    # Save Word2Vec
    save_word2vec_artifacts(w2v_model, vocab, save_dir)

    # Save label encoder
    le_path = os.path.join(save_dir, "label_encoder.joblib")
    joblib.dump(label_encoder, le_path)
    print(f"Saved Label Encoder: {le_path}")

    # Save classifier
    if classifier_type == "dense_nn":
        clf_path = os.path.join(save_dir, "dense_nn_classifier.joblib")
        joblib.dump(classifier, clf_path)
        print(f"Saved Dense NN: {clf_path}")
    elif classifier_type == "lstm":
        clf_path = os.path.join(save_dir, "lstm_classifier.pt")
        torch.save(classifier.state_dict(), clf_path)
        print(f"Saved LSTM model: {clf_path}")


# ============================================================
#  Helper utilities
# ============================================================

def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _print_metrics(m: dict):
    print(f"Accuracy:        {m['accuracy']:.4f}")
    print(f"Macro Precision: {m['macro_precision']:.4f}")
    print(f"Macro Recall:    {m['macro_recall']:.4f}")
    print(f"Macro F1:        {m['macro_f1']:.4f}")
    print(f"Weighted F1:     {m['weighted_f1']:.4f}")
