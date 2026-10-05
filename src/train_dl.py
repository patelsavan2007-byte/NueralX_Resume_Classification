"""
Deep Learning Training Pipeline
Author: Savan (Person 3)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Rule: Word2Vec is an embedding/representation technique, not a classifier.
Combine pooled Word2Vec vectors with a Multi-Layer Perceptron (Dense NN) or LSTM/GRU.
"""

import numpy as np
from typing import Dict, Any
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score


def train_dense_neural_net(
    X_train_emb: np.ndarray,
    y_train: np.ndarray,
    X_val_emb: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Trains a Multi-Layer Perceptron (Dense Neural Network) classifier over pooled Word2Vec embeddings.
    Provides a solid, fast, scikit-learn compatible neural architecture without external heavy dependencies.
    """
    print("Training Dense Neural Network on Word2Vec embeddings...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        random_state=42,
        verbose=False,
    )

    mlp.fit(X_train_emb, y_train)
    preds = mlp.predict(X_val_emb)

    macro_f1 = f1_score(y_val, preds, average="macro")
    weighted_f1 = f1_score(y_val, preds, average="weighted")

    print(f"Dense NN | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
    return {
        "model": mlp,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": classification_report(y_val, preds, output_dict=True),
    }
