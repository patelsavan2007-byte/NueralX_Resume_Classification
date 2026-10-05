"""
Evaluation & Error Analysis Module
Author: Jeel (Person 2) & Team NueralX
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Rule: Report Macro-F1 alongside Accuracy and inspect wrong predictions.
"""

import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Computes standard classification metrics with special emphasis on Macro-F1.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_and_save_confusion_matrix(
    y_true, y_pred, labels: List[str], save_path: str = "reports/figures/confusion_matrix.png"
):
    """
    Plots a normalized confusion matrix heatmap and saves it to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Normalized Confusion Matrix across Categories", fontsize=14)
    plt.xlabel("Predicted Category", fontsize=12)
    plt.ylabel("Actual Category", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def extract_error_analysis(
    texts: List[str], y_true: List[str], y_pred: List[str], top_n: int = 15
) -> pd.DataFrame:
    """
    Finds misclassified samples and returns a summary DataFrame with text snippet.
    """
    records = []
    for text, actual, pred in zip(texts, y_true, y_pred):
        if actual != pred:
            records.append({
                "Actual_Category": actual,
                "Predicted_Category": pred,
                "Snippet": (str(text)[:200] + "...").replace("\n", " "),
            })

    error_df = pd.DataFrame(records)
    return error_df.head(top_n)
