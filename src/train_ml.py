"""
Classical Machine Learning Training Pipeline
Author: Jeel (Person 2)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)
"""

import os
import joblib
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score


def train_and_evaluate_baselines(
    X_train, y_train, X_val, y_val
) -> Dict[str, Any]:
    """
    Trains standard classical ML classifiers on TF-IDF features and compares Macro-F1.
    """
    models = {
        "Multinomial_NB": MultinomialNB(alpha=0.1),
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear_SVM": SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, max_iter=1000, class_weight="balanced", random_state=42),
    }

    results = {}
    best_model_name = None
    best_macro_f1 = -1.0
    best_model = None

    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        macro_f1 = f1_score(y_val, preds, average="macro")
        weighted_f1 = f1_score(y_val, preds, average="weighted")
        print(f"-> {name} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

        results[name] = {
            "model": clf,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report": classification_report(y_val, preds, output_dict=True),
        }

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name
            best_model = clf

    print(f"\nBest Classical Model: {best_model_name} with Macro-F1 = {best_macro_f1:.4f}")
    return {"results": results, "best_model_name": best_model_name, "best_model": best_model}


def save_ml_pipeline(pipeline: Pipeline, filepath: str):
    """
    Serializes fitted ML pipeline to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Saved model pipeline to {filepath}")
