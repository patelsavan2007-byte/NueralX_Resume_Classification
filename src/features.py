"""
Feature Engineering Module: TF-IDF & Word2Vec
Authors: Jeel (Person 2 - TF-IDF) & Savan (Person 3 - Word2Vec via PyTorch)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Rule: Fit feature extractors ONLY on the training split to avoid data leakage!
"""

from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    min_df: int = 2,
    max_df: float = 0.95,
    stop_words: str = "english",
) -> TfidfVectorizer:
    """
    Creates a TfidfVectorizer configured for resume text classification.
    Supports unigrams + bigrams for richer phrase capture.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        sublinear_tf=True,
    )


# Word2Vec functionality has been moved to src/word2vec.py (PyTorch-based)
# to avoid gensim dependency issues on Python 3.14.
# Import from src.word2vec for Word2Vec training and document embedding.
