"""
Feature Engineering Module: TF-IDF & Word2Vec
Authors: Jeel (Person 2 - TF-IDF) & Savan (Person 3 - Word2Vec)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Rule: Fit feature extractors ONLY on the training split to avoid data leakage!
"""

from typing import List, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = Any  # type: ignore


def build_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    min_df: int = 2,
    max_df: float = 0.95,
    stop_words: str = "english",
) -> TfidfVectorizer:
    """
    Creates a scikit-learn TfidfVectorizer configured for resume text classification.
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


def train_word2vec_model(
    tokenized_sentences: List[List[str]],
    vector_size: int = 150,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 20,
    seed: int = 42,
) -> Word2Vec:
    """
    Trains Word2Vec skip-gram/cbow model strictly on the training corpus.
    """
    try:
        from gensim.models import Word2Vec as W2VModel
    except ImportError:
        raise ImportError(
            "gensim is required to train Word2Vec embeddings. Please install it using: pip install gensim"
        )

    model = W2VModel(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    return model


def get_document_embedding(
    tokens: List[str],
    w2v_model: Word2Vec,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Generates a document-level embedding from token lists using pooling (mean or max).
    If no tokens exist in the vocabulary, returns a zero vector.
    """
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.vector_size)

    if pooling == "mean":
        return np.mean(vectors, axis=0)
    elif pooling == "max":
        return np.max(vectors, axis=0)
    else:
        raise ValueError(f"Unsupported pooling method: {pooling}. Use 'mean' or 'max'.")
