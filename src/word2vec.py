"""
Word2Vec Implementation using PyTorch (Skip-Gram with Negative Sampling)
Author: Savan (Person 3)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Task P3.1: Train Word2Vec embeddings on training corpus only.
Task P3.2: Generate document-level representations via mean/max pooling.

Uses optimized vectorized PyTorch Skip-Gram with Negative Sampling.
Guarantees zero data leakage by fitting vocabulary and embeddings exclusively on training data.
"""

import os
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class SkipGramModel(nn.Module):
    """PyTorch Skip-Gram Word2Vec model with Center and Context embeddings."""

    def __init__(self, vocab_size: int, embedding_dim: int = 150):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Initialize embeddings uniformly
        nn.init.uniform_(self.center_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.constant_(self.context_embeddings.weight, 0)
        # Ensure padding stays zero
        with torch.no_grad():
            self.center_embeddings.weight[0].fill_(0)
            self.context_embeddings.weight[0].fill_(0)


def train_word2vec_pytorch(
    tokenized_docs: List[List[str]],
    vector_size: int = 150,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 5,
    batch_size: int = 2048,
    lr: float = 0.005,
    neg_samples: int = 5,
) -> Tuple[SkipGramModel, Dict[str, int]]:
    """
    P3.1: Train Word2Vec (Skip-Gram with Negative Sampling) using PyTorch.
    Trained ONLY on the training corpus to prevent data leakage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Word2Vec on device: {device}")

    # 1. Build vocabulary from training tokens
    word_counts = Counter()
    for doc in tokenized_docs:
        word_counts.update(doc)

    filtered_words = [w for w, c in word_counts.items() if c >= min_count]
    vocab = {"<PAD>": 0}
    for i, w in enumerate(filtered_words, start=1):
        vocab[w] = i

    vocab_size = len(vocab)
    print(f"  Vocabulary size: {vocab_size:,} words (filtered by min_count >= {min_count})")

    # 2. Vectorized Skip-Gram Pair Generation
    centers = []
    contexts = []
    max_doc_tokens = 150  # Cap doc tokens for efficient skip-gram pairs

    for doc in tokenized_docs:
        indices = [vocab[w] for w in doc[:max_doc_tokens] if w in vocab]
        doc_len = len(indices)
        if doc_len < 2:
            continue
        for i, center_idx in enumerate(indices):
            left = max(0, i - window)
            right = min(doc_len, i + window + 1)
            for j in range(left, right):
                if i != j:
                    centers.append(center_idx)
                    contexts.append(indices[j])

    total_pairs = len(centers)
    print(f"  Generated {total_pairs:,} skip-gram pairs from training documents.")

    if total_pairs == 0:
        raise ValueError("No skip-gram pairs generated. Check tokenized documents.")

    # Convert to contiguous PyTorch Tensors for high-throughput batching
    centers_tensor = torch.tensor(centers, dtype=torch.long)
    contexts_tensor = torch.tensor(contexts, dtype=torch.long)
    dataset = TensorDataset(centers_tensor, contexts_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    # 3. Model & Optimizer
    model = SkipGramModel(vocab_size, vector_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 4. Training Loop (Negative Sampling Loss)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for center_batch, context_batch in loader:
            center_batch = center_batch.to(device)
            context_batch = context_batch.to(device)
            curr_batch_len = center_batch.size(0)

            # Positive dot products: (batch, dim) * (batch, dim) -> (batch,)
            center_emb = model.center_embeddings(center_batch)
            context_emb = model.context_embeddings(context_batch)
            pos_score = (center_emb * context_emb).sum(dim=-1)

            # Negative sampling: sample random words from vocabulary
            neg_context = torch.randint(
                1, vocab_size, (curr_batch_len, neg_samples), device=device
            )
            neg_emb = model.context_embeddings(neg_context)  # (batch, neg_samples, dim)

            # Center emb unsqueezed: (batch, 1, dim)
            # Dot product with negative samples: (batch, neg_samples)
            neg_score = (center_emb.unsqueeze(1) * neg_emb).sum(dim=-1)

            # Negative Sampling Objective: -log(sigmoid(pos)) - sum(log(sigmoid(-neg)))
            pos_loss = F.logsigmoid(pos_score)
            neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)
            loss = -(pos_loss + neg_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch [{epoch + 1}/{epochs}] — Average Skip-Gram Loss: {avg_loss:.4f}", flush=True)

    print("Word2Vec training completed successfully.", flush=True)
    return model, vocab


def get_document_vectors_pytorch(
    tokenized_docs: List[List[str]],
    w2v_model: SkipGramModel,
    vocab: Dict[str, int],
    pooling: str = "mean",
) -> np.ndarray:
    """
    P3.2: Convert word vectors to document-level representations via mean/max pooling.
    Fast vectorized numpy implementation over extracted embedding matrix.
    """
    w2v_model.eval()
    vec_size = w2v_model.center_embeddings.embedding_dim
    emb_weights = w2v_model.center_embeddings.weight.detach().cpu().numpy()

    doc_vectors = np.zeros((len(tokenized_docs), vec_size), dtype=np.float32)
    zero_count = 0

    for i, tokens in enumerate(tokenized_docs):
        indices = [vocab[t] for t in tokens if t in vocab and vocab[t] > 0]
        if not indices:
            zero_count += 1
            continue

        vecs = emb_weights[indices]
        if pooling == "mean":
            doc_vectors[i] = np.mean(vecs, axis=0)
        elif pooling == "max":
            doc_vectors[i] = np.max(vecs, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}. Use 'mean' or 'max'.")

    if zero_count > 0:
        print(f"  Info: {zero_count}/{len(tokenized_docs)} documents had no vocabulary matches.")

    return doc_vectors


def save_word2vec_artifacts(
    model: SkipGramModel,
    vocab: Dict[str, int],
    save_dir: str = "models",
):
    """Save Word2Vec model weights and vocabulary."""
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "word2vec_pytorch.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size": model.center_embeddings.num_embeddings,
        "embedding_dim": model.center_embeddings.embedding_dim,
    }, model_path)

    vocab_path = os.path.join(save_dir, "word2vec_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    print(f"Saved Word2Vec weights to: {model_path}")
    print(f"Saved Word2Vec vocabulary to: {vocab_path}")


def load_word2vec_artifacts(save_dir: str = "models"):
    """Load saved Word2Vec model and vocabulary."""
    model_path = os.path.join(save_dir, "word2vec_pytorch.pt")
    vocab_path = os.path.join(save_dir, "word2vec_vocab.json")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = SkipGramModel(checkpoint["vocab_size"], checkpoint["embedding_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return model, vocab
