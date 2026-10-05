"""
P1.4 – P1.9: Complete Exploratory Data Analysis
Author: Poojan (Person 1) & Team NueralX
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Generates all EDA figures and prints analysis summaries.
"""

import os
import sys
import re
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── project paths ───────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

FIGURES_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from src.data_loader import load_resume_csv

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

# ── Load data ───────────────────────────────────────────────────────────
print("=" * 70)
print("P1 COMPLETE EDA — NueralX Resume Classification")
print("=" * 70)

df = load_resume_csv()
print(f"\nLoaded dataset: {df.shape[0]} records, {df.shape[1]} columns")
print(f"Categories: {df['Category'].nunique()}")

# Add text statistics
df["char_count"] = df["Resume_str"].astype(str).apply(len)
df["word_count"] = df["Resume_str"].astype(str).apply(lambda x: len(x.split()))

# ════════════════════════════════════════════════════════════════════════
# P1.4 — Class Distribution
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.4: Class Distribution")
print("─" * 70)

class_counts = df["Category"].value_counts()
print(class_counts.to_string())
print(f"\nMost frequent:  {class_counts.index[0]} ({class_counts.iloc[0]})")
print(f"Least frequent: {class_counts.index[-1]} ({class_counts.iloc[-1]})")
print(f"Imbalance ratio (max/min): {class_counts.iloc[0] / class_counts.iloc[-1]:.2f}")

# Figure 1: Horizontal bar chart of class distribution
fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("viridis", len(class_counts))
bars = ax.barh(class_counts.index[::-1], class_counts.values[::-1], color=colors)
ax.set_xlabel("Number of Resumes", fontsize=12)
ax.set_ylabel("Category", fontsize=12)
ax.set_title("Resume Count by Category (24 Classes)", fontsize=14, fontweight="bold")
for bar, val in zip(bars, class_counts.values[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_4_class_distribution.png"))
plt.close()
print("  ✓ Saved: p1_4_class_distribution.png")

# Figure 2: Pie chart showing class proportions
fig, ax = plt.subplots(figsize=(10, 10))
wedges, texts = ax.pie(class_counts.values, labels=None, colors=colors, startangle=140)
ax.legend(wedges, [f"{cat} ({cnt})" for cat, cnt in zip(class_counts.index, class_counts.values)],
          title="Categories", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
ax.set_title("Class Proportion Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_4_class_pie_chart.png"))
plt.close()
print("  ✓ Saved: p1_4_class_pie_chart.png")

# ════════════════════════════════════════════════════════════════════════
# P1.5 — Resume Length Distributions
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.5: Resume Length Distributions")
print("─" * 70)

print(f"Word count stats:\n{df['word_count'].describe().to_string()}")
print(f"\nChar count stats:\n{df['char_count'].describe().to_string()}")

# Figure 3: Word count distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(df["word_count"], bins=50, kde=True, color="teal", ax=axes[0])
axes[0].set_title("Resume Word Count Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df["word_count"].median(), color="red", linestyle="--", label=f"Median: {df['word_count'].median():.0f}")
axes[0].legend()

sns.histplot(df["char_count"], bins=50, kde=True, color="coral", ax=axes[1])
axes[1].set_title("Resume Character Count Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Character Count")
axes[1].set_ylabel("Frequency")
axes[1].axvline(df["char_count"].median(), color="red", linestyle="--", label=f"Median: {df['char_count'].median():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_5_length_distributions.png"))
plt.close()
print("  ✓ Saved: p1_5_length_distributions.png")

# Figure 4: Box plot of word counts by category
fig, ax = plt.subplots(figsize=(14, 7))
order = df.groupby("Category")["word_count"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Category", y="word_count", order=order, palette="viridis", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_title("Word Count Distribution per Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Word Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_5_wordcount_by_category.png"))
plt.close()
print("  ✓ Saved: p1_5_wordcount_by_category.png")

# ════════════════════════════════════════════════════════════════════════
# P1.6 — Word Frequency Analysis (Top Unigrams)
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.6: Word Frequency Analysis")
print("─" * 70)

# Simple tokenization (lowercase, alpha only) for frequency analysis
def simple_tokenize(text):
    """Lowercase, split, keep alpha tokens of length >= 2."""
    if not isinstance(text, str):
        return []
    return [w.lower() for w in text.split() if w.isalpha() and len(w) >= 2]

all_tokens = []
for text in df["Resume_str"]:
    all_tokens.extend(simple_tokenize(text))

word_freq = Counter(all_tokens)
top_30 = word_freq.most_common(30)
print("Top 30 words (before stopword removal):")
for word, count in top_30:
    print(f"  {word:20s} {count:>6d}")

# Figure 5: Top 30 most common words
fig, ax = plt.subplots(figsize=(14, 6))
words, counts = zip(*top_30)
ax.barh(list(reversed(words)), list(reversed(counts)), color=sns.color_palette("magma", 30))
ax.set_xlabel("Frequency", fontsize=12)
ax.set_title("Top 30 Most Frequent Words (Before Stopword Removal)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_6_top30_words.png"))
plt.close()
print("  ✓ Saved: p1_6_top30_words.png")

# After removing common English stopwords
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
except Exception:
    # Fallback minimal stopword list
    stop_words = {"the", "and", "to", "of", "in", "a", "is", "for", "on", "with",
                  "that", "it", "are", "was", "as", "at", "be", "this", "from",
                  "or", "an", "by", "not", "but", "have", "has", "had", "do",
                  "did", "will", "would", "can", "could", "may", "should", "i",
                  "we", "you", "he", "she", "they", "my", "our", "your", "his",
                  "her", "its", "their", "all", "each", "which", "what", "when",
                  "where", "how", "who", "whom", "been", "being", "were", "am"}

filtered_tokens = [t for t in all_tokens if t not in stop_words]
filtered_freq = Counter(filtered_tokens)
top_30_filtered = filtered_freq.most_common(30)

print("\nTop 30 words (after stopword removal):")
for word, count in top_30_filtered:
    print(f"  {word:20s} {count:>6d}")

# Figure 6: Top 30 words after stopword removal
fig, ax = plt.subplots(figsize=(14, 6))
words_f, counts_f = zip(*top_30_filtered)
ax.barh(list(reversed(words_f)), list(reversed(counts_f)), color=sns.color_palette("crest", 30))
ax.set_xlabel("Frequency", fontsize=12)
ax.set_title("Top 30 Most Frequent Words (After Stopword Removal)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_6_top30_words_no_stopwords.png"))
plt.close()
print("  ✓ Saved: p1_6_top30_words_no_stopwords.png")

# ════════════════════════════════════════════════════════════════════════
# P1.7 — WordClouds (Overall + Top 6 Categories)
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.7: WordCloud Generation")
print("─" * 70)

try:
    from wordcloud import WordCloud

    # Overall WordCloud
    all_text = " ".join(df["Resume_str"].astype(str).tolist())
    wc = WordCloud(width=1200, height=600, max_words=200, background_color="white",
                   colormap="viridis", stopwords=stop_words, random_state=42)
    wc.generate(all_text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("WordCloud — All Resumes", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "p1_7_wordcloud_overall.png"))
    plt.close()
    print("  ✓ Saved: p1_7_wordcloud_overall.png")

    # Per-category WordClouds (top 6 by count)
    top_6_cats = class_counts.index[:6]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, cat in enumerate(top_6_cats):
        cat_text = " ".join(df[df["Category"] == cat]["Resume_str"].astype(str).tolist())
        wc_cat = WordCloud(width=600, height=400, max_words=100, background_color="white",
                           colormap="plasma", stopwords=stop_words, random_state=42)
        wc_cat.generate(cat_text)
        axes[idx].imshow(wc_cat, interpolation="bilinear")
        axes[idx].axis("off")
        axes[idx].set_title(cat, fontsize=13, fontweight="bold")

    plt.suptitle("WordClouds — Top 6 Categories", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "p1_7_wordclouds_top6_categories.png"))
    plt.close()
    print("  ✓ Saved: p1_7_wordclouds_top6_categories.png")

    # Bottom 6 categories (minority classes)
    bottom_6_cats = class_counts.index[-6:]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, cat in enumerate(bottom_6_cats):
        cat_text = " ".join(df[df["Category"] == cat]["Resume_str"].astype(str).tolist())
        wc_cat = WordCloud(width=600, height=400, max_words=100, background_color="white",
                           colormap="inferno", stopwords=stop_words, random_state=42)
        wc_cat.generate(cat_text)
        axes[idx].imshow(wc_cat, interpolation="bilinear")
        axes[idx].axis("off")
        axes[idx].set_title(cat, fontsize=13, fontweight="bold")

    plt.suptitle("WordClouds — Bottom 6 Categories (Minority Classes)", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "p1_7_wordclouds_bottom6_categories.png"))
    plt.close()
    print("  ✓ Saved: p1_7_wordclouds_bottom6_categories.png")

except ImportError:
    print("  ⚠ wordcloud package not installed. Skipping WordCloud figures.")

# ════════════════════════════════════════════════════════════════════════
# P1.8 — N-Gram Analysis (Bigrams and Trigrams)
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.8: N-Gram Analysis (Bigrams & Trigrams)")
print("─" * 70)

def get_ngrams(tokens, n):
    """Generate n-grams from a list of tokens."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# Build bigrams and trigrams from filtered tokens (per resume)
bigram_counter = Counter()
trigram_counter = Counter()

for text in df["Resume_str"]:
    tokens = [t for t in simple_tokenize(text) if t not in stop_words]
    bigram_counter.update(get_ngrams(tokens, 2))
    trigram_counter.update(get_ngrams(tokens, 3))

top_20_bigrams = bigram_counter.most_common(20)
top_20_trigrams = trigram_counter.most_common(20)

print("Top 20 Bigrams:")
for ngram, count in top_20_bigrams:
    print(f"  {ngram:40s} {count:>5d}")

print("\nTop 20 Trigrams:")
for ngram, count in top_20_trigrams:
    print(f"  {ngram:40s} {count:>5d}")

# Figure 7: Top 20 bigrams
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

bi_ngrams, bi_counts = zip(*top_20_bigrams)
axes[0].barh(list(reversed(bi_ngrams)), list(reversed(bi_counts)), color=sns.color_palette("rocket", 20))
axes[0].set_xlabel("Frequency", fontsize=11)
axes[0].set_title("Top 20 Bigrams", fontsize=13, fontweight="bold")

tri_ngrams, tri_counts = zip(*top_20_trigrams)
axes[1].barh(list(reversed(tri_ngrams)), list(reversed(tri_counts)), color=sns.color_palette("mako", 20))
axes[1].set_xlabel("Frequency", fontsize=11)
axes[1].set_title("Top 20 Trigrams", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_8_ngrams_bigrams_trigrams.png"))
plt.close()
print("  ✓ Saved: p1_8_ngrams_bigrams_trigrams.png")

# ════════════════════════════════════════════════════════════════════════
# P1.9 — Class-Wise Vocabulary Analysis
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.9: Class-Wise Vocabulary Analysis")
print("─" * 70)

# Compute unique vocabulary per class
class_vocab = {}
for cat in sorted(df["Category"].unique()):
    cat_tokens = []
    for text in df[df["Category"] == cat]["Resume_str"]:
        cat_tokens.extend([t for t in simple_tokenize(text) if t not in stop_words])
    class_vocab[cat] = set(cat_tokens)

# Vocabulary size per class
vocab_sizes = {cat: len(vocab) for cat, vocab in class_vocab.items()}
print("Vocabulary sizes per category (unique words, excl. stopwords):")
for cat in sorted(vocab_sizes, key=vocab_sizes.get, reverse=True):
    print(f"  {cat:25s} {vocab_sizes[cat]:>6d}")

# Figure 8: Vocabulary size by category
fig, ax = plt.subplots(figsize=(14, 7))
sorted_cats = sorted(vocab_sizes, key=vocab_sizes.get, reverse=True)
ax.barh(list(reversed(sorted_cats)), [vocab_sizes[c] for c in reversed(sorted_cats)],
        color=sns.color_palette("flare", len(sorted_cats)))
ax.set_xlabel("Unique Vocabulary Size", fontsize=12)
ax.set_title("Unique Vocabulary Size per Category (After Stopword Removal)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_9_vocab_size_by_category.png"))
plt.close()
print("  ✓ Saved: p1_9_vocab_size_by_category.png")

# Category-specific discriminative words (TF-IDF style: frequent in class, rare overall)
print("\nTop 10 distinctive words per category (high class-frequency, low global-frequency ratio):")
global_freq = Counter(filtered_tokens)
total_global = len(filtered_tokens)

for cat in sorted(df["Category"].unique()):
    cat_tokens_list = []
    for text in df[df["Category"] == cat]["Resume_str"]:
        cat_tokens_list.extend([t for t in simple_tokenize(text) if t not in stop_words])
    cat_freq = Counter(cat_tokens_list)
    total_cat = len(cat_tokens_list)

    # Score: (freq_in_class / total_class_tokens) / (freq_overall / total_global_tokens)
    scores = {}
    for word, cnt in cat_freq.items():
        if cnt >= 5 and global_freq[word] >= 5:  # minimum frequency filter
            class_tf = cnt / total_cat
            global_tf = global_freq[word] / total_global
            scores[word] = class_tf / global_tf

    top_distinctive = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_words_str = ", ".join([f"{w}({s:.1f}x)" for w, s in top_distinctive])
    print(f"  {cat:25s} → {top_words_str}")

# ════════════════════════════════════════════════════════════════════════
# P1.9 bonus — Correlation heatmap (word count vs. category)
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
cat_stats = df.groupby("Category").agg(
    mean_words=("word_count", "mean"),
    std_words=("word_count", "std"),
    count=("word_count", "count"),
).sort_values("mean_words", ascending=False)

ax.bar(range(len(cat_stats)), cat_stats["mean_words"],
       yerr=cat_stats["std_words"], capsize=3,
       color=sns.color_palette("coolwarm", len(cat_stats)), alpha=0.85)
ax.set_xticks(range(len(cat_stats)))
ax.set_xticklabels(cat_stats.index, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Mean Word Count (±1 SD)", fontsize=12)
ax.set_title("Mean Resume Length by Category", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "p1_9_mean_wordcount_by_category.png"))
plt.close()
print("  ✓ Saved: p1_9_mean_wordcount_by_category.png")

# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("P1.4 – P1.9 EDA COMPLETE")
print(f"All figures saved to: {FIGURES_DIR}")
print("=" * 70)
