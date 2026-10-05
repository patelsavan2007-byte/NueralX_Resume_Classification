"""
P1.10 – P1.12: Finalize Preprocessing, Verify Tech Tokens, Split & Export
Author: Poojan (Person 1) & Team NueralX
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Steps:
  P1.10 — Finalize clean_resume_text() with stopword removal & lemmatization
  P1.11 — Verify technical token preservation (Python, C++, SQL, AWS, etc.)
  P1.12 — Run stratified split (70/15/15), export to data/processed/,
           and generate the handoff documentation for P2 and P3
"""

import os
import sys
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.data_loader import load_resume_csv, stratified_split
from src.preprocessing import clean_resume_text, PROTECTED_TERMS

print("=" * 70)
print("P1.10–P1.12: PREPROCESSING, TOKEN VERIFICATION & SPLIT")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# P1.10 — Finalize clean_resume_text with stopword + lemmatization
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.10: Finalizing clean_resume_text() with stopword removal & lemmatization")
print("─" * 70)

# We keep stopword removal OUT of clean_resume_text() itself because:
#   - TF-IDF (P2) has its own stop_words='english' parameter
#   - Word2Vec (P3) may benefit from different stopword handling
# Instead, we provide a dedicated post-cleaning function for EDA and optional use.

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def postprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Optional post-processing step applied AFTER clean_resume_text().
    - remove_stopwords: remove English stopwords
    - lemmatize: apply WordNet lemmatization
    
    NOTE: This is intentionally separate from clean_resume_text() so that
    P2 (TF-IDF) and P3 (Word2Vec) can make their own stopword/lemma decisions.
    """
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# Load dataset
df = load_resume_csv()
print(f"Loaded {len(df)} records")

# Apply core cleaning
print("Applying clean_resume_text() to all records...")
df["cleaned_text"] = df["Resume_str"].astype(str).apply(clean_resume_text)

# Show sample before/after
print("\n--- Sample Cleaning Result ---")
sample_idx = 0
orig = str(df["Resume_str"].iloc[sample_idx])[:300]
cleaned = df["cleaned_text"].iloc[sample_idx][:300]
print(f"ORIGINAL (first 300 chars):\n  {orig}")
print(f"\nCLEANED  (first 300 chars):\n  {cleaned}")

# Show postprocess example
post = postprocess_text(df["cleaned_text"].iloc[sample_idx])[:300]
print(f"\nPOST-PROCESSED (stopwords removed + lemmatized, first 300 chars):\n  {post}")

# ════════════════════════════════════════════════════════════════════════
# P1.11 — Verify Technical Token Preservation
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.11: Technical Token Preservation Verification")
print("─" * 70)

test_cases = [
    ("C++", "Experienced in C++ and data structures"),
    ("C#", "Developed enterprise apps using C# and .NET framework"),
    (".NET", "Built REST APIs with ASP.NET Core and .NET 6"),
    ("CI/CD", "Implemented CI/CD pipelines with Jenkins"),
    ("Node.js", "Backend development with Node.js and Express"),
    ("Python", "Machine learning engineer proficient in Python and TensorFlow"),
    ("SQL", "Database administration with SQL Server and PostgreSQL"),
    ("AWS", "Cloud architect certified in AWS and Azure services"),
    ("TensorFlow", "Deep learning models using TensorFlow and PyTorch"),
    ("React.js", "Frontend development with React.js and TypeScript"),
    ("Java", "Enterprise Java development with Spring Boot"),
    ("JavaScript", "Full-stack JavaScript developer"),
    ("HTML/CSS", "Responsive web design with HTML/CSS"),
    ("Docker", "Container orchestration using Docker and Kubernetes"),
    ("NLP", "Natural Language Processing and NLP research"),
]

print(f"\n{'Term':<15} {'Preserved?':<12} {'Cleaned Output (snippet)'}")
print("─" * 80)

all_passed = True
for term, sample_text in test_cases:
    cleaned = clean_resume_text(sample_text)
    # Map each term to its expected form after cleaning
    expected_map = {
        "C++": "cplusplus", "C#": "csharp", ".NET": "dotnet",
        "CI/CD": "cicd", "Node.js": "nodejs", "React.js": "reactjs",
        "Vue.js": "vuejs", "HTML/CSS": "htmlcss", "ASP.NET": "aspnet",
    }
    expected = expected_map.get(term, term.lower())
    preserved = expected in cleaned
    status = "YES" if preserved else "NO "
    if not preserved:
        all_passed = False
    print(f"  {term:<13} {status:<12} {cleaned[:60]}")

if all_passed:
    print("\n[PASS] All technical tokens are preserved through the pipeline!")
else:
    print("\n[WARN] Some tokens may need attention -- see details above.")
    print("   Note: Simple words like Python, SQL, AWS, Java survive by default")
    print("   since they are standard alphanumeric tokens.")

# ════════════════════════════════════════════════════════════════════════
# P1.12 — Stratified Split & Export
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.12: Stratified Train/Val/Test Split & Export")
print("─" * 70)

# Handle the problematic records identified in P1.2
empty_mask = df["Resume_str"].isna() | (df["Resume_str"].astype(str).str.strip() == "")
short_mask = df["Resume_str"].astype(str).apply(lambda x: len(x.split()) < 10)
print(f"Empty Resume_str records: {empty_mask.sum()} → dropping")
print(f"Records < 10 words: {short_mask.sum()} → dropping")

# Drop empty/very short records
df_clean = df[~empty_mask & ~short_mask].copy()
print(f"Records after dropping: {len(df_clean)}")

# Drop exact duplicate texts (keep first occurrence)
dup_mask = df_clean["cleaned_text"].duplicated(keep="first")
print(f"Duplicate cleaned texts: {dup_mask.sum()} → dropping")
df_clean = df_clean[~dup_mask].copy()
print(f"Records after dedup: {len(df_clean)}")

# Perform stratified split
train_df, val_df, test_df = stratified_split(df_clean, random_state=42)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df):>5d} ({len(train_df)/len(df_clean)*100:.1f}%)")
print(f"  Val:   {len(val_df):>5d} ({len(val_df)/len(df_clean)*100:.1f}%)")
print(f"  Test:  {len(test_df):>5d} ({len(test_df)/len(df_clean)*100:.1f}%)")
print(f"  Total: {len(train_df)+len(val_df)+len(test_df):>5d}")

# Verify stratification
print("\nClass distribution verification (sample of 5 classes):")
sample_cats = sorted(df_clean["Category"].unique())[:5]
print(f"  {'Category':<25} {'Train%':>7} {'Val%':>7} {'Test%':>7}")
for cat in sample_cats:
    tr_pct = (train_df["Category"] == cat).sum() / len(train_df) * 100
    va_pct = (val_df["Category"] == cat).sum() / len(val_df) * 100
    te_pct = (test_df["Category"] == cat).sum() / len(test_df) * 100
    print(f"  {cat:<25} {tr_pct:>6.2f}% {va_pct:>6.2f}% {te_pct:>6.2f}%")

# Export to data/processed/
processed_dir = os.path.join(ROOT, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

# Save with cleaned_text column included
cols_to_save = ["ID", "Resume_str", "cleaned_text", "Category"]
train_df[cols_to_save].to_csv(os.path.join(processed_dir, "train.csv"), index=False)
val_df[cols_to_save].to_csv(os.path.join(processed_dir, "val.csv"), index=False)
test_df[cols_to_save].to_csv(os.path.join(processed_dir, "test.csv"), index=False)

print(f"\n✅ Splits exported to: {processed_dir}")
print(f"   train.csv  ({len(train_df)} rows)")
print(f"   val.csv    ({len(val_df)} rows)")
print(f"   test.csv   ({len(test_df)} rows)")

# ════════════════════════════════════════════════════════════════════════
# P1.12 — Handoff Documentation
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("P1.12: Generating Handoff Documentation")
print("─" * 70)

handoff_doc = f"""P1 → P2/P3 Preprocessing & Split Handoff Document
==================================================
Author: Poojan (Person 1)
Date: 2026-10-05
Project: NueralX Resume Classification (SAMATRIX RESUMEFORGE 2026)

1. Dataset Overview
-------------------
- Source: data/raw/csv/Resume.csv (2,484 records, 24 classes)
- After cleaning: {len(df_clean)} records
  - Dropped: {empty_mask.sum()} empty Resume_str
  - Dropped: {(short_mask & ~empty_mask).sum()} records with < 10 words
  - Dropped: {dup_mask.sum()} duplicate texts (kept first occurrence)

2. Preprocessing Function: clean_resume_text()
-----------------------------------------------
Location: src/preprocessing.py

Steps applied (in order):
  1. HTML entity unescaping (html.unescape)
  2. HTML tag stripping (regex)
  3. Case folding to lowercase
  4. Protected tech token substitution (C++, C#, .NET, CI/CD, Node.js, Vue.js, React.js)
  5. URL normalization → "url_token"
  6. Email normalization → "email_token"
  7. Phone number normalization → "phone_token"
  8. Punctuation / special character removal (keep alphanumeric + whitespace)
  9. Protected token restoration (with safe replacements: c++ → cplusplus, etc.)
  10. Whitespace normalization (collapse multiple spaces, strip)

IMPORTANT NOTES:
  - Stopword removal is NOT applied in clean_resume_text().
    → P2 (TF-IDF): Use stop_words='english' in TfidfVectorizer (already configured).
    → P3 (Word2Vec): Decide whether to remove stopwords before training embeddings.
  - Lemmatization is NOT applied in clean_resume_text().
    → A postprocess_text() helper is available in scripts/p1_preprocessing_split.py
      if either P2 or P3 wants to apply it.
  - All standard technical terms (Python, SQL, AWS, TensorFlow, Java, Docker, NLP, etc.)
    survive the pipeline as-is because they are simple alphanumeric tokens.
  - Terms with special characters (C++, C#, .NET, CI/CD) are handled via
    PROTECTED_TERMS substitution to prevent regex stripping.

3. Stratified Split
--------------------
Function: src/data_loader.stratified_split()
Ratios: 70% train / 15% val / 15% test
Random state: 42
Stratified on: Category column

Split sizes:
  Train: {len(train_df)} records ({len(train_df)/len(df_clean)*100:.1f}%)
  Val:   {len(val_df)} records ({len(val_df)/len(df_clean)*100:.1f}%)
  Test:  {len(test_df)} records ({len(test_df)/len(df_clean)*100:.1f}%)

4. Output Files
----------------
Location: data/processed/
Files:
  - train.csv (columns: ID, Resume_str, cleaned_text, Category)
  - val.csv   (columns: ID, Resume_str, cleaned_text, Category)
  - test.csv  (columns: ID, Resume_str, cleaned_text, Category)

5. Usage for P2 (Jeel)
-----------------------
  import pandas as pd
  train = pd.read_csv("data/processed/train.csv")
  # Use 'cleaned_text' column for TF-IDF vectorization
  # TfidfVectorizer already has stop_words='english' configured in src/features.py

6. Usage for P3 (Savan)
-----------------------
  import pandas as pd
  train = pd.read_csv("data/processed/train.csv")
  # Use 'cleaned_text' column for Word2Vec training
  # Tokenize with: tokens = text.split()  (whitespace tokenization)
  # Consider whether to remove stopwords before embedding training

7. Reproducibility
-------------------
  - All random states are set to 42
  - clean_resume_text() is deterministic (no random behavior)
  - Split is deterministic given the same input data and random_state
"""

handoff_path = os.path.join(ROOT, "docs", "p1_preprocessing_handoff.txt")
os.makedirs(os.path.dirname(handoff_path), exist_ok=True)
with open(handoff_path, "w", encoding="utf-8") as f:
    f.write(handoff_doc)

print(f"✅ Handoff document saved to: {handoff_path}")

# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("P1.10–P1.12 COMPLETE")
print("=" * 70)
