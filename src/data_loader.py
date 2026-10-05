"""
Data Loading & Splitting Module
Author: Poojan (Person 1) & Team NueralX
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)
"""

import os
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "csv", "Resume.csv")


def load_resume_csv(filepath: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Loads resume dataset from CSV file.
    Expected columns: ID, Resume_str, Resume_html, Category.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Resume dataset not found at: {filepath}.\n"
            f"Please extract Resume.csv into data/raw/csv/ as instructed in data/README.md"
        )
    df = pd.read_csv(filepath)
    required_cols = {"ID", "Resume_str", "Category"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset missing required columns. Expected at least {required_cols}, got {df.columns}")
    return df


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a given PDF resume file using pypdf.
    Attempts layout-aware extraction first to preserve columnar spacing,
    table boundaries, and prevent glued tokens.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        try:
            # Layout mode preserves whitespace between separate text blocks and prevents glued tokens
            text = "\n".join(page.extract_text(extraction_mode="layout") or "" for page in reader.pages)
        except Exception:
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except ImportError:
        raise ImportError("pypdf is required to extract PDF text. Install it with: pip install pypdf")


def stratified_split(
    df: pd.DataFrame,
    text_col: str = "Resume_str",
    label_col: str = "Category",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a 3-way stratified train/val/test split (default 70% / 15% / 15%).
    Ensures identical class balance across all splits and avoids data leakage.
    """
    total = train_size + val_size + test_size
    if not (0.999 <= total <= 1.001):
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")

    # Step 1: Split into train and temp (val + test)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Step 2: Split temp into val and test proportionally
    relative_test_size = test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[label_col],
        random_state=random_state,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    try:
        data = load_resume_csv()
        print(f"Loaded {len(data)} records across {data['Category'].nunique()} categories.")
        train, val, test = stratified_split(data)
        print(f"Splits created: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    except Exception as e:
        print(f"Status: {e}")
