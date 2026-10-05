"""
Text Preprocessing Pipeline Module
Author: Poojan (Person 1) & Team NueralX
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Rule: Preprocessing must be reproducible and identical during training and inference.
Crucially: Preserve meaningful technical tokens (e.g., C++, .NET, CI/CD, SQL, Python, AWS).
"""

import re
import html
from typing import List

# Key technical patterns to protect before aggressive regex cleaning
PROTECTED_TERMS = {
    r"\bc\+\+\b": "cpp_token",
    r"\bc#\b": "csharp_token",
    r"\b\.net\b": "dotnet_token",
    r"\bci/cd\b": "cicd_token",
    r"\bnode\.js\b": "nodejs_token",
    r"\bvue\.js\b": "vuejs_token",
    r"\breact\.js\b": "reactjs_token",
}

REVERSE_PROTECTED = {v: k.replace(r"\b", "").replace("\\", "") for k, v in PROTECTED_TERMS.items()}


def clean_resume_text(text: str, preserve_tech_terms: bool = True) -> str:
    """
    Standardizes and normalizes resume text for feature extraction and modeling.
    
    Steps:
    1. Unescape HTML entities and strip HTML tags.
    2. Protect critical programming languages / terms (e.g., C++, .NET).
    3. Normalize URLs, emails, phone numbers into generic tokens.
    4. Remove non-informative special characters and punctuation.
    5. Restore protected technical tokens.
    6. Normalize whitespace and case fold.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities and strip HTML markup
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Lowercase text
    text = text.lower()

    # 3. Protect specific symbols in tech words
    if preserve_tech_terms:
        for pattern, placeholder in PROTECTED_TERMS.items():
            text = re.sub(pattern, placeholder, text)

    # 4. Normalize URLs, emails, phone numbers
    text = re.sub(r"https?://\S+|www\.\S+", " url_token ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " email_token ", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", " phone_token ", text)

    # 5. Remove unwanted punctuation/symbols while preserving word boundaries
    text = re.sub(r"[^\w\s]", " ", text)

    # 6. Restore protected technical tokens
    if preserve_tech_terms:
        for placeholder, original in REVERSE_PROTECTED.items():
            text = text.replace(placeholder, original.replace("+", "plus").replace("#", "sharp").replace(".", ""))

    # 7. Normalize multiple whitespaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_resume(text: str) -> List[str]:
    """
    Simple whitespace and word boundary tokenizer.
    """
    cleaned = clean_resume_text(text)
    return cleaned.split()


if __name__ == "__main__":
    sample = "Experienced Senior Developer in C++, .NET & Python. Contact: dev@example.com, Phone: +1-555-123-4567. Website: https://github.com"
    print("Original:", sample)
    print("Cleaned :", clean_resume_text(sample))
