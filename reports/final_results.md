# Model Evaluation & Benchmark Report

This document records the comparative benchmark results, metrics, confusion matrix analysis, and error analysis across Classical ML and Deep Learning architectures for the Resume Classification Challenge.

---

## 1. Evaluation Summary Table

| Model Architecture | Feature Representation | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | TF-IDF (Unigram) | - | - | - | - | - | - |
| **Logistic Regression** | TF-IDF (Unigram + Bigram) | - | - | - | - | - | - |
| **Linear SVM (SGD / SVC)** | TF-IDF (Unigram + Bigram) | - | - | - | - | - | - |
| **Dense Neural Net** | Word2Vec Pooled Vectors | - | - | - | - | - | - |
| **LSTM / GRU** | Word2Vec Sequence Embeddings | - | - | - | - | - | - |

> *Note: Macro-F1 is our primary evaluation metric to account for class balance across all 24 categories.*

---

## 2. Per-Class Performance Breakdown

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| ACCOUNTANT | - | - | - | - |
| ADVOCATE | - | - | - | - |
| AGRICULTURE | - | - | - | - |
| APPAREL | - | - | - | - |
| ARTS | - | - | - | - |
| AUTOMOBILE | - | - | - | - |
| AVIATION | - | - | - | - |
| BANKING | - | - | - | - |
| BPO | - | - | - | - |
| BUSINESS-DEVELOPMENT | - | - | - | - |
| CHEF | - | - | - | - |
| CONSTRUCTION | - | - | - | - |
| CONSULTANT | - | - | - | - |
| DESIGNER | - | - | - | - |
| DIGITAL-MEDIA | - | - | - | - |
| ENGINEERING | - | - | - | - |
| FINANCE | - | - | - | - |
| FITNESS | - | - | - | - |
| HEALTHCARE | - | - | - | - |
| HR | - | - | - | - |
| INFORMATION-TECHNOLOGY | - | - | - | - |
| PUBLIC-RELATIONS | - | - | - | - |
| SALES | - | - | - | - |
| TEACHER | - | - | - | - |

---

## 3. Confusion Matrix Analysis
- Figures saved in `reports/figures/confusion_matrix.png`.
- High confusion pairs identified (e.g. `FINANCE` vs `ACCOUNTANT` or `SALES` vs `BUSINESS-DEVELOPMENT`).

---

## 4. Error Analysis & Failure Diagnosis
Inspection of false positives and false negatives on the test set:
1. **Ambiguous / Generic Resumes:** Resumes containing generic business administration or communication skills without deep technical keywords.
2. **Overlapping Categories:** Multi-domain candidates (e.g., IT professionals working in Banking or Sales engineers).
3. **Short / Noisy Text:** Truncated resumes or OCR extraction artifacts.

---

## 5. Final Model Selection Justification
- Chosen Model:
- Justification based on trade-off between Macro-F1 score, inference latency, memory footprint, and robustness to out-of-vocabulary terms.
