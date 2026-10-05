# Model Evaluation & Benchmark Report

This document records the comparative benchmark results, metrics, confusion matrix analysis,
and error analysis across Classical ML architectures for the Resume Classification Challenge.

---

## 1. Evaluation Summary Table (Validation Set)

| Model Architecture | Feature Representation | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Train Time | Pred Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear_SVM** **(Best Classical)** | TF-IDF (Unigram+Bigram) | 0.6989 | 0.7129 | 0.6631 | 0.6626 | 0.6880 | 0.47s | 7.3ms |
| **Linear_SVM_Tuned** | TF-IDF (Unigram+Bigram) | 0.6989 | 0.7129 | 0.6631 | 0.6626 | 0.6880 | 0.44s | 6.4ms |
| **Logistic_Regression_Tuned** | TF-IDF (Unigram+Bigram) | 0.6559 | 0.6763 | 0.6183 | 0.6189 | 0.6460 | 5.04s | 4.1ms |
| **Logistic_Regression** | TF-IDF (Unigram+Bigram) | 0.6210 | 0.6571 | 0.5903 | 0.5884 | 0.6069 | 2.27s | 4.1ms |
| **Multinomial_NB** | TF-IDF (Unigram+Bigram) | 0.5511 | 0.5361 | 0.5068 | 0.4844 | 0.5216 | 0.05s | 7.6ms |
| **Dense NN (MLP)** | Word2Vec Mean Pooling (150-dim) | 0.4826 | 0.4327 | 0.4431 | 0.4244 | 0.4590 | ~2min | ~6ms |
| **BiLSTM** ⭐ **(Best Overall)** | Word2Vec Sequence (150-dim) | **0.7158** | **0.7017** | **0.6617** | **0.6599** | **0.7165** | ~8min | ~28ms |

> *(Best Classical) = Best TF-IDF+ML model by Macro-F1. ⭐ = Overall best model selected for deployment.*
> *Word2Vec: PyTorch Skip-Gram, 150-dim, 5 epochs, vocab=20,371, trained on train split only (zero leakage).*

---

## 2. Per-Class Performance Breakdown (Best Model)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| ACCOUNTANT | 0.5769 | 0.8824 | 0.6977 | 17 |
| ADVOCATE | 0.4444 | 0.4706 | 0.4571 | 17 |
| AGRICULTURE | 0.6667 | 0.2000 | 0.3077 | 10 |
| APPAREL | 0.6364 | 0.4667 | 0.5385 | 15 |
| ARTS | 0.6667 | 0.2500 | 0.3636 | 16 |
| AUTOMOBILE | 0.5000 | 0.3333 | 0.4000 | 6 |
| AVIATION | 0.9286 | 0.7647 | 0.8387 | 17 |
| BANKING | 0.7500 | 0.7059 | 0.7273 | 17 |
| BPO | 1.0000 | 0.2500 | 0.4000 | 4 |
| BUSINESS-DEVELOPMENT | 0.7778 | 0.7778 | 0.7778 | 18 |
| CHEF | 0.8667 | 0.7647 | 0.8125 | 17 |
| CONSTRUCTION | 0.8333 | 0.8824 | 0.8571 | 17 |
| CONSULTANT | 0.5000 | 0.4706 | 0.4848 | 17 |
| DESIGNER | 0.9375 | 0.9375 | 0.9375 | 16 |
| DIGITAL-MEDIA | 0.6667 | 0.5333 | 0.5926 | 15 |
| ENGINEERING | 0.7647 | 0.7647 | 0.7647 | 17 |
| FINANCE | 0.7143 | 0.5882 | 0.6452 | 17 |
| FITNESS | 0.9375 | 0.8824 | 0.9091 | 17 |
| HEALTHCARE | 0.7368 | 0.8235 | 0.7778 | 17 |
| HR | 0.8421 | 0.9412 | 0.8889 | 17 |
| INFORMATION-TECHNOLOGY | 0.6800 | 0.9444 | 0.7907 | 18 |
| PUBLIC-RELATIONS | 0.5600 | 0.8235 | 0.6667 | 17 |
| SALES | 0.5217 | 0.7059 | 0.6000 | 17 |
| TEACHER | 0.6000 | 0.7500 | 0.6667 | 16 |

---

## 3. Confusion Matrix Analysis

- Confusion matrix heatmaps for all models saved in `reports/figures/`.
- High confusion pairs identified between related categories.

---

## 4. Error Analysis & Failure Diagnosis

Inspection of misclassifications on the validation set:

1. **Ambiguous / Generic Resumes:** Resumes with generic business skills lacking domain-specific keywords.
2. **Overlapping Categories:** Multi-domain candidates (e.g., IT professionals in Banking, Sales engineers).
3. **Short / Noisy Text:** Truncated resumes or OCR artifacts reducing discriminative signal.

---

## 5. Final Model Selection Justification

- **Chosen Model:** Linear_SVM
- **Macro-F1:** 0.6626
- **Justification:** Selected based on highest Macro-F1 score ensuring fair evaluation across all 24 categories,
  combined with reasonable training time and prediction latency.
