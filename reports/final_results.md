# Model Evaluation & Benchmark Report

This document records the comparative benchmark results, metrics, confusion matrix analysis,
and error analysis across Classical ML architectures for the Resume Classification Challenge.

---

## 1. Evaluation Summary Table (Validation Set)

| Model Architecture | Feature Representation | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Train Time | Pred Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM** **(Best Overall DL)** | Word2Vec + Bidirectional LSTM | **0.7158** | **0.7017** | **0.6617** | **0.6599** | **0.7165** | ~90s | ~12ms |
| **Linear_SVM** **(Best Classical)** | TF-IDF (Unigram+Bigram) | 0.6882 | 0.6768 | 0.6536 | 0.6518 | 0.6784 | 0.21s | 3.6ms |
| **Linear_SVM_Tuned** | TF-IDF (Unigram+Bigram) | 0.6882 | 0.6768 | 0.6536 | 0.6518 | 0.6784 | 0.24s | 3.1ms |
| **Logistic_Regression_Tuned** | TF-IDF (Unigram+Bigram) | 0.6505 | 0.6740 | 0.6134 | 0.6146 | 0.6413 | 2.24s | 2.1ms |
| **Logistic_Regression** | TF-IDF (Unigram+Bigram) | 0.6183 | 0.6460 | 0.5834 | 0.5795 | 0.6032 | 1.08s | 2.4ms |
| **Multinomial_NB** | TF-IDF (Unigram+Bigram) | 0.5511 | 0.5386 | 0.5068 | 0.4842 | 0.5214 | 0.02s | 5.0ms |
| **Dense NN (MLP)** | Mean-Pooled Word2Vec (150-d) | 0.4826 | 0.4327 | 0.4431 | 0.4244 | 0.4590 | ~15s | ~4.2ms |

> *(Best Overall DL) = Highest validation accuracy (71.58%) and Macro-F1 (0.6599).*
> *(Best Classical) = Linear SVM with tuned TF-IDF (Macro-F1: 0.6518).*

---

## 2. Per-Class Performance Breakdown (Best Model)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| ACCOUNTANT | 0.5556 | 0.8824 | 0.6818 | 17 |
| ADVOCATE | 0.5000 | 0.5882 | 0.5405 | 17 |
| AGRICULTURE | 0.5000 | 0.2000 | 0.2857 | 10 |
| APPAREL | 0.5714 | 0.5333 | 0.5517 | 15 |
| ARTS | 0.5714 | 0.2500 | 0.3478 | 16 |
| AUTOMOBILE | 0.5000 | 0.3333 | 0.4000 | 6 |
| AVIATION | 0.9286 | 0.7647 | 0.8387 | 17 |
| BANKING | 0.7500 | 0.7059 | 0.7273 | 17 |
| BPO | 0.5000 | 0.2500 | 0.3333 | 4 |
| BUSINESS-DEVELOPMENT | 0.6087 | 0.7778 | 0.6829 | 18 |
| CHEF | 0.9333 | 0.8235 | 0.8750 | 17 |
| CONSTRUCTION | 0.7895 | 0.8824 | 0.8333 | 17 |
| CONSULTANT | 0.5333 | 0.4706 | 0.5000 | 17 |
| DESIGNER | 0.9333 | 0.8750 | 0.9032 | 16 |
| DIGITAL-MEDIA | 0.6923 | 0.6000 | 0.6429 | 15 |
| ENGINEERING | 0.7500 | 0.7059 | 0.7273 | 17 |
| FINANCE | 0.7143 | 0.5882 | 0.6452 | 17 |
| FITNESS | 0.9286 | 0.7647 | 0.8387 | 17 |
| HEALTHCARE | 0.6842 | 0.7647 | 0.7222 | 17 |
| HR | 0.8000 | 0.9412 | 0.8649 | 17 |
| INFORMATION-TECHNOLOGY | 0.6800 | 0.9444 | 0.7907 | 18 |
| PUBLIC-RELATIONS | 0.6087 | 0.8235 | 0.7000 | 17 |
| SALES | 0.6000 | 0.5294 | 0.5625 | 17 |
| TEACHER | 0.6111 | 0.6875 | 0.6471 | 16 |

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
- **Macro-F1:** 0.6518
- **Justification:** Selected based on highest Macro-F1 score ensuring fair evaluation across all 24 categories,
  combined with reasonable training time and prediction latency.
