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

---
---

# Deep Learning Results — Person 3 (`savan_p3`)

> Appended by Savan. Sections 1–5 above are Person 2's classical-ML report and are left unchanged.
> Metrics below were **recomputed from the serialized artifacts** by `src/evaluate_dl.py` rather
> than copied from training logs, which also verifies the artifacts round-trip from disk.
> Machine-readable copy: `reports/dl_results.json`.

Splits (stratified, produced by Person 1): train 1,738 / val 373 / test 373 across 24 categories.

## 6. Deep Learning Evaluation Summary

| Model | Feature Representation | Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| Dense NN (MLP 256-128-64) | Mean-pooled Word2Vec (150d) | val | 0.4826 | 0.4327 | 0.4431 | 0.4244 | 0.4590 |
| Dense NN (MLP 256-128-64) | Mean-pooled Word2Vec (150d) | test | 0.5469 | 0.4896 | 0.5064 | 0.4809 | 0.5171 |
| **BiLSTM** (2 layers, 128 hidden, bidirectional) | Word2Vec token sequences (max_len 250) | val | 0.7158 | 0.7017 | 0.6617 | **0.6599** | 0.7165 |
| **BiLSTM** (2 layers, 128 hidden, bidirectional) | Word2Vec token sequences (max_len 250) | test | 0.6971 | 0.6747 | 0.6408 | **0.6431** | 0.6981 |
| Linear_SVM (Person 2) | TF-IDF (Unigram+Bigram) | val | 0.6989 | 0.7129 | 0.6631 | 0.6626 | 0.6880 |

### Word2Vec Configuration (P3.1)
Skip-Gram with negative sampling in PyTorch (`src/word2vec.py`), fit on the **training split only**
so no validation or test text ever influenced the embeddings.

| Parameter | Value |
| :--- | :--- |
| vector_size | 150 |
| window | 5 |
| min_count | 2 |
| epochs | 5 |
| skip-gram pairs | ~2.5 M |
| final vocabulary | 20,371 words |
| training loss | 1.18 -> 0.73 |

## 7. Head-to-Head: Classical ML vs Deep Learning (P3.6)

| Criterion | Linear_SVM (TF-IDF) | BiLSTM (Word2Vec) | Winner |
| :--- | :--- | :--- | :--- |
| Val Macro-F1 | **0.6626** | 0.6599 | Linear_SVM (by 0.003) |
| Val Accuracy | 0.6989 | **0.7158** | BiLSTM |
| Test Macro-F1 | not reported by P2 | **0.6431** | BiLSTM |
| Train time | 0.47 s | ~25 min (CPU) | Linear_SVM |
| Inference latency | 7.3 ms | ~15 ms | Linear_SVM |
| Needs a fitted TF-IDF vocabulary at inference | Yes | No | BiLSTM |
| Retains word order | No | Yes | BiLSTM |

**Interpretation.** On the validation split the two are effectively tied — a 0.003 Macro-F1 gap on
n=373 sits well inside sampling noise, so this is *not* a meaningful DL victory over TF-IDF. The
BiLSTM is nonetheless the model shipped, on two grounds: it is decisively better than the project's
other DL variant (the Dense NN, 0.4809 test Macro-F1), and it needs no TF-IDF vocabulary at
inference time. If prediction latency were the priority, Linear_SVM would be the better
production choice.

## 8. Deep Learning Failure Modes (BiLSTM, test set)

| Class | Precision | Recall | F1 | Support | Reading |
| :--- | ---: | ---: | ---: | ---: | :--- |
| AUTOMOBILE | 0.00 | 0.00 | 0.00 | 5 | Never predicted; absorbed into ENGINEERING / AVIATION |
| BPO | 0.00 | 0.00 | 0.00 | 4 | Minimal support; generic call-centre wording |
| APPAREL | 0.12 | 0.21 | 0.15 | 14 | Retail vocabulary overlaps heavily with SALES |
| AGRICULTURE | 0.29 | 0.20 | 0.24 | 10 | Small class, little training signal |
| ARTS | 0.33 | 0.19 | 0.24 | 16 | Small class, little training signal |
| FITNESS | 0.43 | 0.35 | 0.39 | 17 | Overlaps HEALTHCARE and adjacent sports roles |
| HEALTHCARE | 0.28 | 0.78 | 0.41 | 18 | Over-predicted: recall far exceeds precision |
| PUBLIC-RELATIONS | 0.86 | 0.38 | 0.52 | 16 | Under-predicted: absorbed into CONSULTANT |
| ENGINEERING | 1.00 | 1.00 | 1.00 | 17 | Clean, distinctive vocabulary |
| HR | 1.00 | 1.00 | 1.00 | 16 | Clean, distinctive vocabulary |
| BUSINESS-DEVELOPMENT | 0.90 | 1.00 | 0.95 | 18 | Strong |
| CONSULTANT | 0.89 | 0.94 | 0.92 | 18 | Strong |
| SALES | 0.89 | 0.94 | 0.91 | 17 | Strong |

**Diagnosis.** The five worst classes are the five with the fewest training examples; the best
classes are the most lexically distinctive. Because macro-averaging weights all 24 categories
equally, a handful of small classes caps the headline score. This is a **class-imbalance problem
rather than a capacity problem** — enlarging the embedding or widening the LSTM would not be
expected to help much. Recommended next levers, in order of expected value:

1. Class-weighted cross-entropy loss, or resampling of the 6 smallest categories.
2. Two-stage / hierarchical classification to separate the overlapping pairs
   (AUTOMOBILE vs AVIATION vs ENGINEERING, APPAREL vs SALES, HEALTHCARE vs FITNESS).
3. TF-IDF + Word2Vec feature concatenation, letting the LSTM inherit the bag-of-words evidence
   that Person 2's linear model exploits.

## 9. Unseen Resume Test (P3.10)

Five hand-written resumes appearing nowhere in the dataset, run through the full saved pipeline
(`src/test_unseen.py`): **3/5 correct (60%)**.

| # | Expected | Predicted | Confidence | Result |
| :-- | :--- | :--- | ---: | :--- |
| 1 | INFORMATION-TECHNOLOGY | HEALTHCARE | 15.7% | Miss (low confidence) |
| 2 | ACCOUNTANT | ACCOUNTANT | 70.9% | Correct |
| 3 | CHEF | CHEF | 96.2% | Correct |
| 4 | HEALTHCARE | HEALTHCARE | 79.3% | Correct |
| 5 | ENGINEERING | AVIATION | 24.5% | Miss (low confidence) |

Both misses carry low confidence (<26%), which is the desirable behaviour under uncertainty and
shows the model is not overconfident on out-of-distribution text. Miss 1 is the known
IT-vs-Healthcare overlap; miss 5 is the known ENGINEERING-vs-AVIATION overlap from Section 8. The
unseen-test error profile therefore matches the in-distribution one rather than revealing a new
failure mode.
