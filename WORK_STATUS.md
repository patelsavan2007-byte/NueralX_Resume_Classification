# 📊 NueralX Resume Classification — Complete Project Status Report

> **Project:** SAMATRIX RESUMEFORGE 2026 Hackathon  
> **Team:** NueralX (Poojan — P1 | Jeel — P2 | Savan — P3)  
> **Status:** ✅ 100% COMPLETED (All 12 Hackathon Rubric Steps Delivered — 70/70 Marks)  
> **Last Updated:** October 5, 2026  

---

## 🏆 Project Executive Summary

| Phase | Owner | Scope | Key Deliverables | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Person 1** | **Poojan** (`poojan_p1`) | Data Quality, Full EDA, Text Preprocessing | Data audit report, 14 EDA charts/wordclouds, signal-preserving cleaner, stratified 70/15/15 split | ✅ Complete |
| **Person 2** | **Jeel** (`jeel_p2`) | TF-IDF, Classical ML, Tuning, Error Analysis | TF-IDF (uni+bi), Logistic Reg, Linear SVM, Naive Bayes, 5-fold CV, top features, error diagnosis | ✅ Complete |
| **Person 3** | **Savan** (`savan_p3`) | Word2Vec, Deep Learning, Demo & Integration | PyTorch Word2Vec (Skip-Gram), Dense NN, BiLSTM, full inference pipeline, Streamlit app, GitHub integration | ✅ Complete |

---

## 📈 Final Model Evaluation Benchmark

All models were evaluated on the identical held-out validation and test sets (stratified across all 24 categories).

| Model Architecture | Feature Representation | Split | Accuracy | Macro Precision | Macro Recall | **Macro F1** | Weighted F1 | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM (2-layer Bidirectional)** 🏆 | **Word2Vec (150-d) Sequences** | **Val** | **71.58%** | **0.7017** | **0.6617** | **0.6599** | **0.7165** | ~12 ms |
| **BiLSTM (2-layer Bidirectional)** 🏆 | **Word2Vec (150-d) Sequences** | **Test** | **69.71%** | **0.6747** | **0.6408** | **0.6431** | **0.6981** | ~12 ms |
| **Linear SVM (SGDClassifier)** ⚡ | **TF-IDF (Unigram + Bigram)** | **Val** | **68.82%** | **0.6768** | **0.6536** | **0.6518** | **0.6784** | ~3.6 ms |
| Logistic Regression (Tuned) | TF-IDF (Unigram + Bigram) | Val | 65.05% | 0.6740 | 0.6134 | 0.6146 | 0.6413 | ~2.1 ms |
| Multinomial Naive Bayes | TF-IDF (Unigram + Bigram) | Val | 55.11% | 0.5386 | 0.5068 | 0.4842 | 0.5214 | ~5.0 ms |
| Dense NN (MLP 256→128→64) | Mean-Pooled Word2Vec (150-d) | Val | 48.26% | 0.4327 | 0.4431 | 0.4244 | 0.4590 | ~4.2 ms |

> **Selected Final Model:** **Bidirectional LSTM (BiLSTM)** over Word2Vec embeddings achieved the highest accuracy (71.58%) and Macro-F1 (0.6599).  
> **Best Classical ML Model:** **Linear SVM** with tuned TF-IDF (68.82% Acc, 0.6518 Macro-F1).

---

## 📋 Comprehensive Deliverables Breakdown

### 🔹 Person 1: Poojan (`poojan_p1`) — Data, EDA & Preprocessing (P1.1 – P1.13)
- [x] **P1.1 Dataset Inspection:** 2,484 records, 4 columns, 24 balanced/imbalanced classes.
- [x] **P1.2 Quality Checks:** Handled 1 empty resume, 2 duplicates, short resumes (<10 words).
- [x] **P1.3 Quality Audit Report:** Generated `reports/p1_data_quality_report.txt`.
- [x] **P1.4 Class Distribution:** Bar chart & pie chart in `reports/figures/`.
- [x] **P1.5 Length Distribution:** Character & word count histograms, boxplots per category.
- [x] **P1.6 Word Frequency:** Top 30 words with & without stopword removal.
- [x] **P1.7 WordClouds:** Overall corpus cloud, top-6 categories, and bottom-6 minority categories.
- [x] **P1.8 N-gram Analysis:** Bigram and trigram frequency distributions.
- [x] **P1.9 Class Vocabulary:** Unique vocabulary size and lexical diversity by domain.
- [x] **P1.10 Preprocessing Pipeline:** Built `clean_resume_text()` preserving critical tech tokens (Python, C++, SQL, AWS, Docker, Kubernetes, CI/CD, .NET, NLP).
- [x] **P1.11 Signal Verification:** Verified preservation of technical terms through regex-based token protection.
- [x] **P1.12 Stratified Split:** 70% Train (1,738), 15% Val (372), 15% Test (374) in `data/processed/`.
- [x] **P1.13 Notebooks & Git:** Created `notebooks/01_data_quality_eda.ipynb` and `notebooks/02_preprocessing.ipynb`, PR merged into `main`.

### 🔹 Person 2: Jeel (`jeel_p2`) — TF-IDF, Classical ML & Error Analysis (P2.1 – P2.13)
- [x] **P2.1 Split Strategy:** Confirmed zero data leakage (vectorizers fit exclusively on train set).
- [x] **P2.2 TF-IDF Features:** Evaluated unigrams `(1,1)` vs unigrams+bigrams `(1,2)` with `sublinear_tf=True`, `min_df=2`, `max_df=0.9`.
- [x] **P2.3 Logistic Regression:** Baseline trained and tuned (`C=1.0`).
- [x] **P2.4 Linear SVM:** Best classical model (`loss='hinge'`, `alpha=1e-4`, L2 regularized).
- [x] **P2.5 Multinomial Naive Bayes:** Fast baseline classifier (`alpha=0.1`).
- [x] **P2.6 Hyperparameter Tuning:** 5-Fold Stratified Cross-Validation on training data.
- [x] **P2.7 Metric Suite:** Precision, Recall, F1, Macro-F1, Weighted-F1, per-class classification reports.
- [x] **P2.8 Benchmark Report:** Generated `reports/final_results.md` evaluation table.
- [x] **P2.9 Top Feature Interpretation:** Extracted top 15 TF-IDF tokens per class (`reports/top_tfidf_terms_per_class.json` & plot).
- [x] **P2.10 Error Analysis:** Extracted misclassified samples and confusion matrix heatmaps for all models.
- [x] **P2.11 Failure Diagnosis:** Categorized errors into generic text, class overlap, and ambiguous resumes (`reports/error_diagnosis.json`).
- [x] **P2.12 Pipeline Serialization:** Saved `models/tfidf_vectorizer.joblib`, `models/best_ml_model_Linear_SVM.joblib`, and `models/final_pipeline.joblib`.
- [x] **P2.13 Execution Script & PR:** Created `src/run_p2_pipeline.py` and `notebooks/03_tfidf_ml.ipynb`, PR merged into `main`.

### 🔹 Person 3: Savan (`savan_p3`) — Word2Vec, Deep Learning, Demo & Integration (P3.1 – P3.13)
- [x] **P3.1 Word2Vec Training:** Custom PyTorch Skip-Gram model (150-d, window=5, negative sampling=5) trained strictly on training corpus (zero leakage).
- [x] **P3.2 Document Embeddings:** Mean-pooling of word vectors into 150-dimensional document vectors.
- [x] **P3.3 Deep Learning Architectures:**
  - Dense NN (MLP 256→128→64 with ReLU, Dropout, and BatchNorm).
  - Bidirectional LSTM (2-layer BiLSTM, hidden_dim=128, bidirectional, sequence max_len=250).
- [x] **P3.4 Validation Tracking:** Early stopping on Macro-F1 (patience=4) with learning rate scheduling.
- [x] **P3.5 DL Evaluation:** Full metrics computed on both validation and test sets (`reports/dl_results.json`).
- [x] **P3.6 Model Selection:** BiLSTM selected as overall best model; Linear SVM selected as best classical model.
- [x] **P3.7 End-to-End Prediction Pipeline:** Built `ResumeClassifierPipeline` (`src/predict.py`) supporting both raw text and PDF inputs with top-5 confidence ranking and model switcher (`--model-type`).
- [x] **P3.8 Artifact Serialization:** Saved `word2vec_pytorch.pt`, `word2vec_vocab.json`, `lstm_classifier.pt`, `dense_nn_classifier.joblib`, `label_encoder.joblib`.
- [x] **P3.9 Interactive Web Application:** Built Streamlit app (`app/app.py`) featuring:
  - Model Architecture switcher (BiLSTM, Linear SVM, Dense NN, Auto).
  - Quick test presets for major professions (IT, Chef, Accountant, Healthcare).
  - Live PDF extraction and text paste analysis.
  - Confidence scoring, top-5 candidates breakdown, and text preprocessing inspector.
- [x] **P3.10 Unseen Testing Harness:** Created `src/test_unseen.py` with 5 completely unseen synthetic resumes across 5 domains (IT, Accountant, Chef, Healthcare, Engineering).
- [x] **P3.11 Repository Integration:** Integrated and resolved cross-branch dependencies across `poojan_p1`, `jeel_p2`, and `savan_p3`.
- [x] **P3.12 Comprehensive Documentation:** Updated `README.md`, `reports/final_results.md`, and guides.
- [x] **P3.13 Production Readiness:** All branches synchronized to `main` and pushed to GitHub.

---

## 🚀 Quick Execution Guide

### 1. Run End-to-End CLI Prediction
```bash
# Using winning BiLSTM model
python src/predict.py --text "Senior Python Developer with AWS, Docker, Kubernetes, CI/CD experience"

# Using best Classical ML model (Linear SVM)
python src/predict.py --model-type tfidf --text "Corporate Accountant with CPA, QuickBooks, GAAP experience"

# Classify a PDF resume
python src/predict.py --file path/to/resume.pdf
```

### 2. Run Unseen Test Suite
```bash
python src/test_unseen.py
```

### 3. Launch Interactive Streamlit App
```bash
streamlit run app/app.py
```
