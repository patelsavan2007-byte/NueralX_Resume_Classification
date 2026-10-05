# 🚀 NueralX - Resume Classification System
> **SAMATRIX RESUMEFORGE 2026 Hackathon**  
> End-to-End NLP & Machine Learning Pipeline for Automated Resume Categorization

---

## 📌 Problem Overview
Modern recruitment pipelines receive thousands of resumes across varied domains. The objective of this project is to build an end-to-end, reproducible Resume Classification system that takes raw/extracted resume text and accurately predicts its professional category across **24 distinct job categories**.

### End-to-End Workflow
```
Problem Definition ➔ Data Gathering & Quality Checks ➔ Comprehensive EDA ➔
Text Preprocessing (Preserving Tech Signal) ➔ Feature Engineering (TF-IDF & Word2Vec) ➔
Model Training (Classical ML & Deep Learning) ➔ Rigorous Evaluation (Macro-F1) ➔
Error Analysis & Diagnosis ➔ Interactive Demo (Streamlit)
```

---

## 👥 Team & Task Division

| Team Member | Branch | Primary Ownership | Core Deliverables |
| :--- | :--- | :--- | :--- |
| **Poojan** (Person 1) | `poojan_p1` | Data, Data Quality, EDA & Preprocessing | Data quality report, EDA notebook (`01`), reusable `clean_resume_text()` pipeline, stratified split logic |
| **Jeel** (Person 2) | `jeel_p2` | TF-IDF, Classical ML, Evaluation & Error Analysis | TF-IDF feature pipelines (`(1,1)` vs `(1,2)`), Logistic Regression, Linear SVM, Naive Bayes, comparison table, error diagnosis |
| **Savan** (Person 3) | `savan_p3` | Word2Vec, Deep Learning, Demo & Integration | Corpus-trained Word2Vec, Dense/LSTM neural classifier, Streamlit web demo (`app/app.py`), repository integration |

---

## 🌿 Git Branching Strategy & Workflow
To maintain clean version control and avoid merge conflicts, team members work in isolated feature branches:

1. **`main`**: Production-ready, finalized code and documentation.
2. **`develop`**: Integration branch where tested PRs from team branches are combined.
3. **`poojan_p1`**: Person 1 working branch (Data Quality, EDA, Preprocessing).
4. **`jeel_p2`**: Person 2 working branch (TF-IDF & Classical ML models).
5. **`savan_p3`**: Person 3 working branch (Word2Vec, Deep Learning, Web Demo & Integration).

```
main (stable release)
 └── develop (integration)
      ├── poojan_p1 (Data & EDA)
      ├── jeel_p2 (Classical ML)
      └── savan_p3 (DL & Web Demo)
```

---

## 📁 Repository Structure

```text
NueralX_Resume_Classification/
│
├── .gitignore                      <-- Excludes all raw/processed data, model weights, venv
├── README.md                       <-- Main project overview and guide
├── requirements.txt                <-- Pinned dependencies
│
├── docs/                           <-- Competition problem statement & guidelines
│   ├── Problem statement and Guidelines for Hackthon.docx
│   └── Resume_Classification_Hackathon_Task_Division.docx
│
├── data/
│   ├── README.md                   <-- Dataset schema, categories, and placement guide
│   ├── raw/                        <-- Local storage for raw files (Git-ignored)
│   │   ├── csv/                    <-- Resume.csv & Resume.xlsx
│   │   └── pdf_resumes/            <-- 2,500 categorized PDF documents
│   └── processed/                  <-- Train, Val, Test split CSVs (Git-ignored)
│
├── notebooks/                      <-- Step-by-step hackathon notebooks
│   ├── 01_data_quality_eda.ipynb   <-- P1: Data checks, class distribution, word frequency, n-grams
│   ├── 02_preprocessing.ipynb      <-- P1: Cleaning, token protection, stratified split
│   ├── 03_tfidf_ml.ipynb           <-- P2: TF-IDF baselines (Logistic Regression, Linear SVM, MNB)
│   ├── 04_word2vec_dl.ipynb        <-- P3: Word2Vec embeddings + Neural Classifier
│   └── 05_final_evaluation.ipynb   <-- All: Benchmark comparison, confusion matrix & unseen testing
│
├── src/                            <-- Reusable, production-ready modules
│   ├── __init__.py
│   ├── data_loader.py              <-- CSV/PDF loading & stratified splitting
│   ├── preprocessing.py            <-- Text normalization preserving tech tokens (C++, SQL, AWS)
│   ├── features.py                 <-- TF-IDF vectorizer & Word2Vec pooling
│   ├── train_ml.py                 <-- Classical ML training & hyperparameter routines
│   ├── train_dl.py                 <-- Deep Learning classifier training routines
│   ├── evaluate.py                 <-- Macro-F1, confusion matrix & error analysis extraction
│   └── predict.py                  <-- End-to-end inference pipeline
│
├── models/                         <-- Serialized pipelines & weights (Git-ignored)
│   └── README.md
│
├── app/                            <-- Interactive application
│   └── app.py                      <-- Streamlit web demo for resume classification
│
└── reports/                        <-- Evaluation outputs & figures
    ├── figures/                    <-- Confusion matrix plots & EDA figures
    └── final_results.md            <-- Benchmark comparison tables and analysis
```

---

## 🛠️ Setup & Installation

### 1. Clone & Navigate
```bash
git clone https://github.com/patelsavan2007-byte/NueralX_Resume_Classification.git
cd NueralX_Resume_Classification
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup Dataset Locally
Download the dataset from the official Google Drive link (specified in [`data/README.md`](data/README.md)):
- Place `Resume.csv` inside `data/raw/csv/`.
- Extract PDF folders inside `data/raw/pdf_resumes/`.

---

## 🚀 Running the Streamlit Web Demo
Launch the interactive web application to classify pasted resume text or uploaded PDF resumes:
```bash
streamlit run app/app.py
```

---

## 🎯 Evaluation Strategy & Key Metrics
- **Primary Metric:** **Macro-F1 score** (ensures all 24 categories are evaluated fairly regardless of class imbalance).
- **Secondary Metrics:** Weighted-F1, Accuracy, Precision, Recall, and Confusion Matrix Heatmaps.
- **Error Analysis:** Thorough inspection of misclassified resumes to distinguish between label ambiguity, cross-domain overlap, and vocabulary sparsity.

---

## ✅ What Has Been Completed (Phase 1 — Project Setup)

### GitHub & Repository Setup
- [x] Created GitHub repository: [`NueralX_Resume_Classification`](https://github.com/patelsavan2007-byte/NueralX_Resume_Classification)
- [x] Created all required branches: `main`, `develop`, `poojan_p1`, `jeel_p2`, `savan_p3`
- [x] All branches pushed to remote with upstream tracking configured
- [x] Comprehensive `.gitignore` configured — blocks all `*.csv`, `*.xlsx`, `*.pdf`, `*.zip`, model weights (`*.pkl`, `*.joblib`, `*.h5`, `*.pt`) from being committed
- [x] `requirements.txt` created with pinned minimum versions for all dependencies

### Dataset & Folder Structure
- [x] Dataset downloaded from official [Google Drive link](https://drive.google.com/drive/folders/1I3sQZnnjeutdWXM9fNloVTk26zIxXYky)
- [x] `Resume.csv` (2,484 records) and `Resume.xlsx` extracted into `data/raw/csv/`
- [x] 2,500 PDF resumes extracted into `data/raw/pdf_resumes/` organized across 24 category folders
- [x] **Dataset is NOT pushed to GitHub** — kept local only, protected by `.gitignore`
- [x] `data/README.md` documents dataset schema (columns: `ID`, `Resume_str`, `Resume_html`, `Category`), all 24 categories, and local setup instructions
- [x] `models/README.md` documents expected model artifacts for final inference
- [x] `reports/final_results.md` — evaluation report template with comparison table and per-class metrics ready to fill

### Source Code Modules (Starter Implementations)
- [x] `src/data_loader.py` — CSV/PDF loading utility + stratified 3-way split function (70%/15%/15%)
- [x] `src/preprocessing.py` — Reproducible text cleaning pipeline (HTML stripping, URL/email/phone normalization, technical token protection for C++, .NET, SQL, AWS, etc.)
- [x] `src/features.py` — TF-IDF vectorizer builder + Word2Vec training & document pooling (gensim lazy-loaded for compatibility)
- [x] `src/train_ml.py` — Classical ML training harness (Logistic Regression, Linear SVM, Multinomial NB) with automatic Macro-F1 comparison
- [x] `src/train_dl.py` — Dense Neural Network classifier over pooled Word2Vec embeddings
- [x] `src/evaluate.py` — Metrics computation, confusion matrix heatmap generation, error analysis extraction
- [x] `src/predict.py` — End-to-end inference pipeline class (text or PDF input → preprocessing → model → prediction)

### Notebooks (Skeleton Templates)
- [x] `notebooks/01_data_quality_eda.ipynb` — Starter cells for data loading, missing values, duplicates, class distribution, word count histogram
- [x] `notebooks/02_preprocessing.ipynb` — Starter cells for text cleaning and stratified split export
- [x] `notebooks/03_tfidf_ml.ipynb` — Starter cells for TF-IDF feature extraction and ML baseline training
- [x] `notebooks/04_word2vec_dl.ipynb` — Starter cells for Word2Vec training and neural classifier
- [x] `notebooks/05_final_evaluation.ipynb` — Starter cells for final model comparison and unseen resume testing

### Streamlit Web Demo
- [x] `app/app.py` — Streamlit application skeleton with text paste / PDF upload input modes

---

## 📋 Remaining Tasks (Phase 2–9)

### 🔹 Poojan (Person 1 — `poojan_p1`) — Data, EDA & Preprocessing
| Task ID | Description | Status |
| :--- | :--- | :---: |
| P1.1 | Inspect dataset: identify file types, columns, label column, resume-text column, number of samples and classes | ☐ |
| P1.2 | Data quality: check missing/empty resumes, duplicates, corrupted records, label inconsistencies, very short resumes, noisy text, leakage | ☐ |
| P1.3 | Dataset summary: record total samples, number of classes, samples per class, missing values, duplicate count, basic text stats | ☐ |
| P1.4 | EDA — class distribution: bar chart of resume categories | ☐ |
| P1.5 | EDA — resume length: character count, word count, token count distributions | ☐ |
| P1.6 | EDA — word frequency: top 20–30 frequent words after cleaning | ☐ |
| P1.7 | EDA — WordCloud: overall and class-wise WordClouds | ☐ |
| P1.8 | EDA — n-grams: top unigrams, bigrams, trigrams | ☐ |
| P1.9 | Class-wise vocabulary: top TF-IDF terms or frequency per class | ☐ |
| P1.10 | Preprocessing: finalize `clean_resume_text()` — test stopword removal impact, lemmatization vs stemming | ☐ |
| P1.11 | Preserve signal: verify technical tokens (Python, C++, SQL, AWS, TensorFlow, .NET, NLP) are preserved | ☐ |
| P1.12 | Pipeline handoff: provide Person 2 and Person 3 with exact preprocessing function and documented split logic | ☐ |
| P1.13 | Commit + PR: push notebooks/scripts/figures to `poojan_p1` and create Pull Request into `develop` | ☐ |

### 🔹 Jeel (Person 2 — `jeel_p2`) — TF-IDF, Classical ML & Error Analysis
| Task ID | Description | Status |
| :--- | :--- | :---: |
| P2.1 | Split strategy: stratified train/val/test split (70/15/15 or 80/20), keep test untouched | ☐ |
| P2.2 | TF-IDF baseline: build `TfidfVectorizer`, compare `(1,1)` vs `(1,2)`, tune `min_df`/`max_df`/`max_features` | ☐ |
| P2.3 | Model A: train Logistic Regression using TF-IDF | ☐ |
| P2.4 | Model B: train Linear SVM using TF-IDF | ☐ |
| P2.5 | Model C: optionally train Multinomial Naive Bayes | ☐ |
| P2.6 | Model tuning: use validation data or cross-validation (do NOT tune against test set) | ☐ |
| P2.7 | Metrics: report accuracy, precision, recall, F1, macro-F1, weighted-F1, confusion matrix, per-class metrics | ☐ |
| P2.8 | Model comparison: create clear comparison table, select best classical model by macro-F1 | ☐ |
| P2.9 | Feature interpretation: inspect top weighted TF-IDF terms per class | ☐ |
| P2.10 | Error analysis: collect incorrect predictions with actual/predicted class, confidence, text preview | ☐ |
| P2.11 | Diagnose failures: categorize errors (class overlap, generic resumes, noisy text, short resumes, mislabeled data) | ☐ |
| P2.12 | Save model: serialize fitted preprocessing + vectorizer + model pipeline | ☐ |
| P2.13 | Commit + PR: push ML notebooks/scripts/results to `jeel_p2`, create Pull Request into `develop` | ☐ |

### 🔹 Savan (Person 3 — `savan_p3`) — Word2Vec, Deep Learning, Demo & Integration
| Task ID | Description | Status |
| :--- | :--- | :---: |
| P3.1 | Word2Vec: train on training corpus only (or use pretrained), do NOT train on test set | ☐ |
| P3.2 | Document representation: convert word vectors to document-level via sequence embeddings or mean/max pooling | ☐ |
| P3.3 | DL model: build neural classifier (Word2Vec + LSTM/GRU, or Dense model with pooled vectors) | ☐ |
| P3.4 | Validation: track validation performance, use regularization/early stopping to avoid overfitting | ☐ |
| P3.5 | DL evaluation: report same core metrics as ML models for fair comparison | ☐ |
| P3.6 | Final model selection: coordinate with Person 2, compare classical ML vs DL by metrics, robustness, latency | ☐ |
| P3.7 | Prediction pipeline: raw resume → preprocessing → feature extraction → model → predicted category | ☐ |
| P3.8 | Save artifacts: serialize tokenizer/vectorizer/embedding/model/preprocessing for reproducible inference | ☐ |
| P3.9 | Streamlit demo: finalize interactive UI for resume paste/upload and category prediction | ☐ |
| P3.10 | Unseen testing: test final system on at least 3–5 unseen resume examples | ☐ |
| P3.11 | GitHub integration: resolve merge conflicts, integrate P1/P2/P3, verify paths & dependencies | ☐ |
| P3.12 | README/demo: update README with final results, usage, and demo instructions | ☐ |
| P3.13 | Final PR: merge stable integration into `develop`, run complete project, prepare final merge to `main` | ☐ |

### 🔹 Shared Tasks — All 3 Members
- [ ] All members understand the problem statement, dataset classes, preprocessing decisions, and final model
- [ ] Pull latest `develop` branch before starting major integration work
- [ ] Use clear commit messages: `feat: add tfidf baseline`, `fix: clean empty resumes`, `docs: update setup`
- [ ] Review each other's Pull Requests (do not merge blindly)
- [ ] Verify notebooks can run from a clean environment or steps are documented
- [ ] Use the same test set for final model comparison
- [ ] Report full metrics (not just accuracy): precision, recall, F1, macro-F1, weighted-F1, confusion matrix, per-class results
- [ ] Before final submission: run entire pipeline from raw resume input to prediction
- [ ] Prepare short explanation of why the final model was selected and what its main failure cases are

### 🔹 Final Integration Checklist
- [ ] GitHub repository with clean structure and commit history
- [ ] `README.md` completed with results, limitations, and demo instructions
- [ ] `.gitignore` completed ✅
- [ ] `requirements.txt` / environment file completed ✅
- [ ] Dataset source documented (Google Drive) ✅
- [ ] Data quality checks completed
- [ ] EDA figures completed: class distribution, text length, frequent words, WordCloud, n-grams, class-wise vocabulary
- [ ] Reproducible preprocessing function completed
- [ ] Stratified train/val/test split completed before fitting vectorizers/embeddings
- [ ] TF-IDF baseline completed
- [ ] At least one classical ML classifier completed
- [ ] At least one deep-learning classifier completed
- [ ] Model comparison completed
- [ ] Accuracy + precision + recall + F1 + macro-F1 + weighted-F1 reported
- [ ] Confusion matrix and per-class metrics reported
- [ ] Error analysis completed
- [ ] Final preprocessing + feature extraction + model pipeline saved
- [ ] 3–5 unseen resumes tested
- [ ] Streamlit demo completed
- [ ] README setup and run instructions tested
- [ ] Final branch contains only working code
- [ ] Final GitHub repository reviewed by all 3 members

---

## 📊 Dataset Information
- **Source:** [Google Drive — SAMATRIX RESUMEFORGE 2026](https://drive.google.com/drive/folders/1I3sQZnnjeutdWXM9fNloVTk26zIxXYky)
- **Tabular Data:** `Resume.csv` — 2,484 records, 4 columns (`ID`, `Resume_str`, `Resume_html`, `Category`)
- **PDF Resumes:** 2,500 categorized PDF documents across 24 professional categories
- **24 Target Categories:** ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER

> ⚠️ **Note:** The raw dataset is stored locally in `data/raw/` and is **never committed to GitHub** per competition rules and `.gitignore` configuration.
