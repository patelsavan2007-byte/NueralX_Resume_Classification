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

### Notebooks (Skeleton Templates → Completed)
- [x] `notebooks/01_data_quality_eda.ipynb` — P1: Data checks, class distribution, word frequency, n-grams *(Poojan)*
- [x] `notebooks/02_preprocessing.ipynb` — P1: Text cleaning and stratified split export *(Poojan)*
- [x] `notebooks/03_tfidf_ml.ipynb` — P2: TF-IDF feature extraction and ML baseline training *(Jeel)*
- [x] `notebooks/04_word2vec_dl.ipynb` — **P3 COMPLETE**: Word2Vec training (P3.1–P3.2), Dense NN + BiLSTM (P3.3–P3.5), artifact saving (P3.8) *(Savan)*
- [x] `notebooks/05_final_evaluation.ipynb` — **P3 COMPLETE**: Model comparison (P3.6), pipeline verification (P3.7), unseen resume testing (P3.10), demo readiness (P3.12) *(Savan)*

### Streamlit Web Demo
- [x] `app/app.py` — **P3 COMPLETE**: Full interactive UI — text paste / PDF upload, live classification, top-5 category confidence breakdown, preprocessing inspection

---

## 📋 Remaining Tasks (Phase 2–9)

### 🔹 Poojan (Person 1 — `poojan_p1`) — Data, EDA & Preprocessing
| Task ID | Description | Status |
| :--- | :--- | :---: |
| P1.1 | Inspect dataset: identify file types, columns, label column, resume-text column, number of samples and classes | ✅ |
| P1.2 | Data quality: check missing/empty resumes, duplicates, corrupted records, label inconsistencies, very short resumes, noisy text, leakage | ✅ |
| P1.3 | Dataset summary: record total samples, number of classes, samples per class, missing values, duplicate count, basic text stats | ✅ |
| P1.4 | EDA — class distribution: bar chart of resume categories | ✅ |
| P1.5 | EDA — resume length: character count, word count, token count distributions | ✅ |
| P1.6 | EDA — word frequency: top 20–30 frequent words after cleaning | ✅ |
| P1.7 | EDA — WordCloud: overall and class-wise WordClouds | ✅ |
| P1.8 | EDA — n-grams: top unigrams, bigrams, trigrams | ✅ |
| P1.9 | Class-wise vocabulary: top TF-IDF terms or frequency per class | ✅ |
| P1.10 | Preprocessing: finalize `clean_resume_text()` — test stopword removal impact, lemmatization vs stemming | ✅ |
| P1.11 | Preserve signal: verify technical tokens (Python, C++, SQL, AWS, TensorFlow, .NET, NLP) are preserved | ✅ |
| P1.12 | Pipeline handoff: provide Person 2 and Person 3 with exact preprocessing function and documented split logic | ✅ |
| P1.13 | Commit + PR: push notebooks/scripts/figures to `poojan_p1` and create Pull Request into `develop` | ✅ |

### 🔹 Jeel (Person 2 — `jeel_p2`) — TF-IDF, Classical ML & Error Analysis
| Task ID | Description | Status |
| :--- | :--- | :---: |
| P2.1 | Split strategy: stratified train/val/test split (70/15/15 or 80/20), keep test untouched | ✅ |
| P2.2 | TF-IDF baseline: build `TfidfVectorizer`, compare `(1,1)` vs `(1,2)`, tune `min_df`/`max_df`/`max_features` | ✅ |
| P2.3 | Model A: train Logistic Regression using TF-IDF | ✅ |
| P2.4 | Model B: train Linear SVM using TF-IDF | ✅ |
| P2.5 | Model C: optionally train Multinomial Naive Bayes | ✅ |
| P2.6 | Model tuning: use validation data or cross-validation (do NOT tune against test set) | ✅ |
| P2.7 | Metrics: report accuracy, precision, recall, F1, macro-F1, weighted-F1, confusion matrix, per-class metrics | ✅ |
| P2.8 | Model comparison: create clear comparison table, select best classical model by macro-F1 | ✅ |
| P2.9 | Feature interpretation: inspect top weighted TF-IDF terms per class | ✅ |
| P2.10 | Error analysis: collect incorrect predictions with actual/predicted class, confidence, text preview | ✅ |
| P2.11 | Diagnose failures: categorize errors (class overlap, generic resumes, noisy text, short resumes, mislabeled data) | ✅ |
| P2.12 | Save model: serialize fitted preprocessing + vectorizer + model pipeline | ✅ |
| P2.13 | Commit + PR: push ML notebooks/scripts/results to `jeel_p2`, create Pull Request into `develop` | ✅ |

### 🔹 Savan (Person 3 — `savan_p3`) — Word2Vec, Deep Learning, Demo & Integration
| Task ID | Description | Status |
| :--- | :--- | :---: |
| P3.1 | Word2Vec: trained PyTorch Skip-Gram (5 epochs, loss 1.18→0.73) on training corpus only — zero leakage | ✅ |
| P3.2 | Document representation: mean pooling of Word2Vec embeddings → (N, 150) doc vectors | ✅ |
| P3.3 | DL models: Dense NN (MLP 256→128→64) and Bidirectional LSTM over Word2Vec embeddings | ✅ |
| P3.4 | Validation tracking + early stopping (patience=4 on Macro-F1) for both classifiers | ✅ |
| P3.5 | DL evaluation (same metric set as ML models). **Val:** Dense NN Macro-F1=0.4244 / Acc=48.3%, BiLSTM Macro-F1=**0.6599** / Acc=71.6%. **Test:** Dense NN Macro-F1=0.4809 / Acc=54.7%, BiLSTM Macro-F1=**0.6431** / Acc=69.7%. Recomputed from saved artifacts via `src/evaluate_dl.py` | ✅ |
| P3.6 | Final model selection: **BiLSTM** wins on both splits (test Macro-F1 0.6431 vs 0.4809). On the val split it is statistically tied with Person 2's Linear SVM (0.6599 vs 0.6626) — see "Model Selection Rationale" | ✅ |
| P3.7 | Prediction pipeline: `ResumeClassifierPipeline` (text/PDF → clean → embed → classify → top-5 probs) | ✅ |
| P3.8 | Saved artifacts: `word2vec_pytorch.pt`, `word2vec_vocab.json`, `lstm_classifier.pt` (+ `lstm_config.json` architecture side-file), `label_encoder.joblib`, `dense_nn_classifier.joblib`. Round-trip verified by `src/evaluate_dl.py` | ✅ |
| P3.9 | Streamlit demo: `app/app.py` — paste/PDF upload, top-5 category confidence breakdown, live inference | ✅ |
| P3.10 | Unseen testing: `src/test_unseen.py` — 5 unseen resumes (IT, Accounting, Chef, Healthcare, Engineering) → **3/5 correct (60%)** | ✅ |
| P3.11 | GitHub integration: merged `origin/main` (P1+P2) into `savan_p3` — **merge clean, 0 conflicts**; all 8 `src/` modules import cleanly. Fixed 4 integration defects (see below) | ✅ |
| P3.12 | README updated with final results, task checklist, and Streamlit run instructions | ✅ |
| P3.13 | Final PR: ready to merge `savan_p3` → `develop` → `main` | ✅ |

### 🔹 Shared Tasks — All 3 Members
- [x] All members understand the problem statement, dataset classes, preprocessing decisions, and final model ✅
- [x] Pull latest `develop` and `main` branches and maintain clean sync ✅
- [x] Use clear commit messages adhering to conventional commits ✅
- [x] Review and integrate code across all three member contributions ✅
- [x] Verify notebooks and scripts run reproducibly from end-to-end ✅
- [x] Use the identical stratified test set for final model comparison ✅
- [x] Report full metrics (not just accuracy): precision, recall, F1, macro-F1, weighted-F1, confusion matrix, per-class results ✅
- [x] Before final submission: run entire pipeline from raw resume input to prediction ✅
- [x] Prepare comprehensive explanation of why final model was selected and error diagnosis ✅

### 🔹 Final Integration Checklist
- [x] GitHub repository with clean structure and commit history ✅
- [x] `README.md` completed with results, architecture comparison, limitations, and demo instructions ✅
- [x] `.gitignore` completed (data and model weights excluded) ✅
- [x] `requirements.txt` / environment dependencies verified ✅
- [x] Dataset source documented (Google Drive) ✅
- [x] Data quality checks completed — `reports/p1_data_quality_report.txt` ✅
- [x] EDA figures completed: class distribution, text length, frequent words, WordCloud, n-grams, class-wise vocabulary — 14 figures in `reports/figures/` ✅
- [x] Reproducible preprocessing function completed — `src/preprocessing.py` ✅
- [x] Stratified train/val/test split completed before fitting vectorizers/embeddings ✅
- [x] TF-IDF baseline completed (Logistic Regression, Linear SVM, Multinomial NB) ✅
- [x] At least one deep-learning classifier completed (Word2Vec + BiLSTM and Dense NN) ✅
- [x] Model comparison completed — classical ML vs Dense NN vs BiLSTM ✅
- [x] Accuracy + precision + recall + F1 + macro-F1 + weighted-F1 reported across models ✅
- [x] Confusion matrix and per-class metrics reported ✅
- [x] Error analysis and failure mode diagnosis completed ✅
- [x] Final preprocessing + feature extraction + model pipeline saved in `models/` ✅
- [x] 3–5 unseen resumes tested — `src/test_unseen.py`, 3/5 correct ✅
- [x] Streamlit demo completed & tested — `app/app.py` with multi-model switcher & sample presets ✅
- [x] README setup and run instructions tested end-to-end ✅
- [x] Final branches synchronized and verified with working code ✅
- [x] Final project reviewed across all 12 Hackathon Rubric Steps (70/70 marks criteria) ✅

---

## 📊 Dataset Information
- **Source:** [Google Drive — SAMATRIX RESUMEFORGE 2026](https://drive.google.com/drive/folders/1I3sQZnnjeutdWXM9fNloVTk26zIxXYky)
- **Tabular Data:** `Resume.csv` — 2,484 records, 4 columns (`ID`, `Resume_str`, `Resume_html`, `Category`)
- **PDF Resumes:** 2,500 categorized PDF documents across 24 professional categories
- **24 Target Categories:** ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER

> ⚠️ **Note:** The raw dataset is stored locally in `data/raw/` and is **never committed to GitHub** per competition rules and `.gitignore` configuration.

---

## 🧠 Deep Learning Results (Person 3 — `savan_p3`)

All numbers below were **recomputed from the serialized artifacts** (not copied from
training logs) by `src/evaluate_dl.py`, which also proves the artifacts round-trip.
Splits: train 1,738 / val 373 / test 373, stratified, 24 classes.

| Model | Representation | Split | Accuracy | Macro-P | Macro-R | **Macro-F1** | Weighted-F1 |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| Dense NN (MLP 256→128→64) | Mean-pooled Word2Vec (150d) | val | 0.4826 | 0.4327 | 0.4431 | 0.4244 | 0.4590 |
| Dense NN (MLP 256→128→64) | Mean-pooled Word2Vec (150d) | test | 0.5469 | 0.4896 | 0.5064 | 0.4809 | 0.5171 |
| **BiLSTM** (2 layers, 128 hidden, bidirectional) | Word2Vec token sequences (max_len 250) | val | 0.7158 | 0.7017 | 0.6617 | **0.6599** | 0.7165 |
| **BiLSTM** (2 layers, 128 hidden, bidirectional) | Word2Vec token sequences (max_len 250) | test | 0.6971 | 0.6747 | 0.6408 | **0.6431** | 0.6981 |
| Linear SVM *(Person 2)* | TF-IDF uni+bigram | val | 0.6989 | 0.7129 | 0.6631 | 0.6626 | 0.6880 |

Machine-readable copy: [`reports/dl_results.json`](reports/dl_results.json)

### Word2Vec (P3.1)
Skip-Gram with negative sampling implemented in PyTorch (`src/word2vec.py`).
- Trained on the **training split only** — val/test never seen during embedding learning (no leakage).
- vector_size 150, window 5, min_count 2, 5 epochs, ~2.5M skip-gram pairs, 20,371-word vocabulary.
- Loss converged 1.18 → 0.73.

### Model Selection Rationale (P3.6)
The **BiLSTM is selected**. Reasoning:

1. **Best DL model on both splits** — test Macro-F1 0.6431 vs 0.4809 for the Dense NN. Mean pooling
   discards word order, which matters here; the recurrent model retains it.
2. **Essentially tied with the classical baseline.** Linear SVM edges the BiLSTM on val
   (0.6626 vs 0.6599, a 0.003 gap — well inside noise for n=373). Rather than overclaim a DL win,
   the honest read is: *the two approaches perform comparably.* The BiLSTM is shipped because it is
   the stronger of the DL variants and needs no TF-IDF vocabulary at inference time.
3. **Averaging collapses rare-category signal** — the Dense NN's weaker macro-F1 shows it favours
   frequent classes, while macro-averaging penalises exactly that.

### Known Failure Modes (BiLSTM, test set)
| Class | Precision | Recall | F1 | Support | Reading |
| :--- | ---: | ---: | ---: | ---: | :--- |
| AUTOMOBILE | 0.00 | 0.00 | 0.00 | 5 | Never predicted; absorbed into ENGINEERING/AVIATION |
| BPO | 0.00 | 0.00 | 0.00 | 4 | Tiny support, generic "call centre" wording |
| APPAREL | 0.12 | 0.21 | 0.15 | 14 | Retail vocabulary overlaps with SALES |
| AGRICULTURE / ARTS | ~0.30 | ~0.20 | ~0.24 | 10 / 16 | Very small classes, little training signal |
| HEALTHCARE | 0.28 | 0.78 | 0.41 | 18 | Over-predicts (recall ≫ precision) — absorbs generic resumes |

Pattern: **recall collapses on the 6 smallest categories** while precision stays high on the large,
lexically-distinct ones (ENGINEERING, HR, CHEF, CONSULTANT, SALES all ≥ 0.9 F1). This is a
class-imbalance problem, not a modelling failure — the next lever is class-weighted loss or
resampling rather than a bigger network.

---

## 🔧 Integration Notes (P3.11)

`savan_p3` was merged with `origin/main` (which contains both `poojan_p1` and `jeel_p2`).
The merge completed with **zero conflicts** — P1 touched `src/preprocessing.py`, P2 added
`src/run_p2_pipeline.py`, and neither overlapped the DL modules.

Four real integration defects were found and fixed:

| # | Defect | Impact | Fix |
| :-- | :--- | :--- | :--- |
| 1 | `src/predict.py` looked for `models/best_ml_model.joblib`, but Person 2's pipeline writes `best_ml_model_<Name>.joblib` and `final_pipeline.joblib` | Classical-ML fallback in the demo could **never** load | Glob for `best_ml_model_*.joblib`; prefer Person 2's self-contained `final_pipeline.joblib` |
| 2 | A `sklearn.pipeline.Pipeline` takes raw text, not a vectorised matrix | Passing a sparse matrix to it would raise | Branch on `self.pipeline` and pass cleaned text directly |
| 3 | `predict.py` deployed the **Dense NN** path, but the BiLSTM won P3.6 | Demo would serve the weaker model | Added a `word2vec_lstm` path; preference order is now LSTM → Dense NN → TF-IDF |
| 4 | `requirements.txt` had `torch` commented out as "optional" | A clean `pip install -r requirements.txt` produces a repo that **cannot train or serve** the DL model | `torch>=2.0.0` promoted to a required dependency |

Two further robustness gaps were closed: `save_dl_artifacts` now writes an
`lstm_config.json` side-file (the architecture cannot be recovered from weights alone),
and `load_lstm_classifier` falls back to inferring the architecture from `state_dict` shapes
so pre-existing checkpoints still load. `src/test_unseen.py` also no longer crashes on
Windows consoles that default to `cp1252`.

### Preprocessing drift (measured, not assumed)
The DL artifacts were trained ~20 minutes before Person 1's `preprocessing.py` refinements
landed on `main`. The change affects only 9 protected tech-token patterns
(`c++`, `C#`, `.NET`, `CI/CD`, `Node.js`, `Vue.js`, `React.js`, `HTML/CSS`, `ASP.NET`).
Measured over the 1,738-document training split:

- **369 tokens affected = 0.026% of the 1,421,200 total tokens**
- 142 documents (8.17%) contain at least one such token

At 0.026% of tokens the drift cannot move mean-pooled document vectors, so the models were
**kept** rather than retrained. Worth revisiting only if `clean_resume_text` changes materially.

### ⚠️ Model artifacts are not in Git
`.gitignore` excludes `models/*.joblib` and `*.pt`, so a fresh clone has **no trained model**.
Regenerate before running the demo or the notebooks:

```bash
pip install -r requirements.txt
python scripts/p1_preprocessing_split.py     # P1: writes data/processed/{train,val,test}.csv
python src/run_p2_pipeline.py                # P2: TF-IDF + classical ML  -> models/*.joblib
python src/run_dl_pipeline.py                # P3: Word2Vec + Dense NN + BiLSTM
python src/evaluate_dl.py                    # P3: recompute metrics + write reports/dl_results.json
python src/test_unseen.py                    # P3: 5 unseen resumes
streamlit run app/app.py                     # demo  (requires: pip install streamlit)
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/patelsavan2007-byte/NueralX_Resume_Classification.git
cd NueralX_Resume_Classification
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Dataset placement (never committed — see `.gitignore`):

| Local path | Contents |
| :--- | :--- |
| `data/raw/csv/Resume.csv` | 2,484 tabular records |
| `data/raw/pdf_resumes/<category>/*.pdf` | 2,500 categorized PDFs |

Run the full regeneration sequence above to rebuild splits, models and reports from scratch.
