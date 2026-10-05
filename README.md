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
