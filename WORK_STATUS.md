# 📊 NueralX Resume Classification — Task & Project Progress Status

> **Branch:** `jeel_p2` (Person 2 - Jeel)  
> **Last Updated:** October 5, 2026  
> **Repository:** `D:\Resume_NuralX\NueralX_Resume_Classification`

---

## 🟢 1. WORK DONE (COMPLETED TASKS)

### 🔹 Environment & Data Setup (Phase 1)
- [x] **Dataset Extracted & Verified:** Extracted `Resume.csv` (2,484 records across 24 categories) into `data/raw/csv/`.
- [x] **Virtual Environment Configured:** Python 3.13 environment active with all required dependencies (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`).

---

### 🔹 Person 2 (Jeel — `jeel_p2`) Deliverables (P2.1 – P2.12)

| Task ID | Task Description | Status | Output Artifacts / Implementation |
| :--- | :--- | :---: | :--- |
| **P2.1** | **Stratified Split Strategy** | ✅ Completed | 70% Train (1,738), 15% Val (372), 15% Test (374) split preserving 24 class ratios saved to `data/processed/`. |
| **P2.2** | **TF-IDF Feature Extraction** | ✅ Completed | Built `TfidfVectorizer` with `sublinear_tf=True`, `min_df=2`, `max_df=0.9`, `max_features=10,000`, comparing unigrams `(1,1)` vs unigrams+bigrams `(1,2)`. |
| **P2.3** | **Model A — Logistic Regression** | ✅ Completed | Trained baseline Logistic Regression (`C=1.0`, `solver='lbfgs'`). Validation Acc: `60.48%`, Macro-F1: `0.5601`. |
| **P2.4** | **Model B — Linear SVM** | ✅ Completed | Trained SGDClassifier (`loss='hinge'`, `alpha=1e-4`, L2 penalty). Validation Acc: `62.90%`, Macro-F1: `0.5884`. **Selected as Best Classical Model.** |
| **P2.5** | **Model C — Multinomial Naive Bayes** | ✅ Completed | Trained Multinomial NB (`alpha=0.1`). Validation Acc: `56.72%`, Macro-F1: `0.5215`. |
| **P2.6** | **Hyperparameter Tuning & 5-Fold CV** | ✅ Completed | Evaluated models using 5-Fold Stratified Cross-Validation on training set. Tuned `C` grid `[0.1, 1.0, 10.0]` and `alpha` grid `[1e-5, 1e-4, 1e-3]`. |
| **P2.7** | **Comprehensive Metrics Suite** | ✅ Completed | Calculated Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Confusion Matrices, and Per-Class Breakdown across all 24 categories. |
| **P2.8** | **Benchmark Comparison Report** | ✅ Completed | Generated evaluation summary and updated `reports/final_results.md`. |
| **P2.9** | **Top TF-IDF Term Interpretation** | ✅ Completed | Extracted top 15 weighted n-grams per class. Saved to `reports/top_tfidf_terms_per_class.json` & plot `reports/figures/top_tfidf_terms_per_class.png`. |
| **P2.10** | **Error Analysis & Confusion Matrix** | ✅ Completed | Extracted misclassified samples (112 errors on val set) and identified top confusion pairs (e.g. Finance vs Accountant, Consultant vs Accountant, Digital Media vs PR). |
| **P2.11** | **Failure Diagnosis & Categorization** | ✅ Completed | Categorized errors: Generic/Low-Signal text (76), Class Overlap (16), Ambiguous/Other (96). Saved to `reports/error_diagnosis.json`. |
| **P2.12** | **Model Serialization & Pipeline** | ✅ Completed | Saved `models/tfidf_vectorizer.joblib`, `models/best_ml_model_Linear_SVM.joblib`, `models/final_pipeline.joblib`, and `models/training_metadata.json`. |
| **Automation** | **Standalone Pipeline Script** | ✅ Completed | Created end-to-end executable script `src/run_p2_pipeline.py`. |
| **Notebook** | **Jupyter Notebook (`03_tfidf_ml.ipynb`)** | ✅ Completed | Fully populated `notebooks/03_tfidf_ml.ipynb` with interactive code cells, plots, markdown commentary, and results tables. |

---

## 🟡 2. WORK REMAINING (PENDING TASKS)

### 🔹 Person 2 (Jeel — `jeel_p2`) Finalization
- [ ] **P2.13 — Git Commit & Push:**
  - Stage all generated scripts, notebook updates, reports, and model metadata.
  - Commit changes to `jeel_p2` branch.
  - Push `jeel_p2` branch to remote repository and create Pull Request into `develop` (*pending user explicit approval to push*).

---

### 🔹 Person 1 (Poojan — `poojan_p1`) — Data, EDA & Preprocessing
- [ ] **P1.1 – P1.3:** Data quality audit (check missing values, duplicate resumes, noisy text).
- [ ] **P1.4 – P1.8:** Exploratory Data Analysis (EDA bar charts, word count distributions, unigram/bigram/trigram frequencies, overall and class-wise WordClouds).
- [ ] **P1.9 – P1.12:** Clean text pipeline verification & handoff to Person 2 and Person 3.
- [ ] **P1.13:** Commit & Pull Request into `develop`.

---

### 🔹 Person 3 (Savan — `savan_p3`) — Deep Learning, Demo & Integration
- [ ] **P3.1 – P3.5:** Train Word2Vec embeddings on training corpus & build Deep Learning classifier (LSTM/GRU or Dense Neural Net).
- [ ] **P3.6 – P3.8:** Compare DL vs Classical ML model, build end-to-end inference pipeline, and serialize DL artifacts.
- [ ] **P3.9 – P3.10:** Build Streamlit web interface (`app/app.py`) supporting resume paste & PDF upload, and test on 3–5 unseen resumes.
- [ ] **P3.11 – P3.13:** Merge P1, P2, and P3 into `develop`, perform final end-to-end validation, update README, and merge `develop` into `main`.

---

## 📈 Classical ML Benchmark Summary (Jeel's Ownership)

| Model | Val Accuracy | Val Precision | Val Recall | Val Macro-F1 | Val Weighted-F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear SVM (SGDClassifier)** | **62.90%** | **0.6558** | **0.6290** | **0.5884** | **0.6120** | 🏆 **BEST MODEL** |
| Logistic Regression | 60.48% | 0.6190 | 0.6048 | 0.5601 | 0.5878 | Baseline |
| Multinomial Naive Bayes | 56.72% | 0.5960 | 0.5672 | 0.5215 | 0.5401 | Baseline |

---

## 📁 Key Generated Artifacts Summary

1. **Pipeline Execution Script:** [src/run_p2_pipeline.py](file:///D:/Resume_NuralX/NueralX_Resume_Classification/src/run_p2_pipeline.py)
2. **Notebook:** [notebooks/03_tfidf_ml.ipynb](file:///D:/Resume_NuralX/NueralX_Resume_Classification/notebooks/03_tfidf_ml.ipynb)
3. **Serialized Pipeline:** `models/final_pipeline.joblib` & `models/tfidf_vectorizer.joblib`
4. **Best Model Weights:** `models/best_ml_model_Linear_SVM.joblib`
5. **Evaluation Reports:** [reports/final_results.md](file:///D:/Resume_NuralX/NueralX_Resume_Classification/reports/final_results.md), `reports/error_diagnosis.json`, `reports/top_tfidf_terms_per_class.json`
6. **Plots & Visualizations:** `reports/figures/top_tfidf_terms_per_class.png`, `reports/figures/confusion_matrix_Linear_SVM.png`
