# Models Directory

This directory stores trained model checkpoints, vectorizers, and serialized inference pipelines.

## Model Artifacts (Ignored by Git)
Large binary files (e.g., `.pkl`, `.joblib`, `.h5`, `.keras`, `.pt`) are excluded from Git to keep the repository lightweight.

### Expected Artifacts for Final Inference
1. **Classical ML Pipeline:**
   - `tfidf_vectorizer.joblib`: Fitted `TfidfVectorizer` (with `ngram_range=(1,2)` or tuned parameters, fitted strictly on training data).
   - `best_ml_model.joblib`: Serialized classical ML model (e.g. Linear SVM or Logistic Regression).
2. **Deep Learning Pipeline:**
   - `word2vec_embeddings.model`: Fitted Word2Vec model on the training corpus.
   - `dl_model.keras` or `dl_model.h5`: Trained neural network model (LSTM/GRU or pooled Dense architecture).
   - `label_encoder.joblib`: Mapping between category string labels and integer indices.
3. **End-to-End Pipeline:**
   - `final_pipeline.joblib`: Complete bundle containing text preprocessor, feature extractor, classifier, and label decoder.
