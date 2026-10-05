"""
Streamlit Web Application: NueralX Resume Classifier
Author: Savan (Person 3)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)
"""

import sys
import os
import tempfile
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_resume_text
from src.data_loader import extract_text_from_pdf
from src.predict import ResumeClassifierPipeline

st.set_page_config(
    page_title="NueralX | Resume Classification",
    page_icon="📄",
    layout="wide",
)

st.title("🎯 NueralX Resume Classification")
st.markdown(
    "Automated end-to-end resume categorization across **24 industry domains** "
    "powered by Machine Learning and NLP. Built for **SAMATRIX RESUMEFORGE 2026**."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Resume")
    input_mode = st.radio("Choose Input Method:", ["Paste Text", "Upload PDF Document"], horizontal=True)

    resume_text = ""
    if input_mode == "Paste Text":
        resume_text = st.text_area(
            "Paste raw resume content here:",
            height=300,
            placeholder="e.g. Senior Software Engineer experienced in Python, C++, Docker, AWS, microservices architecture...",
        )
    else:
        uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                resume_text = extract_text_from_pdf(tmp_path)
                st.success(f"Successfully extracted {len(resume_text)} characters from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    predict_btn = st.button("🚀 Classify Resume", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Prediction & Diagnostics")
    if predict_btn:
        if not resume_text or len(resume_text.strip()) < 10:
            st.warning("Please provide meaningful resume text before running classification.")
        else:
            with st.spinner("Analyzing vocabulary and classifying domain..."):
                cleaned = clean_resume_text(resume_text)
                pipeline = ResumeClassifierPipeline()
                result = pipeline.predict(resume_text)

                st.metric("Predicted Domain", result["predicted_category"])
                st.metric("Confidence Score", f"{result['confidence'] * 100:.1f}%")

                with st.expander("🔍 Preprocessing Inspection"):
                    st.write("**Cleaned Resume Tokens (First 300 characters):**")
                    st.code(cleaned[:300] + ("..." if len(cleaned) > 300 else ""))
                    st.caption(f"Raw Character Count: {len(resume_text)} | Cleaned Word Count: {len(cleaned.split())}")

st.divider()
st.caption("NueralX Team: Poojan (Data/EDA) | Jeel (Classical ML) | Savan (DL/Integration & Demo)")
