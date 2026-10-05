"""
Streamlit Web Application: NueralX Resume Classifier
Author: Savan (Person 3)
Task P3.9: Interactive UI for resume classification via text paste or PDF upload.

Run with: streamlit run app/app.py
"""

import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
from src.preprocessing import clean_resume_text
from src.data_loader import extract_text_from_pdf
from src.predict import ResumeClassifierPipeline

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="NueralX | Resume Classification",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS for a polished look ─────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .category-badge {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f2937;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        display: inline-block;
    }
    .confidence-high { color: #059669; }
    .confidence-mid  { color: #d97706; }
    .confidence-low  { color: #dc2626; }
    .stats-box {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎯 NueralX Resume Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Automated resume categorization across <strong>24 industry domains</strong> '
    'powered by Word2Vec + Deep Learning &mdash; SAMATRIX RESUMEFORGE 2026</div>',
    unsafe_allow_html=True,
)

# ── Model Cache ────────────────────────────────────────────────
@st.cache_resource
def get_pipeline(model_key: str):
    return ResumeClassifierPipeline(model_type=model_key)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Architecture")
    model_options = {
        "🏆 Word2Vec + BiLSTM (Deep Learning)": "bilstm",
        "⚡ TF-IDF + Linear SVM (Best Classical)": "tfidf",
        "🧠 Word2Vec + Dense NN (MLP Baseline)": "dense",
        "🤖 Auto (Best Available Model)": "auto",
    }
    selected_option = st.selectbox(
        "Select Classification Model:",
        list(model_options.keys()),
        index=0,
        help="Switch between Deep Learning (BiLSTM), Classical ML (Linear SVM), or Dense NN.",
    )
    chosen_model_type = model_options[selected_option]

    st.divider()
    st.header("ℹ️ About the Project")
    st.markdown(
        "This demo classifies resumes into one of **24 professional categories** "
        "using NLP pipelines trained on 2,484 labeled resumes across 24 domains."
    )
    st.caption("• **Best DL:** BiLSTM (Val Macro-F1: 0.6599, Acc: 71.6%)\n• **Best ML:** Linear SVM (Val Macro-F1: 0.6518, Acc: 68.8%)")

    st.divider()
    st.markdown("**🏷️ Supported Categories (24):**")
    categories = [
        "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS",
        "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT",
        "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
        "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
        "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
    ]
    with st.expander("View All Categories", expanded=False):
        for cat in categories:
            st.caption(f"• {cat}")

    st.divider()
    st.caption("Team NueralX: Poojan (Data/EDA) | Jeel (ML) | Savan (DL/Demo)")

# ── Main Content ───────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Input Resume")
    
    # Quick Sample Presets
    SAMPLE_RESUMES = {
        "None (Empty)": "",
        "IT / Software Engineer": (
            "Senior Software Engineer with 8+ years experience in Python, Java, React, Docker, "
            "Kubernetes, microservices architecture, PostgreSQL, AWS cloud, and CI/CD pipelines. "
            "Led backend architecture serving 15M daily requests with high availability."
        ),
        "Chef / Culinary": (
            "Executive Chef with 10 years experience in French and Italian culinary cuisine. "
            "Menu design, food safety, inventory management, kitchen operations, fine dining catering, "
            "HACCP certified, banquet coordination, recipe development."
        ),
        "Accountant / Finance": (
            "Senior Accountant with CPA certification. Expertise in GAAP, corporate tax filing, "
            "financial audits, QuickBooks, balance sheets, payroll reconciliation, general ledger, "
            "and accounts payable/receivable management."
        ),
        "Healthcare / Nursing": (
            "Registered Nurse (RN) with 6 years experience in ICU and emergency care. Patient triage, "
            "vital signs monitoring, medication administration, BLS/ACLS certified, clinical documentation, "
            "and multidisciplinary hospital team coordination."
        ),
    }

    sample_choice = st.selectbox("Quick Test with Sample Resumes:", list(SAMPLE_RESUMES.keys()), index=0)

    input_mode = st.radio(
        "Choose input method:",
        ["✏️ Paste Text", "📁 Upload PDF"],
        horizontal=True,
    )

    resume_text = ""

    if input_mode == "✏️ Paste Text":
        default_val = SAMPLE_RESUMES[sample_choice] if sample_choice != "None (Empty)" else ""
        resume_text = st.text_area(
            "Paste raw resume content here:",
            value=default_val,
            height=320,
            placeholder=(
                "e.g. Senior Software Engineer with 8+ years of experience in Python, "
                "Java, C++, AWS, Docker, Kubernetes, microservices architecture, CI/CD pipelines. "
                "Led a team of 12 engineers building scalable distributed systems..."
            ),
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a resume PDF document:",
            type=["pdf"],
            help="Supports single-page or multi-page PDF resumes.",
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                resume_text = extract_text_from_pdf(tmp_path)
                st.success(f"✅ Extracted {len(resume_text):,} characters from **{uploaded_file.name}**")
            except Exception as e:
                st.error(f"❌ Failed to read PDF: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    predict_btn = st.button(
        "🚀 Classify Resume",
        type="primary",
        use_container_width=True,
        disabled=len(resume_text.strip()) < 10,
    )

with col2:
    st.subheader("📊 Classification Result")

    if predict_btn and resume_text.strip():
        with st.spinner(f"🔍 Analyzing resume with {selected_option}..."):
            pipeline = get_pipeline(chosen_model_type)
            result = pipeline.predict(resume_text)

        # Category
        st.markdown(f'<div class="category-badge">🏷️ {result["predicted_category"]}</div>',
                    unsafe_allow_html=True)
        st.write("")

        # Confidence
        conf = result.get("confidence", 0)
        conf_pct = conf * 100
        if conf_pct >= 70:
            conf_class = "confidence-high"
            conf_icon = "🟢"
        elif conf_pct >= 40:
            conf_class = "confidence-mid"
            conf_icon = "🟡"
        else:
            conf_class = "confidence-low"
            conf_icon = "🔴"

        st.markdown(
            f'{conf_icon} **Confidence:** <span class="{conf_class}">{conf_pct:.1f}%</span>',
            unsafe_allow_html=True,
        )

        # Pipeline info
        ptype = result.get("pipeline_type", "N/A")
        st.caption(f"Pipeline: {ptype}")

        # Top 5 category breakdown
        top_cats = result.get("top_categories", [])
        if top_cats:
            st.markdown("##### 🏆 Top Candidate Categories")
            for item in top_cats:
                cat_name = item["category"]
                cat_conf = item["confidence"]
                col_name, col_pct = st.columns([3, 1])
                col_name.write(f"• **{cat_name}**")
                col_pct.write(f"**{cat_conf * 100:.1f}%**")
                st.progress(float(np.clip(cat_conf, 0.0, 1.0)))

        st.divider()

        # Stats
        st.markdown("**📈 Resume Statistics:**")
        stat_col1, stat_col2 = st.columns(2)
        stat_col1.metric("Raw Characters", f"{result.get('raw_length', 0):,}")
        stat_col2.metric("Cleaned Words", f"{result.get('cleaned_word_count', 0):,}")

        # Preprocessing preview
        with st.expander("🔍 Preprocessing Inspection", expanded=False):
            cleaned = clean_resume_text(resume_text)
            st.code(cleaned[:500] + ("..." if len(cleaned) > 500 else ""), language=None)

    elif predict_btn:
        st.warning("⚠️ Please provide meaningful resume text (at least 10 characters).")
    else:
        st.info("👈 Paste or upload a resume, then click **Classify Resume** to see the prediction.")

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption("NueralX Resume Classification System | SAMATRIX RESUMEFORGE 2026")
