"""
Streamlit Web Application: NueralX Resume Classifier
Author: Savan (Person 3) — Team NueralX
Task P3.9: Interactive UI for resume classification via text paste or PDF upload.

Run with: streamlit run app/app.py
"""

import sys
import os
import json
import tempfile

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import numpy as np
import pandas as pd
from src.preprocessing import clean_resume_text
from src.data_loader import extract_text_from_pdf
from src.predict import ResumeClassifierPipeline

# ── Constants ──────────────────────────────────────────────────
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

CATEGORIES = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS",
    "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT",
    "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
    "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
]


# ── Helpers: load artifacts once ───────────────────────────────
@st.cache_resource
def get_pipeline(model_key: str):
    return ResumeClassifierPipeline(model_type=model_key)


@st.cache_data
def load_dl_results():
    path = os.path.join(REPORTS_DIR, "dl_results.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_error_diagnosis():
    path = os.path.join(REPORTS_DIR, "error_diagnosis.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_training_metadata():
    path = os.path.join(MODELS_DIR, "training_metadata.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_per_class_metrics():
    """Parse the per-class table from final_results.md."""
    path = os.path.join(REPORTS_DIR, "final_results.md")
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        in_table = False
        for line in lines:
            line = line.strip()
            if line.startswith("| Category"):
                in_table = True
                continue
            if in_table and line.startswith("| :---"):
                continue
            if in_table and line.startswith("|"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 5:
                    try:
                        rows.append({
                            "Category": parts[0],
                            "Precision": float(parts[1]),
                            "Recall": float(parts[2]),
                            "F1-Score": float(parts[3]),
                            "Support": int(parts[4]),
                        })
                    except ValueError:
                        pass
            elif in_table and not line.startswith("|"):
                break
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


@st.cache_data
def load_data_quality():
    path = os.path.join(REPORTS_DIR, "p1_data_quality_report.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="NueralX | Resume Classification — SAMATRIX RESUMEFORGE 2026",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }
    .category-badge {
        font-size: 2rem;
        font-weight: 800;
        color: #1e293b;
        padding: 0.8rem 1.2rem;
        background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 50%, #ede9fe 100%);
        border-radius: 16px;
        border-left: 5px solid #6366f1;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.15);
    }
    .confidence-high { color: #059669; font-weight: 700; }
    .confidence-mid  { color: #d97706; font-weight: 700; }
    .confidence-low  { color: #dc2626; font-weight: 700; }

    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card h3 {
        margin: 0 0 0.3rem 0;
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
    }

    .pipeline-step {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.15rem;
    }
    .pipeline-arrow {
        display: inline-block;
        color: #a78bfa;
        font-size: 1.1rem;
        margin: 0 0.15rem;
        vertical-align: middle;
    }

    .low-conf-warning {
        background: linear-gradient(135deg, #fef3c7, #fffbeb);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #92400e;
    }

    .taxonomy-notice {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.6rem 0 0.9rem 0;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.45;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }

    .ood-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #f59e0b;
        border-left: 5px solid #d97706;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin: 0.7rem 0 0.9rem 0;
        font-size: 0.88rem;
        color: #92400e;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.08);
    }

    .med-conf-notice {
        background: #fefce8;
        border-left: 4px solid #eab308;
        border-radius: 8px;
        padding: 0.65rem 0.9rem;
        margin: 0.6rem 0 0.8rem 0;
        font-size: 0.875rem;
        color: #854d0e;
        line-height: 1.45;
    }

    .scope-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        color: #4f46e5;
        background: #eef2ff;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600 !important;
    }

    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #334155;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎯 NueralX Resume Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">End-to-end NLP pipeline classifying resumes across '
    '<strong>24 professional domains</strong> using TF-IDF + Classical ML and '
    'Word2Vec + Deep Learning &mdash; <strong>SAMATRIX RESUMEFORGE 2026</strong></div>',
    unsafe_allow_html=True,
)

# ── Pipeline Visualization ─────────────────────────────────────
st.markdown(
    '<div style="margin-bottom:1.2rem;">'
    '<span class="pipeline-step">📄 Raw Resume (Text / PDF)</span>'
    '<span class="pipeline-arrow">→</span>'
    '<span class="pipeline-step">🧹 Preprocessing</span>'
    '<span class="pipeline-arrow">→</span>'
    '<span class="pipeline-step">🔢 Feature Extraction</span>'
    '<span class="pipeline-arrow">→</span>'
    '<span class="pipeline-step">🧠 Model Inference</span>'
    '<span class="pipeline-arrow">→</span>'
    '<span class="pipeline-step">🏷️ Predicted Category</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Selection")
    model_options = {
        "🏆 BiLSTM — Deep Learning (Best Overall)": "bilstm",
        "⚡ Linear SVM — Classical ML (Best Classical)": "tfidf",
        "🧠 Dense NN — MLP Baseline": "dense",
    }
    selected_label = st.radio(
        "Choose classification model:",
        list(model_options.keys()),
        index=0,
    )
    chosen_model_type = model_options[selected_label]

    st.divider()

    # Key metrics summary
    st.markdown("### 📊 Model Performance")
    dl = load_dl_results()
    bilstm_val = dl.get("dl_results", {}).get("bilstm", {}).get("val", {})
    st.markdown(
        f"**BiLSTM (DL):** {bilstm_val.get('accuracy', 0):.1%} Acc · "
        f"{bilstm_val.get('macro_f1', 0):.4f} F1"
    )
    st.markdown("**Linear SVM (ML):** 68.82% Acc · 0.6518 F1")
    st.caption("Evaluated on identical stratified validation set (373 resumes, 24 classes)")

    st.divider()
    st.markdown("### 📋 Dataset Summary")
    st.caption(
        "• **2,484** labeled resumes\n"
        "• **24** professional categories\n"
        "• **70/15/15** stratified split\n"
        "• Train: 1,738 · Val: 373 · Test: 373"
    )

    st.divider()
    with st.expander("🏷️ All 24 Categories"):
        cols = st.columns(2)
        for i, cat in enumerate(CATEGORIES):
            cols[i % 2].caption(f"• {cat}")

    st.divider()
    st.caption("**Team NueralX**")
    st.caption("Poojan (Data & EDA) · Jeel (Classical ML) · Savan (DL & Demo)")

# ── Main Tabs ──────────────────────────────────────────────────
tab_predict, tab_eval, tab_eda, tab_errors = st.tabs([
    "🚀 Classify Resume",
    "📊 Model Evaluation & Comparison",
    "📈 Exploratory Data Analysis",
    "🔍 Error Analysis & Confusion Matrices",
])


# ════════════════════════════════════════════════════════════════
# TAB 1: CLASSIFY RESUME
# ════════════════════════════════════════════════════════════════
with tab_predict:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-header">📥 Input Resume</div>', unsafe_allow_html=True)

        SAMPLE_RESUMES = {
            "— Select a sample (optional) —": "",
            "🖥️ IT / Software Engineer": (
                "Senior Software Engineer with 8+ years of experience in full-stack development. "
                "Proficient in Python, Java, JavaScript, React, Node.js, and AWS cloud services. "
                "Led a team of 12 engineers building scalable microservices architecture using Docker, "
                "Kubernetes, and CI/CD pipelines. Implemented real-time data processing systems "
                "handling 10M+ daily transactions. Experience with PostgreSQL, MongoDB, Redis. "
                "Bachelor of Science in Computer Science from MIT. Certified AWS Solutions Architect."
            ),
            "👨‍🍳 Chef / Culinary": (
                "Executive Chef with 10 years experience managing fine dining kitchen operations. "
                "Expert in French, Italian, and Asian culinary techniques. Menu design and costing, "
                "food safety protocols, HACCP certified. Managed team of 25 kitchen staff across "
                "banquet and à la carte services. Recipe development, inventory management, and "
                "vendor negotiations. Catering for events up to 500 guests."
            ),
            "💰 Accountant / Finance": (
                "Senior Accountant with CPA certification and 7 years experience in corporate finance. "
                "Expertise in GAAP compliance, corporate tax preparation, financial statement audits, "
                "QuickBooks Enterprise, SAP FICO, balance sheet reconciliation, payroll processing, "
                "accounts payable/receivable management, and general ledger maintenance. Managed "
                "annual budgets exceeding $12M."
            ),
            "🏥 Healthcare / Nursing": (
                "Registered Nurse (RN) with 6 years experience in ICU and emergency care at level I "
                "trauma center. Patient triage, vital signs monitoring, medication administration, "
                "wound care, IV insertion, BLS/ACLS certified. Strong clinical documentation skills "
                "with Epic EMR. Collaborative team player in multidisciplinary hospital teams."
            ),
            "🏗️ Construction / Engineering": (
                "Civil Project Engineer with 5 years experience in commercial construction. Proficient "
                "in AutoCAD, Revit, structural analysis, OSHA compliance, project scheduling with "
                "Primavera P6, quantity surveying, cost estimation, concrete design, steel detailing, "
                "and site inspection. Managed projects valued at $8M+. PE licensed."
            ),
        }

        sample_choice = st.selectbox("Quick test with sample resumes:", list(SAMPLE_RESUMES.keys()))

        input_mode = st.radio("Choose input method:", ["✏️ Paste Text", "📁 Upload PDF"], horizontal=True)

        resume_text = ""
        if input_mode == "✏️ Paste Text":
            default_val = SAMPLE_RESUMES.get(sample_choice, "")
            resume_text = st.text_area(
                "Paste raw resume content below:",
                value=default_val,
                height=300,
                placeholder="Paste the full resume text here...",
            )
        else:
            uploaded_file = st.file_uploader("Upload a PDF resume:", type=["pdf"])
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    resume_text = extract_text_from_pdf(tmp_path)
                    st.success(f"✅ Extracted **{len(resume_text):,}** characters from **{uploaded_file.name}**")
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

    with col_result:
        st.markdown('<div class="section-header">📊 Classification Result</div>', unsafe_allow_html=True)

        if predict_btn and resume_text.strip():
            with st.spinner(f"Analyzing with {selected_label.split('—')[0].strip()}..."):
                pipeline = get_pipeline(chosen_model_type)
                result = pipeline.predict(resume_text)

            pred_cat = result.get("predicted_category", "UNKNOWN")
            conf = result.get("confidence", 0)
            conf_pct = conf * 100
            ptype = result.get("pipeline_type", "N/A")
            top_cats = result.get("top_categories", [])

            # ── Predicted Category Scope & Badge ──
            st.markdown('<div class="scope-pill">🎯 Predicted Dataset Category (1 of 24 Classes)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="category-badge">🏷️ {pred_cat}</div>', unsafe_allow_html=True)
            st.caption("ℹ️ *Mapped to the closest pre-defined industry category from the 24 training classes.*")

            # ── Dataset Scope Explanation Callout ──
            st.markdown(
                '<div class="taxonomy-notice">'
                '📌 <strong>Dataset Scope Note:</strong> This dataset contains 24 job categories and does not include '
                'AI/ML or Data Science; predictions are therefore limited to the available categories.'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── Confidence with color coding ──
            if conf_pct >= 60:
                conf_class, conf_icon, conf_label = "confidence-high", "🟢", "High"
            elif conf_pct >= 35:
                conf_class, conf_icon, conf_label = "confidence-mid", "🟡", "Moderate"
            else:
                conf_class, conf_icon, conf_label = "confidence-low", "🔴", "Low"

            st.markdown(
                f'{conf_icon} **Confidence:** '
                f'<span class="{conf_class}">{conf_pct:.1f}% ({conf_label})</span>',
                unsafe_allow_html=True,
            )

            # ── Low Confidence / Out-of-Distribution Warning ──
            if conf_pct < 35:
                st.markdown(
                    f'<div class="ood-warning">'
                    f'<strong>⚠️ Low Confidence / Out-of-Distribution Alert ({conf_pct:.1f}%):</strong><br>'
                    f'The model is uncertain about this classification. The resume likely contains modern technical '
                    f'terminology (such as AI/ML, Data Science, or specialized frameworks), spans multiple functional '
                    f'domains, or differs from the 2018 benchmark training corpus.<br>'
                    f'<span style="font-weight:600;">👉 Please consult the <strong>Top-5 Candidate Categories</strong> below for competing domain probabilities.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif conf_pct < 60:
                st.markdown(
                    f'<div class="med-conf-notice">'
                    f'🟡 <strong>Moderate Confidence:</strong> The model identified relevant keyword associations for '
                    f'<strong>{pred_cat}</strong>, but secondary domains share noticeable probability. Check the Top-5 distribution below.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.write("")

            # ── Pipeline path visualization ──
            if "lstm" in ptype:
                feat_label = "Word2Vec Embedding"
                model_label = "BiLSTM Classifier"
            elif "dense" in ptype:
                feat_label = "Mean-Pooled Word2Vec"
                model_label = "Dense NN (MLP)"
            else:
                feat_label = "TF-IDF Vectorizer"
                model_label = "Linear SVM"

            st.caption(f"Pipeline: **Text** → **Preprocessing** → **{feat_label}** → **{model_label}** → **{pred_cat}**")

            st.divider()

            # ── Top 5 Categories with bars ──
            if top_cats:
                st.markdown("##### 🏆 Top-5 Candidate Categories")
                for i, item in enumerate(top_cats):
                    cat_name = item["category"]
                    cat_conf = item["confidence"]
                    is_top = (i == 0)
                    prefix = "**→**" if is_top else "  "
                    label_col, bar_col, pct_col = st.columns([2.5, 4, 1])
                    label_col.markdown(f"{prefix} {'**' + cat_name + '**' if is_top else cat_name}")
                    bar_col.progress(float(np.clip(cat_conf, 0.0, 1.0)))
                    pct_col.markdown(f"**{cat_conf * 100:.1f}%**" if is_top else f"{cat_conf * 100:.1f}%")

            st.divider()

            # ── Resume Statistics ──
            st.markdown("##### 📈 Resume Statistics")
            s1, s2, s3 = st.columns(3)
            s1.metric("Raw Characters", f"{result.get('raw_length', 0):,}")
            s2.metric("Cleaned Words", f"{result.get('cleaned_word_count', 0):,}")
            s3.metric("Model Used", ptype.replace("_", " ").title())

            # ── Preprocessing Inspector ──
            with st.expander("🔍 Preprocessing Inspection (cleaned text)", expanded=False):
                cleaned = clean_resume_text(resume_text)
                st.code(cleaned[:600] + ("..." if len(cleaned) > 600 else ""), language=None)

            # ── Cross-Model Consensus (Quick Multi-Model Comparison) ──
            with st.expander("⚡ Cross-Model Consensus (Compare All 3 Models on this Resume)", expanded=False):
                st.caption("Side-by-side predictions across all 3 benchmark models on the exact same resume text:")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    p_bilstm = get_pipeline("bilstm")
                    res_bilstm = p_bilstm.predict(resume_text)
                    st.markdown("**🏆 BiLSTM (Best DL)**")
                    st.markdown(f"🏷️ `{res_bilstm.get('predicted_category', 'N/A')}`")
                    st.caption(f"Confidence: **{res_bilstm.get('confidence', 0)*100:.1f}%**")
                with col_m2:
                    p_svm = get_pipeline("tfidf")
                    res_svm = p_svm.predict(resume_text)
                    st.markdown("**⚡ Linear SVM (Best ML)**")
                    st.markdown(f"🏷️ `{res_svm.get('predicted_category', 'N/A')}`")
                    st.caption(f"Confidence: **{res_svm.get('confidence', 0)*100:.1f}%**")
                with col_m3:
                    p_dense = get_pipeline("dense")
                    res_dense = p_dense.predict(resume_text)
                    st.markdown("**🧠 Dense NN (MLP)**")
                    st.markdown(f"🏷️ `{res_dense.get('predicted_category', 'N/A')}`")
                    st.caption(f"Confidence: **{res_dense.get('confidence', 0)*100:.1f}%**")

        elif predict_btn:
            st.warning("⚠️ Please provide meaningful resume text (at least 10 characters).")
        else:
            st.info("👈 Paste or upload a resume, then click **Classify Resume** to see the prediction.")
            st.markdown("")
            st.markdown("**How it works:**")
            st.markdown(
                "1. **Input:** Paste resume text or upload a PDF document.\n"
                "2. **Preprocessing:** Text is cleaned, normalized, and technical tokens are preserved.\n"
                "3. **Feature Extraction:** Word2Vec embeddings (for DL) or TF-IDF vectors (for ML).\n"
                "4. **Classification:** The selected model predicts one of 24 professional categories.\n"
                "5. **Output:** Predicted category with confidence score and Top-5 candidates."
            )


# ════════════════════════════════════════════════════════════════
# TAB 2: MODEL EVALUATION & COMPARISON
# ════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-header">📊 Model Evaluation & Benchmark Comparison</div>',
                unsafe_allow_html=True)

    dl = load_dl_results()
    bilstm_val = dl.get("dl_results", {}).get("bilstm", {}).get("val", {})
    bilstm_test = dl.get("dl_results", {}).get("bilstm", {}).get("test", {})
    dense_val = dl.get("dl_results", {}).get("dense_nn", {}).get("val", {})
    dense_test = dl.get("dl_results", {}).get("dense_nn", {}).get("test", {})

    # ── Key Metrics Cards ──
    st.markdown("#### 🏆 Champion Models (Validation Set)")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        '<div class="metric-card"><h3>Best Overall Model</h3>'
        '<div class="value">BiLSTM</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card"><h3>Val Accuracy</h3>'
        f'<div class="value">{bilstm_val.get("accuracy", 0):.1%}</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card"><h3>Val Macro-F1</h3>'
        f'<div class="value">{bilstm_val.get("macro_f1", 0):.4f}</div></div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        '<div class="metric-card"><h3>Total Categories</h3>'
        '<div class="value">24</div></div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # ── Full Comparison Table ──
    st.markdown("#### 📋 Complete Model Comparison Table")

    comparison_data = []
    # BiLSTM
    comparison_data.append({
        "Model": "🏆 BiLSTM (Bidirectional LSTM)",
        "Features": "Word2Vec (150-d) Sequences",
        "Accuracy": f"{bilstm_val.get('accuracy', 0):.4f}",
        "Macro Precision": f"{bilstm_val.get('macro_precision', 0):.4f}",
        "Macro Recall": f"{bilstm_val.get('macro_recall', 0):.4f}",
        "Macro F1": f"{bilstm_val.get('macro_f1', 0):.4f}",
        "Weighted F1": f"{bilstm_val.get('weighted_f1', 0):.4f}",
    })
    # Linear SVM
    comparison_data.append({
        "Model": "⚡ Linear SVM (Best Classical)",
        "Features": "TF-IDF (Unigram+Bigram)",
        "Accuracy": "0.6882",
        "Macro Precision": "0.6768",
        "Macro Recall": "0.6536",
        "Macro F1": "0.6518",
        "Weighted F1": "0.6784",
    })
    # Logistic Regression Tuned
    comparison_data.append({
        "Model": "Logistic Regression (Tuned)",
        "Features": "TF-IDF (Unigram+Bigram)",
        "Accuracy": "0.6505",
        "Macro Precision": "0.6740",
        "Macro Recall": "0.6134",
        "Macro F1": "0.6146",
        "Weighted F1": "0.6413",
    })
    # Logistic Regression Base
    comparison_data.append({
        "Model": "Logistic Regression (Base)",
        "Features": "TF-IDF (Unigram+Bigram)",
        "Accuracy": "0.6183",
        "Macro Precision": "0.6460",
        "Macro Recall": "0.5834",
        "Macro F1": "0.5795",
        "Weighted F1": "0.6032",
    })
    # Multinomial NB
    comparison_data.append({
        "Model": "Multinomial Naive Bayes",
        "Features": "TF-IDF (Unigram+Bigram)",
        "Accuracy": "0.5511",
        "Macro Precision": "0.5386",
        "Macro Recall": "0.5068",
        "Macro F1": "0.4842",
        "Weighted F1": "0.5214",
    })
    # Dense NN
    comparison_data.append({
        "Model": "Dense NN (MLP 256→128→64)",
        "Features": "Mean-Pooled Word2Vec (150-d)",
        "Accuracy": f"{dense_val.get('accuracy', 0):.4f}",
        "Macro Precision": f"{dense_val.get('macro_precision', 0):.4f}",
        "Macro Recall": f"{dense_val.get('macro_recall', 0):.4f}",
        "Macro F1": f"{dense_val.get('macro_f1', 0):.4f}",
        "Weighted F1": f"{dense_val.get('weighted_f1', 0):.4f}",
    })

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.caption("All models evaluated on the same stratified validation set (373 resumes, 24 classes). "
               "Macro-F1 is the primary ranking metric as it gives equal weight to each category.")

    # ── BiLSTM vs SVM on Test set ──
    st.write("")
    st.markdown("#### 🧪 Test Set Performance (Held-Out, Unseen Data)")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**BiLSTM (Deep Learning)**")
        st.metric("Test Accuracy", f"{bilstm_test.get('accuracy', 0):.1%}")
        st.metric("Test Macro-F1", f"{bilstm_test.get('macro_f1', 0):.4f}")
        st.metric("Test Weighted-F1", f"{bilstm_test.get('weighted_f1', 0):.4f}")
    with t2:
        st.markdown("**Dense NN (MLP Baseline)**")
        st.metric("Test Accuracy", f"{dense_test.get('accuracy', 0):.1%}")
        st.metric("Test Macro-F1", f"{dense_test.get('macro_f1', 0):.4f}")
        st.metric("Test Weighted-F1", f"{dense_test.get('weighted_f1', 0):.4f}")

    # ── Per-Class Metrics ──
    st.write("")
    st.markdown("#### 🏷️ Per-Class Performance Breakdown (Best Classical Model — Linear SVM)")
    per_class_df = load_per_class_metrics()
    if not per_class_df.empty:
        # Color the F1-Score column
        st.dataframe(
            per_class_df.style.background_gradient(
                subset=["F1-Score"], cmap="RdYlGn", vmin=0.2, vmax=1.0
            ).format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        # Highlight best and worst
        best_cat = per_class_df.loc[per_class_df["F1-Score"].idxmax()]
        worst_cat = per_class_df.loc[per_class_df["F1-Score"].idxmin()]
        b1, b2 = st.columns(2)
        b1.success(f"🟢 **Best Category:** {best_cat['Category']} (F1: {best_cat['F1-Score']:.4f})")
        b2.error(f"🔴 **Hardest Category:** {worst_cat['Category']} (F1: {worst_cat['F1-Score']:.4f})")
    else:
        st.info("Per-class metrics not available.")

    # ── Model Selection Justification ──
    st.write("")
    st.markdown("#### 💡 Model Selection Rationale")
    st.markdown(
        "- **BiLSTM** is selected as the **overall best model** with the highest validation accuracy "
        f"({bilstm_val.get('accuracy', 0):.1%}) and Macro-F1 ({bilstm_val.get('macro_f1', 0):.4f}).\n"
        "- **Linear SVM** is the **best classical ML model** (68.82% Acc, 0.6518 F1), offering faster "
        "inference (~3.6ms vs ~12ms) with competitive performance.\n"
        "- **Dense NN (MLP)** underperforms because mean-pooling Word2Vec embeddings loses sequential "
        "information that the BiLSTM can capture through its bidirectional architecture.\n"
        "- **Macro-F1** is the primary ranking metric, giving equal weight to each of the 24 categories "
        "regardless of class imbalance (BPO: 22 samples vs BUSINESS-DEVELOPMENT: 120 samples)."
    )


# ════════════════════════════════════════════════════════════════
# TAB 3: EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis (EDA)</div>',
                unsafe_allow_html=True)
    st.caption("Comprehensive data understanding performed before any modeling — generated by Person 1 (Poojan).")

    # ── Dataset Overview ──
    st.markdown("#### 📋 Dataset Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Total Resumes", "2,484")
    ov2.metric("Categories", "24")
    ov3.metric("Imbalance Ratio", "5.45×")
    ov4.metric("PDF Archive", "2,500 files")

    st.write("")

    # ── EDA Figures in organized sections ──
    eda_sections = {
        "📊 Class Distribution": [
            ("p1_4_class_distribution.png", "Resume count by category — shows imbalance between classes"),
            ("p1_4_class_pie_chart.png", "Proportional view of resume distribution across 24 categories"),
        ],
        "📏 Resume Length Analysis": [
            ("p1_5_length_distributions.png", "Character and word count distributions across all resumes"),
            ("p1_5_wordcount_by_category.png", "Box plot of word counts per category — shows length variance"),
        ],
        "📝 Word Frequency & N-grams": [
            ("p1_6_top30_words.png", "Top 30 most frequent words in the entire corpus"),
            ("p1_6_top30_words_no_stopwords.png", "Top 30 words after stopword removal — domain keywords emerge"),
            ("p1_8_ngrams_bigrams_trigrams.png", "Most common bigrams and trigrams revealing multi-word patterns"),
        ],
        "☁️ Word Clouds": [
            ("p1_7_wordcloud_overall.png", "Overall corpus word cloud highlighting dominant vocabulary"),
            ("p1_7_wordclouds_top6_categories.png", "Word clouds for the 6 largest categories"),
            ("p1_7_wordclouds_bottom6_categories.png", "Word clouds for the 6 smallest (minority) categories"),
        ],
        "🔬 Class-Wise Vocabulary Analysis": [
            ("p1_9_vocab_size_by_category.png", "Unique vocabulary size per category"),
            ("p1_9_mean_wordcount_by_category.png", "Average word count per category — domain verbosity patterns"),
        ],
        "🔑 TF-IDF Feature Importance": [
            ("top_tfidf_terms_per_class.png", "Top weighted TF-IDF terms per category — what the ML model learned"),
        ],
    }

    for section_title, figures in eda_sections.items():
        with st.expander(section_title, expanded=False):
            for fig_name, caption in figures:
                fig_path = os.path.join(FIGURES_DIR, fig_name)
                if os.path.exists(fig_path):
                    st.image(fig_path, caption=caption, use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fig_name}")
                st.write("")


# ════════════════════════════════════════════════════════════════
# TAB 4: ERROR ANALYSIS & CONFUSION MATRICES
# ════════════════════════════════════════════════════════════════
with tab_errors:
    st.markdown('<div class="section-header">🔍 Error Analysis & Confusion Matrices</div>',
                unsafe_allow_html=True)

    # ── Error Diagnosis Summary ──
    err = load_error_diagnosis()
    total_errors = err.get("total_errors", 116)
    total_val = 372  # from the splits

    st.markdown("#### ⚠️ Error Diagnosis Summary (Best Classical Model — Linear SVM)")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Errors", f"{total_errors}/{total_val}", delta=f"{total_errors/total_val:.1%} error rate")
    e2.metric("Class Overlap", f"{err.get('class_overlap', 17)}", help="Misclassified due to similar categories")
    e3.metric("Generic Resumes", f"{err.get('generic_resumes', 81)}", help="Low-signal, generic text")
    e4.metric("Short Resumes", f"{err.get('short_resumes', 0)}", help="Too short for reliable classification")

    st.write("")
    st.markdown("#### 🧩 Error Category Breakdown")
    st.markdown(
        f"- **Generic / Low-Signal Text ({err.get('generic_resumes', 81)} errors):** "
        "Resumes with generic business language and no strong domain keywords. "
        "The model cannot distinguish these from multiple overlapping categories.\n"
        f"- **Class Overlap ({err.get('class_overlap', 17)} errors):** "
        "Categories with inherent semantic overlap — e.g., FINANCE vs ACCOUNTANT, "
        "DIGITAL-MEDIA vs PUBLIC-RELATIONS, ENGINEERING vs INFORMATION-TECHNOLOGY.\n"
        f"- **Other / Ambiguous ({total_errors - err.get('class_overlap', 17) - err.get('generic_resumes', 81)} errors):** "
        "Multi-domain resumes, noisy OCR text, or genuinely ambiguous career profiles."
    )

    st.divider()

    # ── Confusion Matrices ──
    st.markdown("#### 📊 Confusion Matrix Heatmaps")
    st.caption("Confusion matrices for all trained classical ML models — darker diagonal = better classification.")

    cm_models = {
        "Linear SVM (Best Classical)": "confusion_matrix_Linear_SVM.png",
        "Linear SVM (Tuned)": "confusion_matrix_Linear_SVM_Tuned.png",
        "Logistic Regression": "confusion_matrix_Logistic_Regression.png",
        "Logistic Regression (Tuned)": "confusion_matrix_Logistic_Regression_Tuned.png",
        "Multinomial Naive Bayes": "confusion_matrix_Multinomial_NB.png",
    }

    selected_cm = st.selectbox("Select model confusion matrix:", list(cm_models.keys()))
    cm_path = os.path.join(FIGURES_DIR, cm_models[selected_cm])
    if os.path.exists(cm_path):
        st.image(cm_path, caption=f"Confusion Matrix — {selected_cm}", use_container_width=True)
    else:
        st.warning(f"Confusion matrix image not found for {selected_cm}")

    st.divider()

    # ── Key Confusion Pairs ──
    st.markdown("#### 🔄 Top Confused Category Pairs")
    confusion_pairs = pd.DataFrame([
        {"Actual": "FINANCE", "Predicted": "ACCOUNTANT", "Count": 5,
         "Explanation": "Both involve financial skills, accounting terminology"},
        {"Actual": "CONSULTANT", "Predicted": "ACCOUNTANT", "Count": 4,
         "Explanation": "Management consulting resumes often mention financial analysis"},
        {"Actual": "SALES", "Predicted": "BUSINESS-DEV", "Count": 4,
         "Explanation": "Sales and business development share customer/revenue language"},
        {"Actual": "DIGITAL-MEDIA", "Predicted": "PUBLIC-RELATIONS", "Count": 3,
         "Explanation": "Both involve media, communications, and marketing content"},
        {"Actual": "ENGINEERING", "Predicted": "INFO-TECH", "Count": 3,
         "Explanation": "Software engineers often categorized under IT"},
        {"Actual": "ARTS", "Predicted": "TEACHER", "Count": 3,
         "Explanation": "Art educators frequently appear in both categories"},
    ])
    st.dataframe(confusion_pairs, use_container_width=True, hide_index=True)

    st.markdown(
        "**Key Insight:** Most errors stem from genuine semantic overlap between related professions, "
        "not from model failure. This reflects real-world ambiguity in resume categorization."
    )

# ── Footer ─────────────────────────────────────────────────────
st.divider()
fc1, fc2, fc3 = st.columns([1, 2, 1])
with fc2:
    st.caption(
        "**NueralX Resume Classification System** · SAMATRIX RESUMEFORGE 2026 Hackathon  \n"
        "Team: Poojan (Data & EDA) · Jeel (Classical ML) · Savan (Deep Learning & Demo)  \n"
        "Built with Python · PyTorch · scikit-learn · Streamlit"
    )
