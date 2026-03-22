import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Header banner */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1420 100%);
    border: 1px solid #2a3050;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    color: #e8eaf0;
    text-align: center;
}
.hero h1 .highlight {
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    text-shadow: 0 0 30px rgba(129, 140, 248, 0.3);
}
.hero p {
    color: #7c85a2;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
    text-align: center;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.8rem;
    margin-top: 1.6rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #1a1f2e 100%);
    border: 1px solid #3730a3;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    color: #c084fc;
    line-height: 1;
}
.result-range {
    font-size: 0.85rem;
    color: #7c85a2;
    margin-top: 0.6rem;
}

/* Metric pills */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
    justify-content: center;
    flex-wrap: wrap;
}
.metric-pill {
    background: #1a1f2e;
    border: 1px solid #2a3050;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.78rem;
    color: #9ca3af;
}
.metric-pill span {
    color: #818cf8;
    font-weight: 600;
}

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div {
    background: #1a1f2e !important;
    border-color: #2a3050 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}
label {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #3d4468;
    font-size: 0.75rem;
    margin-top: 3rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & columns ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("Salary_prediction.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

try:
    model, feature_columns = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💼 <span class="highlight">Salary Prediction</span></h1>
    <p>AI-powered salary estimation · Random Forest · 250,000 data points · R² = 0.959</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files. Make sure `Salary_prediction.pkl` and `columns.pkl` are in the same folder as this app.\n\nError: {load_error}")
    st.stop()

# ── Dropdown options ──────────────────────────────────────────────────────────
JOB_TITLES = [
    "AI Engineer", "Backend Developer", "Business Analyst", "Cloud Engineer",
    "Cybersecurity Analyst", "Data Analyst", "Data Scientist", "DevOps Engineer",
    "Frontend Developer", "Machine Learning Engineer", "Product Manager", "Software Engineer"
]

EDUCATION = ["Bachelor", "Diploma", "High School", "Master", "PhD"]

INDUSTRIES = [
    "Consulting", "Education", "Finance", "Government", "Healthcare",
    "Manufacturing", "Media", "Retail", "Technology", "Telecom"
]

COMPANY_SIZES = ["Enterprise", "Large", "Medium", "Small", "Startup"]

LOCATIONS = [
    "Australia", "Canada", "Germany", "India", "Netherlands", "Remote",
    "Singapore", "Sweden", "UK", "USA"
]

REMOTE_OPTIONS = ["Yes", "No", "Hybrid"]

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">👤 Your Profile</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    job_title = st.selectbox("Job Title", JOB_TITLES)
    education = st.selectbox("Education Level", EDUCATION)
    experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5, step=1)
with col2:
    industry = st.selectbox("Industry", INDUSTRIES)
    company_size = st.selectbox("Company Size", COMPANY_SIZES)
    location = st.selectbox("Location", LOCATIONS)

st.markdown('<div class="section-label">📋 Job Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    remote_work = st.selectbox("Remote Work", REMOTE_OPTIONS)
    skills_count = st.number_input("Number of Skills", min_value=1, max_value=30, value=8, step=1)
with col4:
    certifications = st.number_input("Number of Certifications", min_value=0, max_value=10, value=1, step=1)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict My Salary"):

    # Build input row matching get_dummies encoding
    input_data = {
        "experience_years": experience,
        "skills_count": skills_count,
        "certifications": certifications,
    }

    # One-hot encode all categorical columns (drop_first=True was used in training)
    all_education = ["Bachelor", "Diploma", "High School", "Master", "PhD"]
    all_company   = ["Enterprise", "Large", "Medium", "Small", "Startup"]
    all_industry  = sorted(INDUSTRIES)
    all_remote    = ["Hybrid", "No", "Yes"]
    all_jobs      = sorted(JOB_TITLES)
    all_locations = sorted(LOCATIONS)

    def ohe(prefix, all_vals, chosen, drop_first=True):
        cats = sorted(all_vals)
        if drop_first:
            cats = cats[1:]          # drop_first removes the first alphabetically
        for c in cats:
            input_data[f"{prefix}_{c}"] = 1 if chosen == c else 0

    ohe("education_level", all_education, education)
    ohe("company_size",    all_company,   company_size)
    ohe("industry",        all_industry,  industry)
    ohe("remote_work",     all_remote,    remote_work)
    ohe("job_title",       all_jobs,      job_title)
    ohe("location",        all_locations, location)

    # Align with training columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    low  = prediction * 0.90
    high = prediction * 1.10

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Annual Salary</div>
        <div class="result-value">{prediction:,.0f}</div>
        <div class="result-range">Likely range: {low:,.0f} – {high:,.0f}</div>
        <div class="metrics-row">
            <div class="metric-pill">Model Accuracy <span>R² = 0.959</span></div>
            <div class="metric-pill">Trees <span>150</span></div>
            <div class="metric-pill">Max Depth <span>25</span></div>
            <div class="metric-pill">Training Data <span>200K rows</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · Random Forest Regressor · scikit-learn
</div>
""", unsafe_allow_html=True)
