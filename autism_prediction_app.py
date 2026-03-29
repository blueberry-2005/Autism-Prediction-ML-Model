import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# ─────────────────────────── PAGE CONFIG ────────────────────────────
st.set_page_config(
    page_title="AutiScan · Autism Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── GLOBAL CSS ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0a0f1e;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    #1e2d45;
    --accent:    #3b82f6;
    --accent2:   #06b6d4;
    --green:     #10b981;
    --red:       #ef4444;
    --yellow:    #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font:      'Sora', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}

/* ── Base reset ── */
html, body, [class*="css"] { font-family: var(--font); }
.stApp { background: var(--bg); color: var(--text); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hide default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem; max-width: 1200px; margin: 0 auto; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 60px 48px;
    margin: 28px 0 36px 0;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(6,182,212,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    padding: 6px 16px; border-radius: 50px;
    font-size: 0.75rem; font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 20px;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700; line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 14px;
}
.hero-sub {
    font-size: 1.1rem; color: #94a3b8; line-height: 1.7;
    max-width: 560px; margin-bottom: 32px;
}
.hero-stat {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 18px; margin-right: 12px;
    font-size: 0.85rem;
}
.hero-stat strong { color: var(--accent2); font-family: var(--mono); }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 36px 0 20px 0;
}
.section-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.section-title { font-size: 1.2rem; font-weight: 600; color: var(--text); }
.section-sub { font-size: 0.82rem; color: var(--muted); margin-top: 2px; }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 24px 28px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
}
label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* ── Toggle (radio as buttons) ── */
.stRadio > div { flex-direction: row !important; gap: 8px; }
.stRadio > div > label {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 18px;
    cursor: pointer; font-size: 0.85rem !important;
    transition: all 0.2s;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: white !important;
    border: none !important; border-radius: 10px !important;
    padding: 14px 40px !important; font-size: 1rem !important;
    font-weight: 600 !important; font-family: var(--font) !important;
    letter-spacing: 0.02em; cursor: pointer;
    transition: opacity 0.2s, transform 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── Result cards ── */
.result-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(239,68,68,0.05) 100%);
    border: 1px solid rgba(239,68,68,0.4); border-radius: 16px;
    padding: 32px 36px; text-align: center;
}
.result-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(16,185,129,0.4); border-radius: 16px;
    padding: 32px 36px; text-align: center;
}
.result-icon { font-size: 3.5rem; margin-bottom: 12px; }
.result-label { font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; }
.result-high .result-label { color: #f87171; }
.result-low  .result-label { color: #34d399; }
.result-note { font-size: 0.9rem; color: #94a3b8; line-height: 1.6; }

/* ── Model info cards ── */
.model-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 24px;
    height: 100%;
}
.model-card h4 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }
.model-card p  { font-size: 1.5rem; font-weight: 700; color: var(--text); margin: 0; }
.model-card span { font-size: 0.8rem; color: var(--muted); }

/* ── Score input grid label ── */
.aq-label {
    font-size: 0.78rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 4px; font-weight: 500;
}

/* ── Footer ── */
.footer {
    margin-top: 60px; padding: 32px 0 16px;
    border-top: 1px solid var(--border);
    text-align: center; color: var(--muted);
    font-size: 0.82rem; line-height: 2;
}
.footer strong { color: var(--text); }
.tech-badge {
    display: inline-block;
    background: rgba(255,255,255,0.05); border: 1px solid var(--border);
    border-radius: 6px; padding: 3px 10px; margin: 2px;
    font-family: var(--mono); font-size: 0.75rem; color: #93c5fd;
}

/* ── Divider ── */
.divider { height: 1px; background: var(--border); margin: 28px 0; }

/* ── Sidebar nav ── */
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 8px; margin-bottom: 4px;
    font-size: 0.88rem; cursor: pointer; transition: background 0.2s;
    color: #94a3b8;
}
.nav-item:hover { background: rgba(255,255,255,0.06); color: white; }
.nav-item.active { background: rgba(59,130,246,0.15); color: #93c5fd; }

/* ── Tooltip icon ── */
.tip { font-size: 0.7rem; color: var(--muted); margin-left: 4px; cursor: help; }

/* ── Spinner override ── */
.stSpinner > div { border-color: var(--accent) transparent transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── LOAD MODEL ─────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()


# ─────────────────────────── SIDEBAR ────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 10px 0;">
        <div style="font-size:1.4rem; font-weight:700; color:#e2e8f0; margin-bottom:4px;">🧠 AutiScan</div>
        <div style="font-size:0.75rem; color:#64748b; font-family:'JetBrains Mono',monospace;">v1.0 · ML Prediction System</div>
    </div>
    <div style="height:1px; background:#1e2d45; margin:12px 0 20px 0;"></div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    page = st.radio("", ["🏠  Overview", "🔬  Run Prediction", "📊  Model Info"], label_visibility="collapsed")

    st.markdown("""
    <div style="height:1px; background:#1e2d45; margin:20px 0;"></div>
    <div style="font-size:0.78rem; color:#64748b; line-height:1.8;">
        <div style="margin-bottom:8px; color:#94a3b8; font-weight:500;">⚠️ Disclaimer</div>
        This tool is for <em>educational screening</em> only. 
        Always consult a licensed healthcare professional for diagnosis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute; bottom:28px; left:0; right:0; padding:0 20px;">
        <div style="font-size:0.75rem; color:#374151; text-align:center;">
            Built with ❤️ · Python · Scikit-learn
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW / HERO
# ═══════════════════════════════════════════════════════════════════
if "Overview" in page:

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🧬 &nbsp; AI-powered Screening Tool</div>
        <div class="hero-title">Autism Prediction<br>System</div>
        <div class="hero-sub">
            Leveraging machine learning to provide early autism spectrum disorder 
            screening signals based on behavioural and demographic features.
        </div>
        <div style="margin-bottom:0;">
            <div class="hero-stat">🎯 &nbsp; Accuracy <strong>~81%</strong></div>
            <div class="hero-stat">🌳 &nbsp; Algorithm <strong>Random Forest</strong></div>
            <div class="hero-stat">📋 &nbsp; Features <strong>21</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">⚡</div>
        <div>
            <div class="section-title">How It Works</div>
            <div class="section-sub">Three-step screening pipeline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    steps = [
        ("01", "Input Data", "📝", "Provide demographic info, medical history, and answer the AQ-10 screening questionnaire."),
        ("02", "ML Inference", "🤖", "A trained Random Forest model processes your inputs through a preprocessing pipeline."),
        ("03", "Screening Result", "📊", "Receive an instant risk signal — High or Low — with context to guide next steps."),
    ]
    for col, (num, title, icon, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
            <div class="card">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#3b82f6; margin-bottom:10px;">{num}</div>
                <div style="font-size:1.6rem; margin-bottom:10px;">{icon}</div>
                <div style="font-weight:600; margin-bottom:8px;">{title}</div>
                <div style="font-size:0.83rem; color:#64748b; line-height:1.65;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("→  Start Prediction Now"):
        st.info("Navigate to **🔬 Run Prediction** in the sidebar to begin.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICTION FORM
# ═══════════════════════════════════════════════════════════════════
elif "Prediction" in page:

    st.markdown("""
    <div style="margin:28px 0 6px 0;">
        <div style="font-size:1.6rem; font-weight:700; color:#e2e8f0;">Run a Prediction</div>
        <div style="font-size:0.88rem; color:#64748b; margin-top:4px;">Fill in all sections below and click <em>Analyse</em></div>
    </div>
    """, unsafe_allow_html=True)

    ETHNICITY_OPTIONS = ['?', 'White-European', 'Middle Eastern', 'Pasifika', 'Black',
                         'Others', 'Hispanic', 'Asian', 'Turkish', 'South Asian', 'Latino']
    COUNTRIES = ['Afghanistan','AmericanSamoa','Angola','Argentina','Armenia','Aruba',
                 'Australia','Austria','Azerbaijan','Bahamas','Bangladesh','Belgium',
                 'Bolivia','Brazil','Burundi','Canada','China','Cyprus','Czech Republic',
                 'Egypt','Ethiopia','France','Germany','Hong Kong','Iceland','India',
                 'Iran','Iraq','Ireland','Italy','Japan','Jordan','Kazakhstan','Malaysia',
                 'Mexico','Netherlands','New Zealand','Nicaragua','Niger','Oman','Pakistan',
                 'Romania','Russia','Saudi Arabia','Serbia','Sierra Leone','South Africa',
                 'Spain','Sri Lanka','Sweden','Tonga','Ukraine','United Arab Emirates',
                 'United Kingdom','United States','Viet Nam']
    RELATIONS = ['Self', 'Parent', 'Relative', 'Health care professional', 'Others', '?']

    # ── Section A: Demographics ──
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">👤</div>
        <div>
            <div class="section-title">Demographics</div>
            <div class="section-sub">Basic personal & geographic info</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=100, value=25, help="Age of the individual being screened")
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
    with col3:
        ethnicity = st.selectbox("Ethnicity", ETHNICITY_OPTIONS)

    col4, col5, col6 = st.columns(3)
    with col4:
        country = st.selectbox("Country of Residence", COUNTRIES, index=COUNTRIES.index("India"))
    with col5:
        relation = st.selectbox("Relation to Subject", RELATIONS, help="Who is completing this form?")
    with col6:
        used_app = st.selectbox("Used Screening App Before?", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section B: Medical History ──
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🏥</div>
        <div>
            <div class="section-title">Medical History</div>
            <div class="section-sub">Prior medical conditions relevant to ASD screening</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        jaundice = st.selectbox("Born with Jaundice?", ["No", "Yes"],
                                help="Neonatal jaundice has mild correlation with ASD risk")
    with col8:
        autism_history = st.selectbox("Family Autism History?", ["No", "Yes"],
                                      help="Immediate family member diagnosed with ASD")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section C: AQ-10 Questionnaire ──
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📋</div>
        <div>
            <div class="section-title">AQ-10 Questionnaire</div>
            <div class="section-sub">Autism Quotient screening — answer 0 (No) or 1 (Yes) for each item</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    AQ_QUESTIONS = [
        "Often notices small sounds others don't notice",
        "Usually concentrates on the whole picture rather than small details",
        "Finds it easy to do more than one thing at once",
        "Goes back and forth between activities without difficulty",
        "Doesn't know how to keep a conversation going",
        "Finds it hard to make new friends",
        "Tends to notice patterns in things all the time",
        "Often gets so absorbed in one thing that loses sight of other things",
        "Finds social situations easy",
        "Finds it difficult to work out people's intentions",
    ]

    scores = []
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for i in range(0, 10, 2):
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(f'<div class="aq-label">A{i+1} Score</div>', unsafe_allow_html=True)
            st.caption(AQ_QUESTIONS[i])
            val = st.radio(f"a{i+1}", ["0 · No", "1 · Yes"], key=f"aq{i+1}", horizontal=True, label_visibility="collapsed")
            scores.append(int(val[0]))
        with c_b:
            st.markdown(f'<div class="aq-label">A{i+2} Score</div>', unsafe_allow_html=True)
            st.caption(AQ_QUESTIONS[i+1])
            val = st.radio(f"a{i+2}", ["0 · No", "1 · Yes"], key=f"aq{i+2}", horizontal=True, label_visibility="collapsed")
            scores.append(int(val[0]))
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Result Score ──
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔢</div>
        <div>
            <div class="section-title">Composite Score</div>
            <div class="section-sub">Total AQ score (auto-computed or override)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    auto_result = sum(scores)
    result_score = st.number_input(
        f"Result Score  (auto-calculated: {auto_result})",
        min_value=0.0, max_value=20.0,
        value=float(auto_result), step=0.1,
        help="Sum of AQ-10 scores; can be overridden with a clinical score"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ──
    st.markdown("<div style='margin:28px 0 8px 0;'>", unsafe_allow_html=True)
    predict_btn = st.button("🔬  Analyse · Predict Autism Risk")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Prediction Logic ──
    if predict_btn:
        if model is None:
            st.error("⚠️ Model file (best_model.pkl) not found. Please ensure it is in the same directory.")
        else:
            with st.spinner("Running inference …"):
                time.sleep(1.2)  # UX pause

            # Build dataframe matching training schema
            input_data = pd.DataFrame([{
                "A1_Score": scores[0], "A2_Score": scores[1],
                "A3_Score": scores[2], "A4_Score": scores[3],
                "A4_Score": scores[3], "A5_Score": scores[4],
                "A6_Score": scores[5], "A7_Score": scores[6],
                "A8_Score": scores[7], "A9_Score": scores[8],
                "A10_Score": scores[9],
                "age": float(age),
                "gender": "m" if gender == "Male" else "f",
                "ethnicity": ethnicity,
                "jaundice": jaundice.lower(),
                "austim": autism_history.lower(),
                "contry_of_res": country,
                "used_app_before": used_app.lower(),
                "result": result_score,
                "relation": relation,
            }])

            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                confidence = round(max(proba) * 100, 1)

                st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-high">
                        <div class="result-icon">⚠️</div>
                        <div class="result-label">High Autism Risk Detected</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.95rem; color:#f87171; margin-bottom:14px;">
                            Model confidence: {confidence}%
                        </div>
                        <div class="result-note">
                            The screening model indicates a <strong>higher probability of ASD traits</strong> based on 
                            the provided inputs. This is <em>not</em> a clinical diagnosis.<br>
                            Please consult a licensed psychologist or neurologist for a formal evaluation.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-low">
                        <div class="result-icon">✅</div>
                        <div class="result-label">Low Autism Risk</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.95rem; color:#34d399; margin-bottom:14px;">
                            Model confidence: {confidence}%
                        </div>
                        <div class="result-note">
                            The model does <strong>not detect significant ASD indicators</strong> from the provided data.
                            Results may vary. If you have concerns, a professional consultation is always recommended.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability bar
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown(f"**Low Risk Probability:** `{round(proba[0]*100,1)}%`")
                    st.progress(float(proba[0]))
                with c_right:
                    st.markdown(f"**High Risk Probability:** `{round(proba[1]*100,1)}%`")
                    st.progress(float(proba[1]))

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure `best_model.pkl` was trained with the same feature columns.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════
elif "Model" in page:

    st.markdown("""
    <div style="margin:28px 0 6px 0;">
        <div style="font-size:1.6rem; font-weight:700; color:#e2e8f0;">Model Information</div>
        <div style="font-size:0.88rem; color:#64748b; margin-top:4px;">Architecture, performance, and training details</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📈</div>
        <div><div class="section-title">Performance Metrics</div></div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Accuracy", "81.9%", "On held-out test set"),
        ("Precision (ASD)", "59%", "Class 1 positive pred."),
        ("Recall (ASD)", "64%", "Class 1 sensitivity"),
        ("F1 Score (ASD)", "0.61", "Harmonic mean"),
    ]
    for col, (title, val, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="model-card">
                <h4>{title}</h4>
                <p>{val}</p>
                <span>{sub}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Confusion matrix visualised
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔲</div>
        <div><div class="section-title">Confusion Matrix</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="max-width:360px;">
        <div class="card-title">Test Set (n=160)</div>
        <table style="width:100%; font-family:'JetBrains Mono',monospace; font-size:0.88rem; border-collapse:collapse;">
            <tr>
                <td style="padding:6px 10px; color:#64748b;"></td>
                <td style="padding:6px 10px; color:#64748b; text-align:center;">Pred: 0</td>
                <td style="padding:6px 10px; color:#64748b; text-align:center;">Pred: 1</td>
            </tr>
            <tr>
                <td style="padding:8px 10px; color:#94a3b8;">Actual: 0</td>
                <td style="padding:8px 10px; text-align:center; background:rgba(16,185,129,0.12); border-radius:6px; color:#34d399; font-weight:600;">108</td>
                <td style="padding:8px 10px; text-align:center; background:rgba(239,68,68,0.08); border-radius:6px; color:#f87171;">16</td>
            </tr>
            <tr>
                <td style="padding:8px 10px; color:#94a3b8;">Actual: 1</td>
                <td style="padding:8px 10px; text-align:center; background:rgba(239,68,68,0.08); border-radius:6px; color:#f87171;">13</td>
                <td style="padding:8px 10px; text-align:center; background:rgba(16,185,129,0.12); border-radius:6px; color:#34d399; font-weight:600;">23</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Architecture
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🏗️</div>
        <div><div class="section-title">Model Architecture</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div class="card">
            <div class="card-title">🌳 &nbsp; Algorithm</div>
            <div style="font-size:0.88rem; color:#94a3b8; line-height:2;">
                <div>• <strong style="color:#e2e8f0;">Random Forest Classifier</strong> (primary)</div>
                <div>• n_estimators: <code style="color:#93c5fd;">200</code></div>
                <div>• max_depth: <code style="color:#93c5fd;">20</code></div>
                <div>• XGBoost (comparison baseline)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚙️ &nbsp; Preprocessing Pipeline</div>
            <div style="font-size:0.88rem; color:#94a3b8; line-height:2;">
                <div>• OneHotEncoder for categorical features</div>
                <div>• ColumnTransformer (sklearn Pipeline)</div>
                <div>• SMOTE for class imbalance</div>
                <div>• Hyperparameter tuning via GridSearchCV</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:0;">
        <div class="card-title">📐 &nbsp; Feature Groups</div>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            <span style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#93c5fd;">A1–A10 AQ Scores</span>
            <span style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#93c5fd;">Age</span>
            <span style="background:rgba(6,182,212,0.12); border:1px solid rgba(6,182,212,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#67e8f9;">Gender</span>
            <span style="background:rgba(6,182,212,0.12); border:1px solid rgba(6,182,212,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#67e8f9;">Ethnicity</span>
            <span style="background:rgba(6,182,212,0.12); border:1px solid rgba(6,182,212,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#67e8f9;">Country</span>
            <span style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#6ee7b7;">Jaundice</span>
            <span style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#6ee7b7;">Family Autism</span>
            <span style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#6ee7b7;">Used App</span>
            <span style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#6ee7b7;">Relation</span>
            <span style="background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.25); border-radius:6px; padding:4px 12px; font-size:0.8rem; color:#fcd34d;">Result Score</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────── FOOTER ─────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="margin-bottom:10px;">
        Built by <strong>Your Name</strong> &nbsp;·&nbsp;
        <a href="https://github.com/" style="color:#3b82f6; text-decoration:none;">GitHub</a> &nbsp;·&nbsp;
        <a href="https://linkedin.com/" style="color:#3b82f6; text-decoration:none;">LinkedIn</a>
    </div>
    <div>
        <span class="tech-badge">Python 3.11</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">FastAPI</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">NumPy</span>
    </div>
    <div style="margin-top:14px; font-size:0.75rem; color:#374151;">
        ⚠️ For educational purposes only · Not a medical device
    </div>
</div>
""", unsafe_allow_html=True)
