# -*- coding: utf-8 -*-
"""
University Admissions Probability Estimator
Trained on 1,000+ real applicant profiles · Streamlit App

Run with:  streamlit run admissions_app.py
"""

import json
import math
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Admissions Estimator",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f13;
    color: #e8e6e0;
}

.main { background-color: #0f0f13; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #f0ece0;
}

.stApp { background-color: #0f0f13; }

/* Header */
.header-block {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
    border-bottom: 1px solid #2a2a35;
    margin-bottom: 2rem;
}
.header-block h1 {
    font-size: 2.8rem;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
    color: #f0ece0;
}
.header-block p {
    font-size: 0.95rem;
    color: #7a7870;
    font-weight: 300;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #c8a96e;
    font-weight: 600;
    margin-top: 1.8rem;
    margin-bottom: 0.6rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1a1a24 0%, #16161f 100%);
    border: 1px solid #2e2e3e;
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.result-prob {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    line-height: 1;
    margin-bottom: 0.3rem;
    font-weight: 400;
}
.result-label {
    font-size: 0.85rem;
    color: #7a7870;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.result-uni {
    font-size: 1.05rem;
    color: #c8a96e;
    font-weight: 500;
}

/* Meter bar */
.meter-wrap {
    background: #1e1e28;
    border-radius: 99px;
    height: 10px;
    margin: 1.2rem auto;
    max-width: 320px;
    overflow: hidden;
}
.meter-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

/* Verdict badge */
.verdict {
    display: inline-block;
    padding: 0.3rem 1.1rem;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 0.4rem;
}
.verdict-reach     { background: #3d1f1f; color: #f08080; border: 1px solid #6b3030; }
.verdict-target    { background: #1f2e3d; color: #80b8f0; border: 1px solid #2e4a6b; }
.verdict-likely    { background: #1f3d2a; color: #80f0a0; border: 1px solid #2e6b40; }
.verdict-safety    { background: #2e2d1f; color: #f0e080; border: 1px solid #6b5e2e; }

/* Breakdown table */
.breakdown-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #1e1e28;
    font-size: 0.88rem;
}
.breakdown-row:last-child { border-bottom: none; }
.breakdown-key { color: #8a8880; }
.breakdown-val { color: #e8e6e0; font-weight: 500; }
.breakdown-gap-pos { color: #7dcf8e; }
.breakdown-gap-neg { color: #cf7d7d; }
.breakdown-gap-neu { color: #8a8880; }

/* Input labels */
label { color: #c0bdb0 !important; font-size: 0.88rem !important; }

/* Streamlit widget overrides */
.stSlider > div > div > div { background-color: #c8a96e !important; }
.stSelectbox > div > div { background-color: #16161f !important; border-color: #2e2e3e !important; }
.stNumberInput > div > div > input {
    background-color: #16161f !important;
    border-color: #2e2e3e !important;
    color: #e8e6e0 !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #c8a96e, #b8923a);
    color: #0f0f13;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    width: 100%;
    letter-spacing: 0.5px;
    margin-top: 1rem;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; }

.stRadio > div { gap: 0.5rem; }
.stRadio label { font-size: 0.88rem !important; }

/* Expander */
.st-expander { background-color: #16161f !important; border-color: #2e2e3e !important; }

/* Footer */
.footer {
    text-align: center;
    color: #3a3a4a;
    font-size: 0.75rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #1e1e28;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & university list ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("admissions_model.joblib")

@st.cache_data
def load_universities():
    with open("university_list.json") as f:
        return json.load(f)

try:
    bundle = load_model()
    prep   = bundle["prep"]
    model  = bundle["model"]
    FEATURES = bundle["features"]
    universities = load_universities()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Could not load model: {e}")
    st.stop()


# ── Helper: predict ───────────────────────────────────────────────────────────
def predict(row_dict):
    df = pd.DataFrame([row_dict])[FEATURES]
    Xt = prep.transform(df)
    prob = model.predict_proba(Xt)[0, 1]
    return float(np.clip(prob, 0.001, 0.999))


def verdict(p):
    if p < 0.15:
        return "Reach", "reach"
    elif p < 0.40:
        return "Target", "target"
    elif p < 0.70:
        return "Likely", "likely"
    else:
        return "Safety", "safety"


def meter_color(p):
    if p < 0.15: return "#cf7d7d"
    elif p < 0.40: return "#7d9ccf"
    elif p < 0.70: return "#7dcf8e"
    else: return "#cfc97d"


def gap_class(val):
    if val > 0: return "breakdown-gap-pos"
    elif val < 0: return "breakdown-gap-neg"
    return "breakdown-gap-neu"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>🎓 Admissions Estimator</h1>
    <p>Trained on 1,000+ real applicant profiles &nbsp;·&nbsp; 196 universities</p>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">University</div>', unsafe_allow_html=True)
university = st.selectbox("Select university", universities, label_visibility="collapsed")

col1, col2 = st.columns(2)

st.markdown('<div class="section-label">Academic Profile</div>', unsafe_allow_html=True)
ac1, ac2 = st.columns(2)
with ac1:
    sat = st.number_input("SAT Score", min_value=400, max_value=1600, value=1300, step=10)
with ac2:
    gpa = st.number_input("GPA (unweighted)", min_value=0.0, max_value=4.0, value=3.5, step=0.05, format="%.2f")

st.markdown('<div class="section-label">Extracurriculars & Activities</div>', unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns(3)
with ex1:
    ec_score = st.slider("EC Score (1–5)", 1.0, 5.0, 3.0, 0.5)
with ex2:
    sports_years = st.slider("Sport (years)", 0, 4, 0)
with ex3:
    job_years = st.slider("Job (years)", 0, 4, 0)

ex4, ex5, ex6 = st.columns(3)
with ex4:
    internship = st.radio("Internship?", ["No", "Yes"], horizontal=True)
with ex5:
    award = st.radio("Award / Distinction?", ["No", "Yes"], horizontal=True)
with ex6:
    leadership = st.radio("Leadership role?", ["No", "Yes"], horizontal=True)

st.markdown('<div class="section-label">Background</div>', unsafe_allow_html=True)
bg1, bg2, bg3, bg4 = st.columns(4)
with bg1:
    income = st.selectbox("Income Level", [1, 2, 3, 4],
                          format_func=lambda x: {1:"Very Low",2:"Low",3:"Middle",4:"High"}[x])
with bg2:
    first_gen = st.radio("First-gen?", ["No", "Yes"], horizontal=True)
with bg3:
    legacy = st.radio("Legacy?", ["No", "Yes"], horizontal=True)
with bg4:
    pass  # spacer

run = st.button("Estimate My Chances →")

# ── Results ───────────────────────────────────────────────────────────────────
if run:
    # Load university stats for display
    try:
        unis_df = pd.read_csv("/mnt/user-data/uploads/universities_-_Sheet1__1_.csv")
        uni_row = unis_df[unis_df["University"] == university]
        if not uni_row.empty:
            uni_sat  = uni_row["official_median_SAT"].values[0]
            uni_rate = uni_row["official_accept_rate"].values[0]
            uni_rank = uni_row["ranking"].values[0]
        else:
            uni_sat = uni_rate = uni_rank = None
    except:
        uni_sat = uni_rate = uni_rank = None

    row = {
        "sat_gap":              sat - (uni_sat if uni_sat else sat),
        "GPA_unweighted":       gpa,
        "SAT_total":            sat,
        "official_median_SAT":  uni_sat if uni_sat else sat,
        "official_accept_rate": uni_rate if uni_rate else 0.5,
        "ranking":              uni_rank if uni_rank else 100,
        "rank_inv":             1.0 / uni_rank if uni_rank else 0.01,
        "selectivity":          1.0 - (uni_rate if uni_rate else 0.5),
        "Job_years":            job_years,
        "Sports_years":         sports_years,
        "Internship_flag":      1 if internship == "Yes" else 0,
        "Award_flag":           1 if award == "Yes" else 0,
        "EC_score":             ec_score,
        "Leadership_flag":      1 if leadership == "Yes" else 0,
        "First_gen":            1 if first_gen == "Yes" else 0,
        "Legacy":               1 if legacy == "Yes" else 0,
        "Income_level":         income,
    }

    prob = predict(row)
    pct  = prob * 100
    vlabel, vclass = verdict(prob)
    color = meter_color(prob)

    # ── Result card ──
    st.markdown(f"""
    <div class="result-card">
        <div class="result-prob" style="color:{color}">{pct:.1f}%</div>
        <div class="result-label">Estimated Admission Probability</div>
        <div class="result-uni">{university}</div>
        <div class="meter-wrap">
            <div class="meter-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div>
        </div>
        <span class="verdict verdict-{vclass}">{vlabel}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Breakdown ──
    with st.expander("📊 Score Breakdown", expanded=True):
        sat_gap = sat - (uni_sat if uni_sat else sat)
        rows = [
            ("SAT vs. university median",
             f"{sat:,} vs {int(uni_sat):,}" if uni_sat else f"{sat:,}",
             f"{sat_gap:+d}", sat_gap),
            ("GPA (unweighted)", f"{gpa:.2f}", None, None),
            ("EC Score", f"{ec_score:.1f} / 5", None, None),
            ("Sport", f"{sports_years} yr{'s' if sports_years!=1 else ''}", None, None),
            ("Job experience", f"{job_years} yr{'s' if job_years!=1 else ''}", None, None),
            ("Internship", internship, None, None),
            ("Award / Distinction", award, None, None),
            ("Leadership role", leadership, None, None),
            ("First-generation", first_gen, None, None),
            ("Legacy", legacy, None, None),
            ("Income level", {1:"Very Low",2:"Low",3:"Middle",4:"High"}[income], None, None),
        ]
        if uni_rate:
            rows.append(("University accept rate", f"{uni_rate*100:.1f}%", None, None))
        if uni_rank:
            rows.append(("University ranking", f"#{int(uni_rank)}", None, None))

        html = '<div style="margin-top:0.5rem">'
        for label, val, gap, gap_val in rows:
            gap_html = ""
            if gap is not None:
                gc = gap_class(gap_val)
                gap_html = f' &nbsp;<span class="{gc}">({gap})</span>'
            html += f"""
            <div class="breakdown-row">
                <span class="breakdown-key">{label}</span>
                <span class="breakdown-val">{val}{gap_html}</span>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # ── Compare multiple schools ──
    st.markdown('<div class="section-label" style="margin-top:2rem">Compare Other Schools</div>', unsafe_allow_html=True)
    st.caption("See how your profile stacks up across several universities at once.")

    compare_unis = st.multiselect(
        "Add universities to compare",
        [u for u in universities if u != university],
        max_selections=6,
        label_visibility="collapsed",
    )

    if compare_unis:
        unis_df_local = pd.read_csv("/mnt/user-data/uploads/universities_-_Sheet1__1_.csv")
        results = []
        all_unis = [university] + compare_unis
        for u in all_unis:
            ur = unis_df_local[unis_df_local["University"] == u]
            if ur.empty:
                continue
            u_sat  = ur["official_median_SAT"].values[0]
            u_rate = ur["official_accept_rate"].values[0]
            u_rank = ur["ranking"].values[0] if not pd.isna(ur["ranking"].values[0]) else 999
            r = {**row,
                 "sat_gap": sat - u_sat,
                 "official_median_SAT": u_sat,
                 "official_accept_rate": u_rate,
                 "ranking": u_rank,
                 "rank_inv": 1.0 / u_rank,
                 "selectivity": 1.0 - u_rate}
            p = predict(r)
            vl, vc = verdict(p)
            results.append({
                "University": u,
                "Probability": p,
                "Verdict": vl,
                "_color": meter_color(p),
                "_class": vc,
            })

        results.sort(key=lambda x: x["Probability"], reverse=True)

        html = '<div style="margin-top:0.5rem">'
        for r in results:
            pct_r = r["Probability"] * 100
            is_current = "★ " if r["University"] == university else ""
            html += f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.6rem 0;border-bottom:1px solid #1e1e28">
                <div style="flex:1;font-size:0.88rem;color:#c0bdb0">{is_current}{r['University']}</div>
                <div style="width:120px;background:#1e1e28;border-radius:99px;height:8px;overflow:hidden">
                    <div style="width:{min(pct_r,100):.1f}%;height:100%;background:{r['_color']};border-radius:99px"></div>
                </div>
                <div style="width:52px;text-align:right;font-weight:600;color:{r['_color']}">{pct_r:.1f}%</div>
                <span class="verdict verdict-{r['_class']}" style="width:58px;text-align:center">{r['Verdict']}</span>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Model trained on 1,026 real applicant profiles · 9,014 application outcomes · 196 universities<br>
    AUC 0.80 · For guidance only — admissions involves many factors beyond this model
</div>
""", unsafe_allow_html=True)
