# -*- coding: utf-8 -*-
"""
University Admissions Probability Estimator
Trained at runtime from CSVs — no joblib, no version conflicts.
Run with:  streamlit run app.py
Repo must contain: app.py, requirements.txt, decisions.csv, students.csv, universities.csv
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Admissions Estimator",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0f0f13; color: #e8e6e0; }
.main, .stApp { background-color: #0f0f13; }
h1,h2,h3 { font-family: 'DM Serif Display', serif !important; color: #f0ece0; }
.header-block { text-align:center; padding:2.5rem 0 1rem 0; border-bottom:1px solid #2a2a35; margin-bottom:2rem; }
.header-block h1 { font-size:2.8rem; letter-spacing:-0.5px; margin-bottom:0.3rem; }
.header-block p { font-size:0.95rem; color:#7a7870; font-weight:300; }
.section-label { font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#c8a96e; font-weight:600; margin-top:1.8rem; margin-bottom:0.6rem; }
.result-card { background:linear-gradient(135deg,#1a1a24 0%,#16161f 100%); border:1px solid #2e2e3e; border-radius:16px; padding:2.2rem 2rem; text-align:center; margin:1.5rem 0; box-shadow:0 8px 32px rgba(0,0,0,0.4); }
.result-prob { font-family:'DM Serif Display',serif; font-size:5rem; line-height:1; margin-bottom:0.3rem; font-weight:400; }
.result-label { font-size:0.85rem; color:#7a7870; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:1rem; }
.result-uni { font-size:1.05rem; color:#c8a96e; font-weight:500; }
.meter-wrap { background:#1e1e28; border-radius:99px; height:10px; margin:1.2rem auto; max-width:320px; overflow:hidden; }
.meter-fill { height:100%; border-radius:99px; }
.verdict { display:inline-block; padding:0.3rem 1.1rem; border-radius:99px; font-size:0.82rem; font-weight:600; letter-spacing:0.5px; margin-top:0.4rem; }
.verdict-reach  { background:#3d1f1f; color:#f08080; border:1px solid #6b3030; }
.verdict-target { background:#1f2e3d; color:#80b8f0; border:1px solid #2e4a6b; }
.verdict-likely { background:#1f3d2a; color:#80f0a0; border:1px solid #2e6b40; }
.verdict-safety { background:#2e2d1f; color:#f0e080; border:1px solid #6b5e2e; }
.breakdown-row { display:flex; justify-content:space-between; align-items:center; padding:0.45rem 0; border-bottom:1px solid #1e1e28; font-size:0.88rem; }
.breakdown-row:last-child { border-bottom:none; }
.breakdown-key { color:#8a8880; }
.breakdown-val { color:#e8e6e0; font-weight:500; }
.gap-pos { color:#7dcf8e; }
.gap-neg { color:#cf7d7d; }
label { color:#c0bdb0 !important; font-size:0.88rem !important; }
div[data-testid="stButton"] > button { background:linear-gradient(135deg,#c8a96e,#b8923a); color:#0f0f13; font-weight:700; font-size:1rem; border:none; border-radius:10px; padding:0.75rem 2rem; width:100%; letter-spacing:0.5px; margin-top:1rem; }
div[data-testid="stButton"] > button:hover { opacity:0.88; }
.footer { text-align:center; color:#3a3a4a; font-size:0.75rem; padding:2rem 0 1rem 0; border-top:1px solid #1e1e28; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ── Train model at startup (cached — runs once) ───────────────────────────────
FEATURES = ['sat_gap','GPA_unweighted','SAT_total','official_median_SAT',
            'official_accept_rate','ranking','rank_inv','selectivity',
            'Job_years','Sports_years','Internship_flag','Award_flag',
            'EC_score','Leadership_flag','First_gen','Legacy','Income_level']

NAME_FIXES = {
    "University of Washington - Seattle": "University of Washington",
    "University of Washington - St Louis": "Washington University in St. Louis",
    "Texas Institute of Technology": "Texas Tech University",
}

@st.cache_resource(show_spinner="Training model on your data… (one-time, ~30s)")
def train_model():
    decisions   = pd.read_csv("decisions.csv")
    students    = pd.read_csv("students.csv")
    universities = pd.read_csv("universities.csv")

    def explode_col(df, col, label):
        sub = df[['ID', col]].copy()
        sub[col] = sub[col].fillna('').apply(
            lambda x: [u.strip() for u in x.split(',') if u.strip()])
        sub = sub.explode(col).rename(columns={col: 'University'})
        sub = sub[sub['University'] != '']
        sub['Decision'] = label
        return sub

    df_long = pd.concat([
        explode_col(decisions, 'Accepts', 1),
        explode_col(decisions, 'Rejects', 0),
    ], ignore_index=True)
    df_long['University'] = df_long['University'].replace(NAME_FIXES)

    student_cols = ['ID','GPA_unweighted','SAT_total','Income_level','Job_years',
                    'Sports_years','Internship_flag','Award_flag','EC_score',
                    'Leadership_flag','First_gen','Legacy']
    df = df_long.merge(students[student_cols], on='ID', how='left')
    df = df.merge(universities, on='University', how='left')

    df['sat_gap']     = df['SAT_total'] - df['official_median_SAT']
    df['rank_inv']    = 1.0 / df['ranking'].replace(0, np.nan)
    df['selectivity'] = 1.0 - df['official_accept_rate']

    df = df.dropna(subset=['SAT_total','GPA_unweighted','official_median_SAT','Decision']).copy()
    df['Decision'] = df['Decision'].astype(int)

    X = df[FEATURES].copy()
    y = df['Decision']
    groups = df['ID'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(gss.split(X, y, groups=groups))
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    df_train = df.iloc[train_idx]

    # ── Survivorship bias correction ──────────────────────────────────────────
    # Students who self-report results skew towards those who got in.
    # Harvard shows 23% accept rate in the data vs 4% official.
    # Fix: reweight each university's samples so the weighted accept rate
    # matches the official rate. Rejects get weight=1, accepts get w_a < 1.
    weights = np.ones(len(df_train))
    for uni, grp in df_train.groupby('University'):
        official_rate = grp['official_accept_rate'].iloc[0]
        if pd.isna(official_rate): continue
        data_rate = grp['Decision'].mean()
        if data_rate <= 0 or data_rate >= 1: continue
        official_rate = np.clip(official_rate, 0.01, 0.99)
        n_acc = int(grp['Decision'].sum())
        n_rej = len(grp) - n_acc
        if n_acc == 0 or n_rej == 0: continue
        w_a = (official_rate * n_rej) / ((1 - official_rate) * n_acc)
        w_a = np.clip(w_a, 0.05, 20.0)
        acc_pos = np.where(df_train.index.isin(grp[grp['Decision']==1].index))[0]
        rej_pos = np.where(df_train.index.isin(grp[grp['Decision']==0].index))[0]
        weights[acc_pos] = w_a
        weights[rej_pos] = 1.0

    num_pipe   = Pipeline([("imp", SimpleImputer(strategy="median"))])
    preprocess = ColumnTransformer([("num", num_pipe, FEATURES)],
                                   remainder="drop", sparse_threshold=0.0)
    prep = clone(preprocess)
    prep.fit(X_train)
    Xt_train = prep.transform(X_train)

    feat_names = prep.get_feature_names_out()
    mono = np.zeros(len(feat_names), dtype=int)
    idx = np.where(feat_names == "num__sat_gap")[0]
    if len(idx): mono[idx[0]] = +1

    base = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=5, max_iter=500,
        min_samples_leaf=25, random_state=42, monotonic_cst=mono)
    base.fit(Xt_train, y_train, sample_weight=weights)

    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    cal.fit(Xt_train, y_train, sample_weight=weights)

    uni_list = sorted(universities['University'].dropna().unique().tolist())
    uni_lookup = universities.set_index('University').to_dict('index')

    return prep, cal, uni_list, uni_lookup

@st.cache_data
def load_uni_lookup():
    return pd.read_csv("universities.csv").set_index('University').to_dict('index')

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    prep, model, universities, uni_lookup = train_model()
except Exception as e:
    st.error(f"Failed to train model: {e}")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(row_dict):
    df = pd.DataFrame([row_dict])[FEATURES]
    Xt = prep.transform(df)
    return float(np.clip(model.predict_proba(Xt)[0, 1], 0.001, 0.999))

def get_uni_stats(uni):
    info = uni_lookup.get(uni, {})
    return (info.get('official_median_SAT'), 
            info.get('official_accept_rate'), 
            info.get('ranking'))

def build_row(sat, gpa, ec, income, job, sport, internship, award, leadership, first_gen, legacy, uni):
    u_sat, u_rate, u_rank = get_uni_stats(uni)
    return {
        'sat_gap':              sat - (u_sat or sat),
        'GPA_unweighted':       gpa,
        'SAT_total':            sat,
        'official_median_SAT':  u_sat or sat,
        'official_accept_rate': u_rate or 0.5,
        'ranking':              u_rank or 100,
        'rank_inv':             1.0 / (u_rank or 100),
        'selectivity':          1.0 - (u_rate or 0.5),
        'Job_years':            job,
        'Sports_years':         sport,
        'Internship_flag':      1 if internship == "Yes" else 0,
        'Award_flag':           1 if award == "Yes" else 0,
        'EC_score':             ec,
        'Leadership_flag':      1 if leadership == "Yes" else 0,
        'First_gen':            1 if first_gen == "Yes" else 0,
        'Legacy':               1 if legacy == "Yes" else 0,
        'Income_level':         income,
    }

def verdict(p):
    if p < 0.15:   return "Reach",  "reach"
    elif p < 0.40: return "Target", "target"
    elif p < 0.70: return "Likely", "likely"
    else:          return "Safety", "safety"

def meter_color(p):
    if p < 0.15:   return "#cf7d7d"
    elif p < 0.40: return "#7d9ccf"
    elif p < 0.70: return "#7dcf8e"
    else:          return "#cfc97d"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>🎓 Admissions Estimator</h1>
    <p>Trained on 1,000+ real applicant profiles &nbsp;·&nbsp; 196 universities</p>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">University</div>', unsafe_allow_html=True)
university = st.selectbox("University", universities, label_visibility="collapsed")

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
    award = st.radio("Award?", ["No", "Yes"], horizontal=True)
with ex6:
    leadership = st.radio("Leadership?", ["No", "Yes"], horizontal=True)

st.markdown('<div class="section-label">Background</div>', unsafe_allow_html=True)
bg1, bg2, bg3, bg4 = st.columns(4)
with bg1:
    income = st.selectbox("Income Level", [1,2,3,4],
                          format_func=lambda x: {1:"Very Low",2:"Low",3:"Middle",4:"High"}[x])
with bg2:
    first_gen = st.radio("First-gen?", ["No", "Yes"], horizontal=True)
with bg3:
    legacy = st.radio("Legacy?", ["No", "Yes"], horizontal=True)
with bg4:
    pass

run = st.button("Estimate My Chances →")

# ── Results ───────────────────────────────────────────────────────────────────
if run:
    row  = build_row(sat, gpa, ec_score, income, job_years, sports_years,
                     internship, award, leadership, first_gen, legacy, university)
    prob = predict(row)
    pct  = prob * 100
    vlabel, vclass = verdict(prob)
    color = meter_color(prob)
    u_sat, u_rate, u_rank = get_uni_stats(university)

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

    with st.expander("📊 Score Breakdown", expanded=True):
        int(sat - (u_sat or sat))
        gc = "gap-pos" if sat_gap > 0 else "gap-neg"
        income_labels = {1:"Very Low",2:"Low",3:"Middle",4:"High"}
        rows_html = [
            ("SAT vs median", f"{sat:,} vs {int(u_sat):,}" if u_sat else f"{sat:,}",
             f'<span class="{gc}">({sat_gap:+d})</span>' if u_sat else ""),
            ("GPA (unweighted)", f"{gpa:.2f}", ""),
            ("EC Score", f"{ec_score:.1f} / 5", ""),
            ("Sport", f"{sports_years} yrs", ""),
            ("Job experience", f"{job_years} yrs", ""),
            ("Internship", internship, ""),
            ("Award", award, ""),
            ("Leadership", leadership, ""),
            ("First-generation", first_gen, ""),
            ("Legacy", legacy, ""),
            ("Income level", income_labels[income], ""),
        ]
        if u_rate: rows_html.append(("Accept rate", f"{u_rate*100:.1f}%", ""))
        if u_rank: rows_html.append(("Ranking", f"#{int(u_rank)}", ""))

        html = '<div style="margin-top:0.5rem">'
        for k, v, extra in rows_html:
            html += f'<div class="breakdown-row"><span class="breakdown-key">{k}</span><span class="breakdown-val">{v} {extra}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:2rem">Compare Schools</div>', unsafe_allow_html=True)
    st.caption("Add universities to compare your odds side by side.")
    compare_unis = st.multiselect("Compare", [u for u in universities if u != university],
                                  max_selections=6, label_visibility="collapsed")

    if compare_unis:
        results = []
        for u in [university] + compare_unis:
            u_sat2, u_rate2, u_rank2 = get_uni_stats(u)
            if u_sat2 is None: continue
            r = build_row(sat, gpa, ec_score, income, job_years, sports_years,
                          internship, award, leadership, first_gen, legacy, u)
            p = predict(r)
            vl, vc = verdict(p)
            results.append({"uni": u, "prob": p, "vl": vl, "vc": vc, "col": meter_color(p)})
        results.sort(key=lambda x: x["prob"], reverse=True)

        html = '<div style="margin-top:0.5rem">'
        for r in results:
            star = "★ " if r["uni"] == university else ""
            pt = r["prob"] * 100
            html += f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.6rem 0;border-bottom:1px solid #1e1e28">
                <div style="flex:1;font-size:0.88rem;color:#c0bdb0">{star}{r['uni']}</div>
                <div style="width:120px;background:#1e1e28;border-radius:99px;height:8px;overflow:hidden">
                    <div style="width:{min(pt,100):.1f}%;height:100%;background:{r['col']};border-radius:99px"></div>
                </div>
                <div style="width:52px;text-align:right;font-weight:600;color:{r['col']}">{pt:.1f}%</div>
                <span class="verdict verdict-{r['vc']}" style="width:58px;text-align:center;font-size:0.75rem">{r['vl']}</span>
            </div>"""
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Model trained on 1,026 real applicant profiles · 9,014 outcomes · 196 universities · AUC 0.80<br>
    For guidance only — admissions involves many factors beyond this model
</div>
""", unsafe_allow_html=True)
