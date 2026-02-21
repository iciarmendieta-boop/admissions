import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Admissions Predictor", layout="centered")

# =============================
# Load model + expected columns
# =============================
bundle = joblib.load("admissions_model (1).joblib")
prep = bundle["prep"]
model = bundle["model"]

with open("model_columns.json", "r") as f:
    model_cols = json.load(f)

# Infer the university column name expected by the model
UNI_COL = "University" if "University" in model_cols else ("university" if "university" in model_cols else None)
if UNI_COL is None:
    st.error("Your model_columns.json does not include a University/university column. Check training artifacts.")
    st.stop()

# Infer the state column name expected by the model
STATE_COL = "state_or_international" if "state_or_international" in model_cols else (
    "State_or_international" if "State_or_international" in model_cols else None
)

# =============================
# Load university data
# =============================
uni_df = pd.read_csv("universities - Sheet1.csv")

uni_name_col = None
for cand in ["university", "University", "school", "School", "Name", "name", "Institution", "institution"]:
    if cand in uni_df.columns:
        uni_name_col = cand
        break

if uni_name_col is None:
    st.error(f"Could not find a university-name column in your CSV. Columns found: {list(uni_df.columns)}")
    st.stop()

uni_df[uni_name_col] = uni_df[uni_name_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

# =============================
# UI
# =============================
st.title("🎓 University Admissions Predictor")

# -----------------------------
# Batch prediction (CSV)
# -----------------------------
st.header("Batch prediction (CSV)")

uploaded_file = st.file_uploader(
    "Upload a CSV of student rows (any extra columns will be ignored).",
    type=["csv"]
)

if uploaded_file is not None:
    df_in = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df_in.head())

    X = df_in.copy()

    # Drop extra cols not used by the model
    extra = [c for c in X.columns if c not in model_cols]
    if extra:
        st.info(f"Ignoring extra columns not used by the model: {extra[:15]}" + (" ..." if len(extra) > 15 else ""))
        X = X.drop(columns=extra)

    # Add missing cols expected by model
    missing = [c for c in model_cols if c not in X.columns]
    if missing:
        st.warning(f"Missing columns were added as NaN: {missing[:15]}" + (" ..." if len(missing) > 15 else ""))
        for c in missing:
            X[c] = np.nan

    # Reorder columns exactly as training
    X = X[model_cols]

    # Predict (IMPORTANT: preprocess first)
    try:
        Xt = prep.transform(X)
        probs = model.predict_proba(Xt)[:, 1]

        out = df_in.copy()
        out["predicted_probability"] = probs

        st.subheader("Predictions")
        st.dataframe(out.head())

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("Prediction failed. This usually means the uploaded CSV doesn't match what the preprocessor expects.")
        st.exception(e)

st.divider()

# -----------------------------
# Single prediction
# -----------------------------
st.header("Single prediction")

st.subheader("Student Profile")
SAT_total = st.number_input("SAT_total", 400, 1600, 1300, 10)
GPA_unweighted = st.number_input("GPA_unweighted", 0.0, 4.0, 3.7, 0.01)
EC_score = st.slider("EC_score (1–5)", 1, 5, 3)
Income_level = st.slider("Income_level (1–4)", 1, 4, 3)

state_or_international = st.selectbox(
    "State (if US citizen) or International",
    [
        "international",
        "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
        "Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana",
        "Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
        "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina",
        "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina",
        "South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia",
        "Wisconsin","Wyoming"
    ],
    index=0
)

gender = st.selectbox("gender", ["female", "male", "other"])
major = st.text_input("major", "Computer Science")

st.subheader("University")
university = st.selectbox("Select University", sorted(uni_df[uni_name_col].unique()))
uni_row = uni_df[uni_df[uni_name_col] == university].iloc[0].to_dict()

# Build row
row = {
    "SAT_total": SAT_total,
    "GPA_unweighted": GPA_unweighted,
    "EC_score": EC_score,
    "Income_level": Income_level,
    "gender": gender,
    "major": major,
    UNI_COL: university,  # <-- FIXED: match training column name
}

# Optional state column if your model expects it
if STATE_COL is not None:
    row[STATE_COL] = state_or_international

# Add all university features from your CSV (safe; extras will be dropped by reindex)
row.update(uni_row)

df_input = pd.DataFrame([row]).reindex(columns=model_cols, fill_value=np.nan)

with st.expander("Debug: model input row"):
    st.dataframe(df_input)

if st.button("Predict Admission Probability"):
    try:
        Xt = prep.transform(df_input)
        p = float(model.predict_proba(Xt)[0, 1])

        # Optional blending with official accept rate (if present)
        alpha = 0.6
        prior = uni_row.get("official_accept_rate", None)
        if prior is not None and not pd.isna(prior):
            prior = float(prior)
            if prior > 1:
                prior = prior / 100.0
            p = alpha * p + (1 - alpha) * prior

        st.subheader("Result")
        st.metric("Acceptance Probability", f"{p:.1%}")
    except Exception as e:
        st.error("Prediction failed for the single row.")
        st.exception(e)
