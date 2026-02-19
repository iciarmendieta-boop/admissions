import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Admissions Predictor", layout="centered")

def to_dense_func(X):
    return X.toarray() if hasattr(X, "toarray") else X

# =============================
# Load model + expected columns
# =============================
model = joblib.load("admissions_model.joblib")

with open("model_columns.json", "r") as f:
    model_cols = json.load(f)

st.header("Batch prediction (CSV)")

uploaded_file = st.file_uploader(
    "Upload a CSV of student rows (any extra columns will be ignored).",
    type=["csv"]
)

if uploaded_file is not None:
    df_in = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df_in.head())

    # --- Align to training columns ---
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

    # --- Predict ---
    try:
        probs = model.predict_proba(X)[:, 1]
        out = df_in.copy()
        out["predicted_probability"] = probs

        st.subheader("Predictions")
        st.dataframe(out.head())

        # Download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("Prediction failed. This usually means the uploaded CSV doesn't match what the pipeline expects.")
        st.exception(e)


# =============================
# Load university data (SAT/rank/acceptance rate)
# =============================
uni_df = pd.read_csv("universities - Sheet1.csv")

# Try to find the university name column without you renaming anything
uni_name_col = None
for cand in ["university", "University", "school", "School", "Name", "name", "Institution", "institution"]:
    if cand in uni_df.columns:
        uni_name_col = cand
        break

if uni_name_col is None:
    st.error(f"Could not find a university-name column in your CSV. Columns found: {list(uni_df.columns)}")
    st.stop()

# Clean names
uni_df[uni_name_col] = uni_df[uni_name_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

# =============================
# UI
# =============================
st.title("🎓 University Admissions Predictor")

st.header("Student Profile")
SAT_total = st.number_input("SAT_total", 400, 1600, 1300, 10)
GPA_unweighted = st.number_input("GPA_unweighted", 0.0, 4.0, 3.7, 0.01)
EC_score = st.slider("EC_score (1–5)", 1, 5, 3)
Income_level = st.slider("Income_level (1–4)", 1, 4, 3)
# State / International
state_or_international = st.selectbox(
    "State (if US citizen) or International",
    [
        "International",
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

st.header("University")
university = st.selectbox("Select University", sorted(uni_df[uni_name_col].unique()))

# Grab that university’s features (SAT/rank/acceptance, etc.)
uni_row = uni_df[uni_df[uni_name_col] == university].iloc[0].to_dict()

# =============================
# Build prediction row
# =============================
row = {
    "SAT_total": SAT_total,
    "GPA_unweighted": GPA_unweighted,
    "EC_score": EC_score,
    "Income_level": Income_level,
    "gender": gender,
    "major": major,
    "university": university,
    "state_or_international": state_or_international,
}


# Add university numeric features to the row
# (keeps ALL columns from the CSV as features)
row.update(uni_row)

# Put into DataFrame and align columns exactly as model expects
df_input = pd.DataFrame([row]).reindex(columns=model_cols, fill_value=np.nan)

# =============================
# Predict
# =============================
if st.button("Predict Admission Probability"):
    p = float(model.predict_proba(df_input)[0, 1])

    if p < 0.25:
        tier = "High Reach"
    elif p < 0.45:
        tier = "Reach"
    elif p < 0.70:
        tier = "Target"
    else:
        tier = "Safety"

    st.subheader("Result")
    st.metric("Acceptance Probability", f"{p:.1%}")
    st.write("Tier:", tier)

