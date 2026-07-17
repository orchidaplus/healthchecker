import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Health Checker — Orchid A+", page_icon="🩺", layout="centered")

st.markdown("""
<style>
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.1rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
    p, li { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stroke_risk_model.joblib")
SCORES_PATH = os.path.join(os.path.dirname(__file__), "reference_risk_scores.npy")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_reference_scores():
    return np.load(SCORES_PATH)


model = load_model()
reference_scores = load_reference_scores()

st.title("🩺 Health Checker")
st.caption("A demo risk-screening tool — not a diagnosis.")

with st.expander("What this tool is and isn't", expanded=True):
    st.markdown(
        """
This tool estimates relative stroke risk from five factors (age, systolic blood
pressure, glucose, cigarettes/day, and sex) using a model trained on a public
Framingham-style dataset (~4,200 patients, ~15% positive stroke rate).

**Read the numbers with this in mind:**
- The model's precision at typical operating points is roughly **20-35%** — most
  "elevated risk" flags will *not* correspond to an actual stroke. It is a
  screening nudge, not a prediction.
- It only uses 5 self-reported/measured variables. It does **not** account for
  ECG findings, family history, prior TIA/stroke, atrial fibrillation, or
  medication history — all known to matter clinically.
- **This is not a medical device and does not replace a clinician.** If you or
  someone else has stroke symptoms right now (sudden numbness, confusion,
  trouble speaking or seeing, severe headache, loss of balance), call
  emergency services immediately — don't wait on a risk score.
        """
    )

st.subheader("Enter patient information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50, step=1)
    sysBP = st.number_input("Systolic blood pressure (mmHg)", min_value=80.0, max_value=260.0, value=130.0, step=1.0)
    glucose = st.number_input("Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=85.0, step=1.0)
with col2:
    gender = st.selectbox("Sex", ["Male", "Female"])
    cigsPerDay = st.number_input("Cigarettes per day", min_value=0.0, max_value=80.0, value=0.0, step=1.0)

if st.button("Estimate risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [{
            "age": age,
            "glucose": glucose,
            "sysBP": sysBP,
            "cigsPerDay": cigsPerDay,
            "Gender": gender,
        }]
    )

    risk_prob = model.predict_proba(input_df)[0, 1]
    percentile = (reference_scores < risk_prob).mean() * 100

    st.divider()
    st.subheader("Result")

    if risk_prob >= 0.5:
        tier, color = "Elevated", "🔴"
    elif risk_prob >= 0.3:
        tier, color = "Moderate", "🟠"
    else:
        tier, color = "Lower", "🟢"

    c1, c2 = st.columns(2)
    c1.metric("Estimated risk score", f"{risk_prob:.0%}")
    c2.metric("Relative to dataset", f"Higher than {percentile:.0f}% of patients")

    st.markdown(f"**{color} {tier} relative risk band**")

    st.progress(min(max(risk_prob, 0.0), 1.0))

    st.caption(
        "This score reflects relative risk within this dataset's population, not a "
        "calibrated probability of having a stroke. Because the model's precision is "
        "limited (see note above), treat an elevated score as 'worth discussing with a "
        "doctor,' not as a diagnosis — and treat a low score as reassuring but not conclusive."
    )

    with st.expander("What's driving this estimate"):
        st.markdown(
            f"""
Based on the systematic feature analysis behind this model, the strongest
drivers of stroke risk in this dataset are, in order: **age**, **systolic
blood pressure**, **glucose**, **smoking (cigarettes/day)**, and **sex**.
For this input:
- Age **{age}** {'is above' if age > 55 else 'is below'} the higher-risk range typically seen in this data (55+).
- Systolic BP **{sysBP:.0f} mmHg** {'is elevated (>140 is hypertensive range)' if sysBP > 140 else 'is within a typical range'}.
- Glucose **{glucose:.0f} mg/dL** {'is in the diabetic range (>=126)' if glucose >= 126 else 'is within a typical range'}.
            """
        )

st.divider()
st.caption(
    "Model: logistic regression, class-weighted, trained on 5 features selected via "
    "systematic leave-one-feature-out and univariate PR-AUC search. "
    "For informational/educational purposes only."
)
