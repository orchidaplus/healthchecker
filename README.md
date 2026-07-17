# Stroke Risk Screening — Streamlit App

A minimal, honestly-framed risk-screening demo built on the trimmed 5-feature
model (`age`, `sysBP`, `glucose`, `cigsPerDay`, `Gender`) that matched the
full 15-feature model's performance in the analysis notebook.

## Files
- `app.py` — the Streamlit app
- `stroke_risk_model.joblib` — trained scikit-learn pipeline (preprocessing + logistic regression)
- `reference_risk_scores.npy` — risk scores for the training population, used to show "higher than X% of patients"
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Deploy for free (Streamlit Community Cloud)
1. Push this folder to a GitHub repo (public or private).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app," pick the repo/branch, set the main file to `app.py`.
4. Deploy — it installs `requirements.txt` automatically and gives you a public URL.

## Retraining on new/more data
If you get a richer dataset (see the "do I need different data" notes below),
retrain by re-running the training script logic in `app.py`'s companion
training step: load your CSV, fit the same `ColumnTransformer` + `LogisticRegression`
pipeline on your chosen features, then `joblib.dump` it to `stroke_risk_model.joblib`
and regenerate `reference_risk_scores.npy` from `predict_proba` on the training set.

## Known limitations (also shown in-app)
- Precision at typical thresholds is ~20-35% — most "elevated risk" flags won't
  correspond to an actual stroke.
- Only 5 variables are used; no ECG, family history, or prior-event data.
- Not a medical device. Educational/demo use only.
