"""Optional Streamlit demo.  Run:  streamlit run app.py

EDUCATIONAL DEMO ONLY. Not a medical device and not a clinically validated tool.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Breast Cancer Classifier (demo)", layout="wide")
st.title("Breast Cancer Wisconsin classifier (educational demo)")
st.warning("Not a diagnostic tool. Trained on a small public dataset with no external validation.")

model_path = ROOT / "models" / "model.joblib"
if not model_path.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

model = joblib.load(model_path)
meta = json.load(open(ROOT / "models" / "metadata.json"))
X = load_breast_cancer(as_frame=True).data
medians = X.median()

mode = st.radio("Input", ["Pick a dataset sample", "Adjust key features manually"], horizontal=True)
if mode == "Pick a dataset sample":
    idx = st.number_input("Sample index", 0, len(X) - 1, 0)
    row = X.iloc[[int(idx)]]
else:
    st.caption("Features not shown are set to the dataset median.")
    key = ["worst texture", "worst symmetry", "worst concavity", "worst concave points",
           "mean concave points", "worst radius"]
    vals = medians.copy()
    cols = st.columns(3)
    for i, f in enumerate(key):
        vals[f] = cols[i % 3].slider(f, float(X[f].min()), float(X[f].max()), float(medians[f]))
    row = pd.DataFrame([vals])

p = float(model.predict_proba(row[X.columns])[:, 1][0])
thr = meta["threshold"]
c1, c2 = st.columns(2)
c1.metric("Estimated malignant probability", f"{p:.1%}")
c2.metric(f"Flag at tuned threshold ({thr:.2f})", "Flagged" if p >= thr else "Not flagged")
st.progress(min(max(p, 0.0), 1.0))
st.caption(f"Model: {meta['model']}. The threshold favours recall, so it errs towards flagging.")
