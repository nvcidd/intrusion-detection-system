import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from collections import Counter

# ===============================
# LIMIT CPU (EC2 FIX)
# ===============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="AI IDS", layout="wide")

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    from xgboost import XGBClassifier

    xgb_model = XGBClassifier()
    xgb_model.load_model("xgb_model.json")

    iso_model = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("features.pkl")

    return xgb_model, iso_model, scaler, le, feature_columns


xgb_model, iso_model, scaler, le, feature_columns = load_models()

# ===============================
# UI
# ===============================
st.title("AI Intrusion Detection System")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

live_mode = st.sidebar.toggle("Live Simulation", False)
max_rows = st.sidebar.slider("Max Rows", 1000, 20000, 10000)

# ===============================
# MAIN
# ===============================
if uploaded_file:

    start_time = time.time()

    # LOAD DATA
    df = pd.read_csv(uploaded_file, low_memory=False, encoding="latin1")
    df.columns = df.columns.str.strip()

    original_rows = len(df)

    # REMOVE LABEL
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # CLEAN
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # ALIGN FEATURES
    df = df.reindex(columns=feature_columns, fill_value=0)

    # LIVE MODE SAMPLE
    if live_mode and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    processed_rows = len(df)

    # ===============================
    # 🔥 SINGLE PASS (NO BATCH BUG)
    # ===============================
    X_scaled = scaler.transform(df)

    xgb_proba = xgb_model.predict_proba(X_scaled)
    if_preds = iso_model.predict(X_scaled)

    max_idx = np.argmax(xgb_proba, axis=1)
    max_prob = np.max(xgb_proba, axis=1)

    # FIX LOW CONF BENIGN
    forced_attack = (max_idx == 0) & (max_prob < 0.4)
    final_idx = np.where(forced_attack, 1, max_idx)

    # SAFE INDEX
    final_idx = np.clip(final_idx, 0, len(le.classes_) - 1)

    labels = le.inverse_transform(final_idx)
    anomalies = (if_preds == -1)

    # ===============================
    # COUNT (CORRECT)
    # ===============================
    total_records = len(df)
    total_attacks = 0
    total_anomalies = int(np.sum(anomalies))
    attack_counter = Counter()
    confidences = max_prob.tolist()

    for label in labels:
        if label != "BENIGN":
            total_attacks += 1
            attack_counter[label] += 1

    end_time = time.time()

    # ===============================
    # DISPLAY
    # ===============================
    st.markdown(f"**Original rows:** `{original_rows}`")
    st.markdown(f"**Processed rows:** `{processed_rows}`")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Traffic", total_records)
    col2.metric("Attacks", total_attacks)
    col3.metric("Anomalies", total_anomalies)
    col4.metric("Avg Confidence", f"{np.mean(confidences):.2f}")

    st.markdown(f"**Processing Time:** `{round(end_time - start_time, 2)} sec`")

    # ===============================
    # CHARTS
    # ===============================
    if attack_counter:
        st.subheader("Attack Distribution")
        chart_df = pd.DataFrame({
            "Attack": list(attack_counter.keys()),
            "Count": list(attack_counter.values())
        }).set_index("Attack")

        st.bar_chart(chart_df)

    benign = total_records - total_attacks

    st.subheader("Traffic Split")
    split_df = pd.DataFrame({
        "Type": ["Benign", "Attack"],
        "Count": [benign, total_attacks]
    }).set_index("Type")

    st.bar_chart(split_df)

else:
    st.info("Upload a CSV file to start")