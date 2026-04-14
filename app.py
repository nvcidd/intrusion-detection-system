import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import time
import os

# ===============================
# LIMIT CPU
# ===============================
os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(page_title="IDS Dashboard", layout="wide")

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

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    accept_multiple_files=True
)

live_mode = st.sidebar.toggle("Live Simulation", False)
max_rows = st.sidebar.slider("Max Rows", 1000, 20000, 10000, 1000)
batch_size = st.sidebar.slider("Batch Size", 100, 1000, 200, 100)

# ===============================
# MAIN
# ===============================
if uploaded_files:

    total_records = 0
    total_attacks = 0
    total_anomalies = 0

    attack_counter = Counter()
    confidences = []

    start_time = time.time()

    for file in uploaded_files:

        df = pd.read_csv(file, encoding="latin1", low_memory=False, on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        st.write("Original rows:", len(df))

        if "Label" in df.columns:
            df_features = df.drop(columns=["Label"])
        else:
            df_features = df.copy()

        df_features = df_features.apply(pd.to_numeric, errors='coerce')
        df_features = df_features.replace([np.inf, -np.inf], 0)
        df_features = df_features.fillna(0)

        df_features = df_features.reindex(columns=feature_columns, fill_value=0)

        st.write("Processed rows:", len(df_features))

        if live_mode and len(df_features) > max_rows:
            df_features = df_features.sample(n=max_rows, random_state=42)

        # ===============================
        # 🔥 PROCESS ALL DATA AT ONCE (NO LOOP BREAK)
        # ===============================
        X_scaled = scaler.transform(df_features)

        xgb_proba = xgb_model.predict_proba(X_scaled)
        if_preds = iso_model.predict(X_scaled)

        max_idx = np.argmax(xgb_proba, axis=1)
        max_prob = np.max(xgb_proba, axis=1)

        forced_attack = (max_idx == 0) & (max_prob < 0.4)
        final_idx = np.where(forced_attack, 1, max_idx)

        final_idx = np.clip(final_idx, 0, len(le.classes_) - 1)

        labels = le.inverse_transform(final_idx)
        anomalies = (if_preds == -1)

        # ===============================
        # 🔥 CORRECT COUNTING
        # ===============================
        total_records += len(labels)
        confidences.extend(np.clip(max_prob, 0, 0.99))

        for label in labels:
            if label != "BENIGN":
                total_attacks += 1
                attack_counter[label] += 1

        total_anomalies += int(np.sum(anomalies))

    end_time = time.time()

    # ===============================
    # DISPLAY
    # ===============================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Traffic", total_records)
    col2.metric("Attacks", total_attacks)
    col3.metric("Anomalies", total_anomalies)
    col4.metric("Avg Confidence", f"{np.mean(confidences):.2f}")

    st.write("Processing Time:", round(end_time - start_time, 2), "sec")

    if attack_counter:
        st.bar_chart(pd.DataFrame({
            "Attack": list(attack_counter.keys()),
            "Count": list(attack_counter.values())
        }).set_index("Attack"))

    benign = total_records - total_attacks

    st.bar_chart(pd.DataFrame({
        "Type": ["Benign", "Attack"],
        "Count": [benign, total_attacks]
    }).set_index("Type"))

    fig, ax = plt.subplots()
    ax.pie([benign, total_attacks], labels=["Benign", "Attack"], autopct="%1.1f%%")
    st.pyplot(fig)

else:
    st.info("Upload dataset")