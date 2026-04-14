import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import time
import os

# ===============================
# 🔥 LIMIT CPU USAGE
# ===============================
os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(page_title="IDS Dashboard", layout="wide")

# ===============================
# LOAD MODELS (CACHE)
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

    try:
        xgb_model.set_params(n_jobs=1)
    except:
        pass

    return xgb_model, iso_model, scaler, le, feature_columns


xgb_model, iso_model, scaler, le, feature_columns = load_models()

# ===============================
# UI DESIGN (UNCHANGED)
# ===============================
st.markdown("""<style>
body { background-color: #0a0f1c; color: #e5e7eb; }
.title { font-size: 38px; font-weight: 600; margin-bottom: 20px; }
.card { padding: 20px; border-radius: 14px; background: linear-gradient(145deg, #0f172a, #111827); border: 1px solid #1f2937; }
.metric-value { font-size: 30px; font-weight: 600; }
.metric-label { font-size: 14px; color: #9ca3af; }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="title">AI Intrusion Detection System</div>', unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
uploaded_files = st.sidebar.file_uploader(
    "Upload Network Logs",
    type=["csv"],
    accept_multiple_files=True
)

live_mode = st.sidebar.toggle("Live Simulation", False)
max_rows = st.sidebar.slider("Max Rows", 1000, 20000, 10000, 1000)
batch_size = st.sidebar.slider("Batch Size", 100, 1000, 200, 100)

# ===============================
# CARD
# ===============================
def card(label, value):
    return f"""
    <div class="card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

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

        # ✅ FIXED CSV LOAD
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

        # ✅ FIXED FEATURE ALIGNMENT
        df_features = df_features.reindex(columns=feature_columns, fill_value=0)

        st.write("Processed rows:", len(df_features))

        if live_mode and len(df_features) > max_rows:
            df_features = df_features.sample(n=max_rows, random_state=42)

        # ===============================
        # 🔥 BATCH INFERENCE (FIXED)
        # ===============================
        for i in range(0, len(df_features), batch_size):

            batch = df_features.iloc[i:i+batch_size]

            batch_scaled = scaler.transform(batch)

            xgb_proba = xgb_model.predict_proba(batch_scaled)
            if_preds = iso_model.predict(batch_scaled)

            max_idx = np.argmax(xgb_proba, axis=1)
            max_prob = np.max(xgb_proba, axis=1)

            forced_attack = (max_idx == 0) & (max_prob < 0.4)
            final_idx = np.where(forced_attack, 1, max_idx)

            final_idx = np.clip(final_idx, 0, len(le.classes_) - 1)

            labels = le.inverse_transform(final_idx)
            anomalies = (if_preds == -1)

            # ✅ FIXED COUNTING (CRITICAL)
            total_records += len(labels)

            confidences.extend(np.clip(max_prob, 0, 0.99))

            for label in labels:
                if label != "BENIGN":
                    total_attacks += 1
                    attack_counter[label] += 1

            total_anomalies += int(np.sum(anomalies))

    end_time = time.time()

    # ===============================
    # METRICS
    # ===============================
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(card("Total Traffic", total_records), unsafe_allow_html=True)
    col2.markdown(card("Attacks", total_attacks), unsafe_allow_html=True)
    col3.markdown(card("Anomalies", total_anomalies), unsafe_allow_html=True)
    col4.markdown(card("Avg Confidence", f"{np.mean(confidences):.2f}"), unsafe_allow_html=True)

    st.markdown(card("Processing Time", f"{end_time - start_time:.2f}s"), unsafe_allow_html=True)

    if attack_counter:
        st.subheader("Attack Distribution")
        st.bar_chart(pd.DataFrame({
            "Attack": list(attack_counter.keys()),
            "Count": list(attack_counter.values())
        }).set_index("Attack"))

    benign = total_records - total_attacks

    st.subheader("Traffic Ratio")
    st.bar_chart(pd.DataFrame({
        "Type": ["Benign", "Attack"],
        "Count": [benign, total_attacks]
    }).set_index("Type"))

    fig, ax = plt.subplots()
    ax.pie([benign, total_attacks], labels=["Benign", "Attack"], autopct="%1.1f%%")
    st.pyplot(fig)

else:
    st.info("Upload dataset to start")