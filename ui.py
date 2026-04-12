import streamlit as st
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import time
import gc

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# ===============================
# CLEAN DARK UI (NO HEAVY ANIMATION)
# ===============================
st.markdown("""
<style>
body {
    background-color: #0a0f1c;
    color: #e5e7eb;
}
.card {
    padding: 16px;
    border-radius: 10px;
    background-color: #111827;
    border: 1px solid #1f2937;
}
.metric-value {
    font-size: 26px;
    font-weight: 600;
}
.metric-label {
    font-size: 13px;
    color: #9ca3af;
}
.alert {
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}
.danger { background-color: #2a1212; color: #ef4444; }
.warning { background-color: #2a230f; color: #f59e0b; }
.success { background-color: #0f2a1b; color: #22c55e; }
</style>
""", unsafe_allow_html=True)

st.title("AI Intrusion Detection System")

# ===============================
# LOAD FEATURES
# ===============================
feature_columns = joblib.load("features.pkl")
API_URL = "https://intrusion-detection-system-6lvo.onrender.com/predict"

# ===============================
# SIDEBAR (SAFE LIMITS)
# ===============================
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    accept_multiple_files=True
)

st.sidebar.markdown("### Controls")

live_mode = st.sidebar.toggle("Live Mode", False)

# HARD LIMITS (IMPORTANT FOR RENDER)
max_rows = 5000
batch_size = 200

# ===============================
# CARD FUNCTION
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

        df = pd.read_csv(file, encoding="latin1", low_memory=False)
        df.columns = df.columns.str.strip()

        # DROP LABEL
        if "Label" in df.columns:
            df_features = df.drop(columns=["Label"])
        else:
            df_features = df.copy()

        # CLEAN
        df_features = df_features.apply(pd.to_numeric, errors='coerce')
        df_features = df_features.replace([np.inf, -np.inf], 0)
        df_features = df_features.fillna(0)

        # ALIGN FEATURES
        for col in feature_columns:
            if col not in df_features.columns:
                df_features[col] = 0

        df_features = df_features[feature_columns]

        # 🔥 ALWAYS SAMPLE (CRITICAL FIX)
        if len(df_features) > max_rows:
            df_features = df_features.sample(n=max_rows, random_state=42)

        # ===============================
        # BATCH PROCESSING (SAFE)
        # ===============================
        for i in range(0, len(df_features), batch_size):

            batch = df_features.iloc[i:i+batch_size]

            try:
                response = requests.post(
                    API_URL,
                    json={"features": batch.values.tolist()},
                    timeout=30
                )

                result = response.json()
                predictions = result.get("results", [])

            except Exception as e:
                st.error(f"API Error: {e}")
                break

            for res in predictions:

                total_records += 1

                conf = min(res["confidence"], 0.99)
                confidences.append(conf)

                pred = res["prediction"]

                if pred != "BENIGN":
                    total_attacks += 1
                    attack_counter[pred] += 1

                if res["anomaly"]:
                    total_anomalies += 1

        # 🔥 FREE MEMORY
        del df
        del df_features
        gc.collect()

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

    # ===============================
    # ALERTS
    # ===============================
    if total_attacks > 0:
        st.markdown(f'<div class="alert danger">{total_attacks} attacks detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert success">No attacks detected</div>', unsafe_allow_html=True)

    if total_anomalies > 0:
        st.markdown(f'<div class="alert warning">{total_anomalies} anomalies detected</div>', unsafe_allow_html=True)

    # ===============================
    # ATTACK CHART (LIGHTWEIGHT)
    # ===============================
    if attack_counter:
        st.subheader("Attack Distribution")
        chart_df = pd.DataFrame({
            "Attack": list(attack_counter.keys()),
            "Count": list(attack_counter.values())
        })
        st.bar_chart(chart_df.set_index("Attack"))

    # ===============================
    # SIMPLE RATIO (NO HEAVY PIE)
    # ===============================
    benign = total_records - total_attacks

    st.subheader("Traffic Summary")

    ratio_df = pd.DataFrame({
        "Type": ["Benign", "Attack"],
        "Count": [benign, total_attacks]
    })

    st.bar_chart(ratio_df.set_index("Type"))

else:
    st.info("Upload CSV file to start")