import streamlit as st
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import time

def safe_api_call(payload, timeout=60):
    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)

        if response.status_code != 200:
            return None

        try:
            return response.json()
        except:
            return None

    except:
        return None

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# ===============================
# ADVANCED DARK UI (SMOOTH + NO GLOW)
# ===============================
st.markdown("""
<style>

body {
    background-color: #0a0f1c;
    color: #e5e7eb;
}

/* TITLE */
.title {
    font-size: 38px;
    font-weight: 600;
    margin-bottom: 20px;
    animation: fadeIn 1s ease-in;
}

/* CARDS */
.card {
    padding: 20px;
    border-radius: 14px;
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid #1f2937;
    transition: all 0.25s ease;
}

.card:hover {
    transform: translateY(-5px) scale(1.01);
    border: 1px solid #374151;
}

/* METRICS */
.metric-value {
    font-size: 30px;
    font-weight: 600;
}

.metric-label {
    font-size: 14px;
    color: #9ca3af;
}

/* ALERTS */
.alert {
    padding: 12px;
    border-radius: 8px;
    margin-top: 10px;
    animation: fadeIn 0.6s ease-in;
}

.danger { background-color: #2a1212; color: #ef4444; }
.warning { background-color: #2a230f; color: #f59e0b; }
.success { background-color: #0f2a1b; color: #22c55e; }

/* PROGRESS BAR */
.progress-bar {
    height: 6px;
    border-radius: 10px;
    background: #1f2937;
    overflow: hidden;
    margin-top: 10px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    animation: load 1.5s ease-in-out forwards;
}

/* ANIMATIONS */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes load {
    from { width: 0; }
    to { width: 100%; }
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">AI Intrusion Detection System</div>', unsafe_allow_html=True)

# ===============================
# LOAD FEATURES
# ===============================
feature_columns = joblib.load("features.pkl")
API_URL = "https://intrusion-detection-system-6lvo.onrender.com/predict"

# ===============================
# SIDEBAR
# ===============================
uploaded_files = st.sidebar.file_uploader(
    "Upload Network Logs",
    type=["csv"],
    accept_multiple_files=True
)

st.sidebar.markdown("### Controls")
live_mode = st.sidebar.toggle("Live Simulation", False)
max_rows = st.sidebar.slider("Max Rows", 1000, 20000, 10000, 1000)
batch_size = st.sidebar.slider("Batch Size", 100, 1000, 500, 100)

# ===============================
# CARD FUNCTION
# ===============================
def card(label, value):
    return f"""
    <div class="card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: 70%"></div>
        </div>
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

    y_true = []
    y_pred = []

    start_time = time.time()

    counter_placeholder = st.empty()
    live_placeholder = st.empty()

    for file in uploaded_files:

        df = pd.read_csv(file, encoding="latin1", low_memory=False)
        df.columns = df.columns.str.strip()

        if "Label" in df.columns:
            y_true.extend(df["Label"].tolist())
            df_features = df.drop(columns=["Label"])
        else:
            df_features = df.copy()

        # CLEAN
        df_features = df_features.apply(pd.to_numeric, errors='coerce')
        df_features = df_features.replace([np.inf, -np.inf], 0)
        df_features = df_features.fillna(0)

        # ALIGN
        for col in feature_columns:
            if col not in df_features.columns:
                df_features[col] = 0

        df_features = df_features[feature_columns]

        # LIMIT
        if live_mode and len(df_features) > max_rows:
            df_features = df_features.sample(n=max_rows, random_state=42)

        # ===============================
        # LIVE MODE (BATCHED)
        # ===============================
        if live_mode:

            for i in range(0, len(df_features), batch_size):

                batch = df_features.iloc[i:i+batch_size]

                result = safe_api_call({"features": batch.values.tolist()}, timeout=60)

                if result is None or "results" not in result:
                    continue

                predictions = result["results"]

                for res in predictions:

                    total_records += 1

                    conf = min(res["confidence"], 0.99)
                    confidences.append(conf)

                    pred = res["prediction"]
                    y_pred.append(pred)

                    if pred != "BENIGN":
                        total_attacks += 1
                        attack_counter[pred] += 1

                    if res["anomaly"]:
                        total_anomalies += 1

                # LIVE UI UPDATE
                counter_placeholder.markdown(f"""
                <div class="card">
                    🚨 Attacks: {total_attacks} &nbsp;&nbsp; | &nbsp;&nbsp;
                    📦 Processed: {total_records}
                </div>
                """, unsafe_allow_html=True)

                live_placeholder.markdown(f"""
                <div class="card">
                    <b>Last Prediction:</b> {pred}<br>
                    <b>Confidence:</b> {conf:.2f}<br>
                    <b>Anomaly:</b> {res['anomaly']}
                </div>
                """, unsafe_allow_html=True)

        # ===============================
        # NORMAL MODE
        # ===============================
        else:

            result = safe_api_call({"features": df_features.values.tolist()}, timeout=120)

            if result is None or "results" not in result:
                continue

            predictions = result["results"]

            for res in predictions:

                total_records += 1

                conf = min(res["confidence"], 0.99)
                confidences.append(conf)

                pred = res["prediction"]
                y_pred.append(pred)

                if pred != "BENIGN":
                    total_attacks += 1
                    attack_counter[pred] += 1

                if res["anomaly"]:
                    total_anomalies += 1

    end_time = time.time()

    # ===============================
    # METRICS
    # ===============================
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(card("Total Traffic", total_records), unsafe_allow_html=True)
    col2.markdown(card("Attacks", total_attacks), unsafe_allow_html=True)
    col3.markdown(card("Anomalies", total_anomalies), unsafe_allow_html=True)
    col4.markdown(card("Avg Confidence", f"{np.mean(confidences):.2f}"), unsafe_allow_html=True)

    # ===============================
    # PERFORMANCE
    # ===============================
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
    # CHARTS (SMOOTH)
    # ===============================
    if attack_counter:
        st.subheader("Attack Distribution")

        chart_data = pd.DataFrame({
            "Attack": list(attack_counter.keys()),
            "Count": list(attack_counter.values())
        })

        st.bar_chart(chart_data.set_index("Attack"))

    # ===============================
    # TRAFFIC RATIO
    # ===============================
    benign = total_records - total_attacks

    st.subheader("Traffic Ratio")

    ratio_df = pd.DataFrame({
        "Type": ["Benign", "Attack"],
        "Count": [benign, total_attacks]
    })

    st.bar_chart(ratio_df.set_index("Type"))
        # ===============================
    # PIE CHART (ATTACK vs BENIGN)
    # ===============================
    st.subheader("Attack vs Benign (%)")

    benign = total_records - total_attacks

    fig, ax = plt.subplots()

    ax.pie(
        [benign, total_attacks],
        labels=["Benign", "Attack"],
        autopct="%1.1f%%",
        colors=["#0f766e", "#1e3a8a"]
    )

    ax.set_ylabel("")
    st.pyplot(fig)
else:
    st.info("Upload dataset to start")
