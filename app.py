from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import time
import os

# ===============================
# 🔥 LIMIT CPU + MEMORY USAGE
# ===============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = Flask(__name__)

# ===============================
# LAZY LOAD MODELS
# ===============================
xgb_model = None
iso_model = None
scaler = None
le = None
feature_columns = None


def load_models():
    global xgb_model, iso_model, scaler, le, feature_columns

    if xgb_model is None:
        xgb_model = joblib.load("ids_xgboost_multiclass.pkl")
        iso_model = joblib.load("isolation_forest.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        feature_columns = joblib.load("features.pkl")

        # 🔥 LIMIT XGBOOST THREADS
        try:
            xgb_model.set_params(n_jobs=1)
        except:
            pass


@app.route("/")
def home():
    return "IDS API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    try:
        # LOAD MODELS ONLY WHEN NEEDED
        load_models()

        req = request.get_json()

        if not req or "features" not in req:
            return jsonify({"error": "No features provided"}), 400

        data = req["features"]

        # ===============================
        # PREPROCESS
        # ===============================
        df = pd.DataFrame(data, columns=feature_columns)

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        # SCALE
        df_scaled = scaler.transform(df)

        # ===============================
        # PREDICTION
        # ===============================
        xgb_proba = xgb_model.predict_proba(df_scaled)
        if_preds = iso_model.predict(df_scaled)

        # ===============================
        # LOGIC
        # ===============================
        max_idx = np.argmax(xgb_proba, axis=1)
        max_prob = np.max(xgb_proba, axis=1)

        forced_attack = (max_idx == 0) & (max_prob < 0.4)
        final_idx = np.where(forced_attack, 1, max_idx)

        labels = le.inverse_transform(final_idx)
        anomalies = (if_preds == -1)

        # ===============================
        # RESPONSE
        # ===============================
        results = [
            {
                "prediction": labels[i],
                "confidence": float(max_prob[i]),
                "anomaly": bool(anomalies[i])
            }
            for i in range(len(labels))
        ]

        end_time = time.time()

        return jsonify({
            "results": results,
            "meta": {
                "rows": len(results),
                "processing_time_sec": round(end_time - start_time, 3)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# LOCAL RUN
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)