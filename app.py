from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import time

app = Flask(__name__)

# ===============================
# LOAD MODELS (LOAD ONCE)
# ===============================
xgb_model = joblib.load("ids_xgboost_multiclass.pkl")
iso_model = joblib.load("isolation_forest.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("features.pkl")


@app.route("/")
def home():
    return "🚨 IDS API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    try:
        req = request.get_json()

        # ===============================
        # VALIDATION
        # ===============================
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
        # 🔥 BATCH PREDICTION
        # ===============================
        xgb_proba = xgb_model.predict_proba(df_scaled)
        if_preds = iso_model.predict(df_scaled)

        # ===============================
        # 🔥 VECTORIZED LOGIC (FASTER)
        # ===============================
        max_idx = np.argmax(xgb_proba, axis=1)
        max_prob = np.max(xgb_proba, axis=1)

        # Apply threshold rule
        forced_attack = (max_idx == 0) & (max_prob < 0.4)
        final_idx = np.where(forced_attack, 1, max_idx)

        labels = le.inverse_transform(final_idx)

        anomalies = (if_preds == -1)

        # ===============================
        # FORMAT RESPONSE
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


if __name__ == "__main__":
    app.run(debug=False, threaded=True)