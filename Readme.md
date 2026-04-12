# AI Intrusion Detection System (IDS)

## Overview

This project implements a machine learning-based Intrusion Detection System (IDS) for analyzing network traffic and detecting malicious activities.
It combines supervised learning for attack classification and unsupervised learning for anomaly detection, providing both accuracy and robustness.

The system includes:

* Machine learning-based attack classification
* Anomaly detection for unknown or suspicious patterns
* Flask-based backend API
* Streamlit-based dashboard interface

---

## Tech Stack

* Python
* XGBoost (multi-class classification)
* Isolation Forest (anomaly detection)
* Flask (backend API)
* Streamlit (dashboard UI)
* Pandas, NumPy, Matplotlib, Seaborn

---

## Project Structure

```
project/
│
├── app.py
├── ui.py
├── features.pkl
├── scaler.pkl
├── label_encoder.pkl
├── ids_xgboost_multiclass.pkl
├── isolation_forest.pkl
├── final_realistic_dataset.csv
└── README.md
```

---

## Features

* Multi-class attack detection (DoS, DDoS, PortScan, etc.)
* Real-time traffic simulation mode
* Anomaly detection using Isolation Forest
* Attack type distribution visualization
* Confusion matrix for performance evaluation
* Confidence score tracking
* Interactive dashboard with configurable parameters

---

## Dataset

* Based on the CICIDS2017 dataset
* A custom dataset is created by:

  * Sampling multiple traffic files
  * Combining benign and attack traffic
  * Shuffling for realistic distribution

---

## Model Details

### XGBoost Classifier

* Used for multi-class classification of network traffic
* Trained on a balanced dataset
* Outputs predicted attack type and confidence score

### Isolation Forest

* Trained only on benign traffic
* Identifies anomalous or previously unseen patterns

---

## How to Run

### 1. Start Backend

```
python app.py
```

### 2. Run Dashboard

```
streamlit run ui.py
```

### 3. Use the System

* Upload a CSV file in CICIDS format
* View detection results and analytics in the dashboard

---

## Deployment

To deploy the system:

* Update the API URL in `ui.py`:

```
API_URL = "https://your-cloud-url/predict"
```

* Deploy using platforms such as Render, Railway, AWS, or Azure

---

## Output

The system provides:

* Total traffic processed
* Number of detected attacks
* Number of anomalies
* Attack distribution charts
* Confusion matrix
* Confidence analysis

---

## Limitations

* Confidence scores from tree-based models may be overestimated
* Model performance depends on dataset quality and balance
* Real-time deployment requires integration with live network data

---

## Future Improvements

* Integration with real-time packet capture
* Model calibration for better probability estimation
* Real-time alerting system
* Enhanced dashboard for SOC environments
* Cloud-native deployment and scaling



Developed as a machine learning and cloud security project focused on intelligent network intrusion detection.

