# Intrusion Detection System (IDS)

## Overview

This project is a cloud-deployed AI-based Intrusion Detection System that analyzes network traffic and classifies it as benign or malicious. It uses machine learning models trained on the CICIDS dataset and provides an interactive dashboard for visualization and analysis.

The system supports real-time simulation, batch processing of network logs, and displays attack distributions and traffic insights.

---

## Features

* Multi-class attack classification using XGBoost
* Anomaly detection using Isolation Forest
* Real-time simulation mode for faster analysis
* Interactive Streamlit dashboard
* Attack type distribution visualization
* Traffic split visualization (Benign vs Attack)
* Cloud deployment using AWS EC2
* Handles large datasets with optimized preprocessing

---

## Tech Stack

* Python
* Streamlit
* XGBoost
* Scikit-learn
* Pandas, NumPy
* Matplotlib
* AWS EC2

---

## Dataset

The model is trained on the CICIDS dataset, which includes various types of network traffic such as:

* Benign traffic
* DDoS
* Port Scan
* Web Attacks
* Infiltration

A balanced dataset was created by sampling from multiple CSV files to ensure fair representation of attack types.

---

## Project Structure

```
intrusion-detection-system/
│
├── app.py                     # Main Streamlit application
├── requirements.txt          # Dependencies
├── features.pkl              # Feature column order
├── scaler.pkl                # Feature scaler
├── label_encoder.pkl         # Label encoder
├── isolation_forest.pkl      # Anomaly detection model
├── xgb_model.json            # XGBoost model (JSON format)
├── README.md                 # Documentation
└── dataset (optional)        # Sample dataset for testing
```

---

## How It Works

1. Upload a CSV file containing network traffic data
2. Data is cleaned and aligned with training features
3. Features are scaled using a pre-trained scaler
4. XGBoost predicts attack type
5. Isolation Forest detects anomalies
6. Results are displayed with metrics and visualizations

---

## Running Locally

### 1. Clone the repository

```
git clone https://github.com/nvcidd/intrusion-detection-system.git
cd intrusion-detection-system
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
streamlit run app.py
```

### 4. Open in browser

```
http://localhost:8501
```

---

## Cloud Deployment (AWS EC2)

### 1. Launch EC2 Instance

* Choose Amazon Linux or Ubuntu
* Allow inbound traffic on ports:

  * 22 (SSH)
  * 8501 (Streamlit)

---

### 2. Connect to EC2

```
ssh -i IDS.pem ubuntu@<PUBLIC_IP>
```

---

### 3. Install dependencies

```
sudo apt update
sudo apt install python3-pip -y
pip3 install -r requirements.txt
```

---

### 4. Clone project

```
git clone https://github.com/nvcidd/intrusion-detection-system.git
cd intrusion-detection-system
```

---

### 5. Run application

```
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

### 6. Access application

```
http://<PUBLIC_IP>:8501
```

---

## Usage Notes

* Enable "Live Simulation" for faster processing on large datasets
* Use smaller datasets (1k–10k rows) for cloud deployment
* Large datasets can be tested locally
* Public IP may change when instance is restarted

---

## Performance Considerations

* CPU usage is limited to prevent crashes on low-memory instances
* Batch processing was removed to avoid Streamlit rerun issues
* Models are loaded lazily for memory efficiency

---

## Future Improvements

* Add domain and HTTPS support
* Deploy using Docker
* Add real-time network packet capture
* Improve UI with advanced analytics
* Auto-start service on EC2 boot


