# TrafficGuard AI: Urban Accident Hotspot & Risk Mapping

![TrafficGuard AI](https://img.shields.io/badge/Status-Active-brightgreen) ![Flutter](https://img.shields.io/badge/Frontend-Flutter_Web-02569B?logo=flutter) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi) ![Python](https://img.shields.io/badge/ML-XGBoost-3776AB?logo=python)

TrafficGuard AI is an intelligent, full-stack urban traffic dashboard designed to detect accident hotspots and predict accident severity. Built using a US Accidents dataset, the system employs unsupervised spatial machine learning (DBSCAN) to geographically cluster high-risk zones, and supervised learning (XGBoost) to predict accident impact based on environmental and temporal features. The architecture features a Python FastAPI backend that serves model inferences and a simulated real-time accident stream to a responsive, cross-platform Flutter Web frontend. The project emphasizes Explainable AI (XAI) using SHAP values to provide actionable insights into the leading causes of severe traffic incidents, demonstrating a robust prototype for smart-city infrastructure.

---

## 📸 Screenshots

*(Replace the placeholders below with the screenshots you captured!)*

### Live Simulation Dashboard
*(Insert Screenshot 5 here - e.g., `![Live Dashboard](images/screenshot5.png)`)*

### Machine Learning & Explainable AI
*(Insert Screenshots 2 & 3 here - The Predictor Form and Result)*

### Macro Data Overview
*(Insert Screenshot 1 or 4 here - The Full US View)*

---

## 🛠 Technical Architecture

TrafficGuard AI is a microservices-oriented application bridging spatial data science, machine learning, and cross-platform UI development. 

**Machine Learning Pipeline:** 
The data pipeline preprocesses a subset of the US Accidents dataset, applying median imputation for missing values and one-hot encoding for categorical variables. Unsupervised spatial clustering is executed via the DBSCAN algorithm using a haversine distance metric to extract dense geographical accident hotspots. For predictive modeling, an XGBoost classifier was trained and evaluated against a Random Forest baseline to predict multi-class accident severity, achieving superior F1-scores on imbalanced data. To ensure model transparency, the system integrates SHAP (SHapley Additive exPlanations) TreeExplainer, extracting both global and local feature importance metrics.

**Backend (FastAPI):**
Constructed with FastAPI and Uvicorn, the backend exposes RESTful endpoints for streaming simulated accident data, fetching pre-calculated hotspots, and serving real-time inference requests to the XGBoost model.

**Frontend (Flutter Web):**
Built in Flutter Web, leveraging `flutter_map` with a mathematical color inversion matrix for highly performant, dynamic dark-mode rendering of OpenStreetMap tiles. The UI features state management via the `provider` package, asynchronous data streaming, glassmorphism UI components, and a functional prediction form that dynamically queries the ML backend.

---

## 🎓 Academic Research Abstract

The rapid expansion of urban centers necessitates data-driven Intelligent Transportation Systems (ITS) to mitigate traffic-related hazards. This research presents TrafficGuard AI, an integrated system leveraging spatial clustering and explainable machine learning to analyze urban traffic accidents. Utilizing the US Accidents dataset, we apply the DBSCAN algorithm with a haversine metric to successfully identify over 140 localized, high-risk accident hotspots. Concurrently, an XGBoost classification model is developed to predict accident severity by analyzing complex, non-linear relationships among meteorological conditions, temporal factors, and road topology. To overcome the "black box" nature of ensemble trees, SHAP is integrated to interpret predictive outputs, revealing factors such as visibility and temperature as primary drivers of severe incidents. The methodologies are deployed via a FastAPI backend to a real-time, interactive Flutter dashboard, providing a scalable, transparent framework for smart-city planners to visualize risk density and understand causal accident factors.

**Keywords & Research Tags:**
`Intelligent Transportation Systems (ITS)`, `Smart Cities`, `Spatial Data Mining`, `Explainable AI (XAI)`, `Machine Learning`, `XGBoost`, `DBSCAN`, `SHAP`, `Geospatial Visualization`, `Predictive Modeling`

---

## 🚀 How to Run the Project Locally

### 1. Start the Backend API (Python)
Ensure you have Python installed, then set up the environment and start the FastAPI server:
```bash
cd api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```
*The API will be available at `http://127.0.0.1:8000`*

### 2. Start the Frontend (Flutter Web)
Ensure you have the Flutter SDK installed, then run the web app:
```bash
cd flutter_app
flutter clean
flutter pub get
flutter run -d chrome
```

---

## 🔮 Limitations and Future Work
While TrafficGuard AI demonstrates a robust conceptual framework, it currently relies on a statically processed subset of historical data and simulates real-time streaming, rather than connecting to live IoT traffic sensors. The DBSCAN clustering parameters were generalized across a vast geographic area, which may overlook micro-level urban topologies. Future work should focus on integrating genuine real-time data APIs (e.g., Waze or city camera feeds) and exploring Spatiotemporal Graph Neural Networks (STGNNs) to account for the sequential, directional nature of road networks, thereby improving predictive accuracy in dense, complex city grids.
