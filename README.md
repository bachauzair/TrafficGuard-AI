# TrafficGuard AI: Urban Accident Hotspot & Risk Mapping

TrafficGuard AI is a full-stack traffic dashboard that finds accident hotspots and predicts how severe an accident is likely to be. It runs on a subset of the US Accidents dataset: DBSCAN clusters accident locations into hotspots, and XGBoost predicts severity from environmental and time-based features. A FastAPI backend serves model predictions and a simulated real-time accident feed to a Flutter Web frontend. SHAP is used throughout so the severity predictions can actually be explained, rather than treated as a black box.

Screenshots
<img width="1917" height="1022" alt="1" src="https://github.com/user-attachments/assets/a5370d49-578c-46f0-912a-f9019b25356f" /> <img width="1917" height="1017" alt="2" src="https://github.com/user-attachments/assets/6c91708d-8026-49c8-af38-85c64fe65271" /> <img width="1917" height="1010" alt="3" src="https://github.com/user-attachments/assets/52d9143d-f420-423f-a949-58feea668cfb" /> <img width="476" height="305" alt="4" src="https://github.com/user-attachments/assets/9e80a5ee-a167-4e74-8355-ef2ac0575ed1" /> <img width="470" height="310" alt="5" src="https://github.com/user-attachments/assets/bd953f6c-249f-4bca-b956-902c667fe083" />
Technical architecture

TrafficGuard AI combines spatial data analysis, machine learning, and a cross-platform frontend.

Machine learning pipeline

The pipeline processes a subset of the US Accidents dataset: missing values are filled in with the median, and categorical variables are one-hot encoded. DBSCAN, using a haversine distance metric, clusters accident locations into dense geographic hotspots. For severity prediction, an XGBoost classifier was trained and compared against a Random Forest baseline, and XGBoost came out ahead on F1 score for the imbalanced classes. SHAP's TreeExplainer pulls out both overall and per-prediction feature importance, so you can see why the model flagged a given accident as severe rather than just trusting the label.

Backend (FastAPI)

The backend is built with FastAPI and Uvicorn. It exposes REST endpoints for streaming simulated accident data, fetching precomputed hotspots, and running inference with the XGBoost model.

Frontend (Flutter Web)

The frontend is built in Flutter Web with flutter_map. A color inversion matrix renders OpenStreetMap tiles in dark mode without needing a separate set of dark map tiles. State is managed with the provider package, data streams in asynchronously, the UI uses glassmorphism-style panels, and a prediction form queries the backend model directly.

Research abstract

Growing cities need data-driven systems to cut down on traffic-related risk. TrafficGuard AI applies spatial clustering and explainable machine learning to accident data to do this. Using DBSCAN with a haversine metric on the US Accidents dataset, the system identifies over 140 localized, high-risk hotspots. An XGBoost model predicts accident severity from weather, time, and road-topology features, picking up on non-linear relationships that simpler models tend to miss. SHAP is used to interpret the model's output, and visibility and temperature come out as the strongest drivers of severe accidents. The system runs as a FastAPI backend feeding a Flutter dashboard, giving planners a way to see where risk is concentrated and check which factors are driving a given prediction.

Keywords and research tags: Intelligent Transportation Systems (ITS), Smart Cities, Spatial Data Mining, Explainable AI (XAI), Machine Learning, XGBoost, DBSCAN, SHAP, Geospatial Visualization, Predictive Modeling

Running the project locally
1. Start the backend (Python)

Make sure Python is installed, then set up the environment and start the FastAPI server:

bash
cd api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8000

The API will be available at http://127.0.0.1:8000

2. Start the frontend (Flutter Web)

Make sure the Flutter SDK is installed, then run the web app:

bash
cd flutter_app
flutter clean
flutter pub get
flutter run -d chrome
Limitations and future work

This is a prototype, not a production system, and it has the limitations you'd expect from one. It runs on a static, historical slice of the data, and the "real-time" stream is simulated rather than pulled from live sensors. The DBSCAN parameters were tuned across the whole geographic area at once, so the clustering can miss patterns specific to a single neighborhood or intersection. Connecting a genuine real-time feed, such as city camera data or a service like Waze, would be a natural next step. It would also be worth trying spatiotemporal graph neural networks, since they're built to handle the sequential, directional structure of road networks in a way DBSCAN and XGBoost currently don't.
