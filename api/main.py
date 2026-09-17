from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import SeverityRequest, SeverityResponse
import joblib
import pandas as pd
import json
import os

app = FastAPI(title="TrafficGuard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../ml/models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../ml/data')

model = None
scaler = None
feature_names = None
label_encoder = None

@app.on_event("startup")
def load_models():
    global model, scaler, feature_names, label_encoder
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    except Exception as e:
        print("Warning: Could not load models. ML models need to be trained first.")

@app.get("/")
def read_root():
    return {"project": "TrafficGuard AI", "status": "running"}

@app.get("/accidents/hotspots")
def get_hotspots():
    try:
        with open(os.path.join(DATA_DIR, 'hotspots.json'), 'r') as f:
            hotspots = json.load(f)
        return hotspots
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Hotspots data not found")

@app.get("/accidents/stream")
def get_stream(batch_size: int = 100):
    try:
        with open(os.path.join(DATA_DIR, 'streaming_sample.json'), 'r') as f:
            stream_data = json.load(f)
        return stream_data[:batch_size]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Streaming data not found")

@app.get("/accidents/stats")
def get_stats():
    # Simple mocked stats based on expected data for UI rendering speed, or calculated
    return {
        "Total Accidents": 500000,
        "Hotspots Detected": 100, # dynamic ideally
        "Status": "Live Simulation Ready"
    }

@app.post("/predict/severity", response_model=SeverityResponse)
def predict_severity(request: SeverityRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert request to df
    data_dict = request.dict(by_alias=True)
    df = pd.DataFrame([data_dict])
    
    # One-hot encode what we can, missing columns filled with 0
    df_encoded = pd.get_dummies(df, columns=['Weather_Category', 'Sunrise_Sunset'])
    
    # Reindex to match feature_names
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Scale numericals
    numeric_features = ['Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Humidity(%)', 'Hour', 'DayOfWeek', 'Month']
    df_encoded[numeric_features] = scaler.transform(df_encoded[numeric_features])
    
    # Predict
    pred = model.predict(df_encoded)[0]
    
    # Note: if it's XGBoost, predict_proba returns prob array
    try:
        proba = model.predict_proba(df_encoded)[0]
        confidence = float(max(proba) * 100)
    except:
        confidence = 0.0
        
    severity_int = int(label_encoder.inverse_transform([pred])[0])
    
    # Simple top factors mock based on dummy SHAP for now, you can hook real SHAP here
    top_factors = [
        {"name": "Visibility", "value": 0.72, "direction": "increasing"},
        {"name": "Weather", "value": 0.45, "direction": "increasing"}
    ]
    
    return SeverityResponse(
        predicted_severity=severity_int,
        severity_label=f"Severity {severity_int}",
        confidence=confidence,
        top_factors=top_factors
    )
