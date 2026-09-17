import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

print("Loading cleaned dataset for Modeling...")
df = pd.read_csv('../data/cleaned_accidents.csv')

# Drop NA rows
df = df.dropna()

X = df.drop(columns=['ID', 'Severity', 'Start_Lat', 'Start_Lng'])
y = df['Severity']

# We need to map y to 0-indexed for XGBoost
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, '../models/label_encoder.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

numeric_features = ['Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Humidity(%)', 'Hour', 'DayOfWeek', 'Month']
categorical_features = ['Weather_Category', 'Season', 'Is_Weekend', 'Junction', 'Traffic_Signal', 'Crossing', 'Station', 'Sunrise_Sunset']
# City and State are too high cardinality for simple OHE without explosion. We'll drop them for the ML model for simplicity, or use frequency encoding.
X_train = X_train.drop(columns=['City', 'State'])
X_test = X_test.drop(columns=['City', 'State'])

# Convert categories to pandas category type for XGBoost native categorical support, or use OHE
X_train_encoded = pd.get_dummies(X_train, columns=['Weather_Category', 'Sunrise_Sunset'])
X_test_encoded = pd.get_dummies(X_test, columns=['Weather_Category', 'Sunrise_Sunset'])

# Align columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train_scaled[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test_scaled[numeric_features])

joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(list(X_train_encoded.columns), '../models/feature_names.pkl')

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return acc, p, r, f

rf_metrics = get_metrics(y_test, rf_preds)
xgb_metrics = get_metrics(y_test, xgb_preds)

print("Random Forest Metrics:", rf_metrics)
print("XGBoost Metrics:", xgb_metrics)

best_model = xgb_model if xgb_metrics[3] >= rf_metrics[3] else rf
best_name = "XGBoost" if xgb_metrics[3] >= rf_metrics[3] else "Random Forest"
print(f"Selected Best Model: {best_name}")

joblib.dump(best_model, '../models/best_model.pkl')

metrics_data = {
    'Random Forest': {'Accuracy': rf_metrics[0], 'Precision': rf_metrics[1], 'Recall': rf_metrics[2], 'F1': rf_metrics[3]},
    'XGBoost': {'Accuracy': xgb_metrics[0], 'Precision': xgb_metrics[1], 'Recall': xgb_metrics[2], 'F1': xgb_metrics[3]},
    'Best': best_name
}
with open('../outputs/metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=2)

print("Generating SHAP values...")
# Use a background sample for SHAP
background = shap.sample(X_train_scaled, 100)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(background)

# Feature importance based on SHAP
if isinstance(shap_values, list):
    mean_shap = np.abs(shap_values[0]).mean(axis=0)
else:
    mean_shap = np.abs(shap_values).mean(axis=0)
    
if len(mean_shap.shape) > 1:
    mean_shap = mean_shap.mean(axis=0) # average over classes if multi-class

feature_importance = pd.DataFrame({'feature': X_train_scaled.columns, 'importance': mean_shap})
feature_importance = feature_importance.sort_values('importance', ascending=False)
feature_importance.to_csv('../outputs/feature_importance.csv', index=False)

print("Modeling complete.")
