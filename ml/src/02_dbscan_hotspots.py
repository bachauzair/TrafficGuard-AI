import pandas as pd
import numpy as np
import json
from sklearn.cluster import DBSCAN
from math import radians

print("Loading cleaned dataset for DBSCAN...")
df = pd.read_csv('../data/cleaned_accidents.csv')

# DBSCAN on a sample to make it computationally feasible, e.g. 50,000 for hotspot detection
# In a real scenario we'd do it per city/region or use HDBSCAN
sample_size = min(50000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy()

# Convert lat/lng to radians for haversine metric
coords = np.radians(df_sample[['Start_Lat', 'Start_Lng']].values)

# eps in radians (e.g., 500 meters)
# Earth radius in km = 6371.0088
# 500 meters = 0.5 km
eps_rad = 0.5 / 6371.0088

print("Running DBSCAN...")
db = DBSCAN(eps=eps_rad, min_samples=20, algorithm='ball_tree', metric='haversine').fit(coords)
df_sample['Cluster'] = db.labels_

clusters = df_sample[df_sample['Cluster'] != -1]
print(f"Found {len(clusters['Cluster'].unique())} hotspots.")

hotspots_data = []
for cluster_id in clusters['Cluster'].unique():
    c_data = clusters[clusters['Cluster'] == cluster_id]
    
    avg_lat = c_data['Start_Lat'].mean()
    avg_lng = c_data['Start_Lng'].mean()
    count = len(c_data)
    avg_sev = c_data['Severity'].mean()
    peak_hour = c_data['Hour'].mode()[0]
    common_weather = c_data['Weather_Category'].mode()[0]
    
    # Simple risk classification
    if avg_sev >= 3.0 or count > 100:
        risk = 'High'
    elif avg_sev >= 2.5 or count > 50:
        risk = 'Medium'
    else:
        risk = 'Low'
        
    hotspots_data.append({
        'id': int(cluster_id),
        'lat': float(avg_lat),
        'lng': float(avg_lng),
        'accident_count': int(count),
        'avg_severity': float(avg_sev),
        'risk_level': risk,
        'peak_hour': int(peak_hour),
        'common_weather': str(common_weather)
    })

with open('../data/hotspots.json', 'w') as f:
    json.dump(hotspots_data, f, indent=2)

print("Hotspots saved to hotspots.json")
