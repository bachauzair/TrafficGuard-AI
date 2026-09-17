import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os
import json

# Ensure directories exist
os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../data', exist_ok=True)

print("Loading dataset...")
# Load a sample to keep memory usage reasonable, e.g., 500,000 records
file_path = '../../US_Accidents_March23.csv'
df = pd.read_csv(file_path, nrows=500000)

print(f"Original shape: {df.shape}")

# 1. Feature Engineering (Temporal)
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df.dropna(subset=['Start_Time'], inplace=True)
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Season: 1=Winter, 2=Spring, 3=Summer, 4=Fall
def get_season(month):
    if month in [12, 1, 2]: return 1
    elif month in [3, 4, 5]: return 2
    elif month in [6, 7, 8]: return 3
    else: return 4
df['Season'] = df['Month'].apply(get_season)

# 2. Weather features (Simplify weather conditions)
def simplify_weather(weather):
    if pd.isna(weather): return 'Unknown'
    weather = weather.lower()
    if 'clear' in weather or 'fair' in weather: return 'Clear'
    if 'cloud' in weather or 'overcast' in weather: return 'Cloudy'
    if 'rain' in weather or 'drizzle' in weather or 'shower' in weather: return 'Rain'
    if 'snow' in weather or 'ice' in weather or 'sleet' in weather: return 'Snow'
    if 'fog' in weather or 'mist' in weather or 'haze' in weather: return 'Fog'
    if 'thunder' in weather or 't-storm' in weather: return 'Storm'
    return 'Other'

df['Weather_Category'] = df['Weather_Condition'].apply(simplify_weather)

# 3. Handle Missing Values
cols_to_fill_median = ['Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Humidity(%)']
for col in cols_to_fill_median:
    df[col].fillna(df[col].median(), inplace=True)

# 4. Select features for ML and Frontend
features = [
    'ID', 'Severity', 'Start_Lat', 'Start_Lng', 
    'Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Humidity(%)',
    'Weather_Category', 'Hour', 'DayOfWeek', 'Month', 'Season', 'Is_Weekend',
    'Junction', 'Traffic_Signal', 'Crossing', 'Station', 'Sunrise_Sunset',
    'City', 'State'
]
df = df[features].copy()

# Fill categorical missing
df['Sunrise_Sunset'].fillna('Day', inplace=True)
df['City'].fillna('Unknown', inplace=True)

# Convert boolean to int
bool_cols = ['Junction', 'Traffic_Signal', 'Crossing', 'Station']
for col in bool_cols:
    df[col] = df[col].astype(int)

print(f"Cleaned shape: {df.shape}")
df.to_csv('../data/cleaned_accidents.csv', index=False)

# EDA Visualizations
plt_sns.set_theme(style="darkgrid")

# Severity Distribution
plt.figure(figsize=(8,5))
plt_sns.countplot(data=df, x='Severity', palette='viridis')
plt.title('Severity Distribution')
plt.savefig('../outputs/figures/severity_dist.png')
plt.close()

# Accidents by Hour
plt.figure(figsize=(10,5))
plt_sns.countplot(data=df, x='Hour', palette='magma')
plt.title('Accidents by Hour of Day')
plt.savefig('../outputs/figures/accidents_by_hour.png')
plt.close()

# Accidents by Weather
plt.figure(figsize=(10,5))
plt_sns.countplot(data=df, x='Weather_Category', order=df['Weather_Category'].value_counts().index, palette='crest')
plt.title('Accidents by Weather Category')
plt.savefig('../outputs/figures/accidents_by_weather.png')
plt.close()

print("EDA and preprocessing complete. Data saved.")

# Save a subset for the frontend simulation (e.g. 1000 records)
stream_df = df.sample(n=1000, random_state=42).sort_values('Hour')
stream_data = []
for _, row in stream_df.iterrows():
    stream_data.append({
        'id': row['ID'],
        'lat': row['Start_Lat'],
        'lng': row['Start_Lng'],
        'severity': int(row['Severity']),
        'hour': int(row['Hour']),
        'weather': row['Weather_Category'],
        'city': row['City'],
        'state': row['State']
    })
with open('../data/streaming_sample.json', 'w') as f:
    json.dump(stream_data, f)
print("Simulation streaming data saved.")
