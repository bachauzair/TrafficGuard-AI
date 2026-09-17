import pandas as pd
import json
import os

file_path = 'US_Accidents_March23.csv'
try:
    if not os.path.exists(file_path):
        print(json.dumps({'error': f'File not found: {os.path.abspath(file_path)}'}))
    else:
        # Read just a small sample to inspect columns and basic info
        df = pd.read_csv(file_path, nrows=5)
        info = {
            'columns': list(df.columns),
            'shape': None, # We don't read full to save time, but we can check filesize or use wc -l later
        }
        print(json.dumps(info, indent=2))
except Exception as e:
    print(json.dumps({'error': str(e)}))
