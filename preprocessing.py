# Script to clean up data set.
# Some columns are not useful for training and could lead to overfitting or add noise.

import os
import pandas as pd
import numpy as np
import kagglehub
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# CONFIGURATION
SAMPLE_RATE = 1  # Usage: 0.1 = use 10% of data. Set to 1.0 for full dataset.
OUTPUT_DIR = "processed_data"

def load_and_process():
    # 1. Download Data
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("solarmainframe/ids-intrusion-csv")
    print(f"Dataset path: {path}")

    # 2. Load and Merge
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    li = []
    
    for filename in all_files:
        print(f"Reading {os.path.basename(filename)}...")
        try:
            df_temp = pd.read_csv(filename, index_col=None, header=0)
            if SAMPLE_RATE < 1.0:
                df_temp = df_temp.sample(frac=SAMPLE_RATE, random_state=42)
            li.append(df_temp)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    df = pd.concat(li, axis=0, ignore_index=True)
    
    # 3. Cleaning
    print("Cleaning data...")
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 4. BINARY MAPPING (The Key Change)
    print("Applying Binary Mapping...")
    # If label is 'Benign', set to 0. Otherwise (any attack), set to 1.
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    
    label_mapping = {0: 'Benign', 1: 'Attack'}
    print(f"Class Distribution:\n{df['Label'].value_counts()}")

    # 5. Splitting
    # Drop identifier columns to prevent overfitting
    drop_cols = ['Timestamp', 'Dst Port', 'Flow ID', 'Source IP', 'Src IP', 'Dst IP', 'Destination IP']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Ensure only numeric columns remain
    X = X.select_dtypes(include=[np.number])
    y = df['Label']

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 6. Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    joblib.dump(X_train, os.path.join(OUTPUT_DIR, 'X_train.pkl'))
    joblib.dump(X_test, os.path.join(OUTPUT_DIR, 'X_test.pkl'))
    joblib.dump(y_train, os.path.join(OUTPUT_DIR, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(OUTPUT_DIR, 'y_test.pkl'))
    joblib.dump(label_mapping, os.path.join(OUTPUT_DIR, 'label_mapping.pkl'))
    
    print("Binary preprocessing complete.")

if __name__ == "__main__":
    load_and_process()