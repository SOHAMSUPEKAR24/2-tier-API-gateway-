import os
import pandas as pd
import json

def verify_cic_ids():
    cic_path = '/Users/sohamsupekar/Desktop/MajorProject/DATASET/cic.csv'
    
    print("=== 1. CIC-IDS Verification (Tier 1) ===")
    print(f"Absolute Path: {cic_path}")
    
    if not os.path.exists(cic_path):
        print(f"ERROR: File not found at {cic_path}")
        return
        
    size_bytes = os.path.getsize(cic_path)
    size_gb = size_bytes / (1024**3)
    print(f"File Size: {size_gb:.2f} GB")
    
    # Read first 5 rows
    df_head = pd.read_csv(cic_path, nrows=5)
    print("\nExact Column Names:")
    print(list(df_head.columns))
    
    print("\nDistribution of 'Label' column for the first 10,000 rows:")
    # It might be 'Label', ' Label', etc. We'll find it by case-insensitive match
    label_col = [c for c in df_head.columns if 'label' in c.lower()]
    if not label_col:
        print("ERROR: 'Label' column not found.")
        return
    label_col = label_col[0]
    
    # Read in chunks for the first 10,000 rows
    label_counts = pd.Series(dtype=int)
    chunk_size = 5000
    rows_read = 0
    
    for chunk in pd.read_csv(cic_path, usecols=[label_col], chunksize=chunk_size):
        label_counts = label_counts.add(chunk[label_col].value_counts(), fill_value=0)
        rows_read += len(chunk)
        if rows_read >= 10000:
            break
            
    print(label_counts.to_string())

def verify_ember():
    ember_path = '/Users/sohamsupekar/Desktop/MajorProject/DATASET/ember2018/train_features_0.jsonl'
    
    print("\n=== 2. EMBER 2018 Verification (Tier 2) ===")
    print(f"Absolute Path: {ember_path}")
    
    if not os.path.exists(ember_path):
        print(f"ERROR: File not found at {ember_path}")
        return
        
    size_bytes = os.path.getsize(ember_path)
    size_gb = size_bytes / (1024**3)
    print(f"File Size: {size_gb:.2f} GB")
    
    print("\nTop-level JSON keys in first 2 rows:")
    with open(ember_path, 'r') as f:
        for i in range(2):
            line = f.readline()
            if not line:
                break
            try:
                data = json.loads(line)
                print(f"Row {i+1} keys: {list(data.keys())}")
            except Exception as e:
                print(f"Row {i+1} parse error: {e}")

if __name__ == "__main__":
    verify_cic_ids()
    verify_ember()
