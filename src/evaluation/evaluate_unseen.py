import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json

# Ensure path is correct for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.real_feature_adapter import adapt_tier1_features, adapt_tier2_features

def run_evaluation_on_new_dataset():
    print("==================================================")
    print("PHASE 1: RE-TRAINING THE MODEL ON ORIGINAL DATA")
    print("==================================================")
    df_t1 = pd.read_csv('data/generated/tier1_real_corpus.csv')
    X_t1_all = adapt_tier1_features(df_t1)
    y_t1_all = (df_t1['Label'] != 'Benign').astype(int).values
    
    t2_records = []
    with open('data/generated/tier2_real_corpus.jsonl', 'r') as f:
        for line in f:
            t2_records.append(json.loads(line))
            
    X_t2_all = adapt_tier2_features(t2_records)
    y_t2_all = np.array([1 if r['label'] == 1 else 0 for r in t2_records])
    
    np.random.seed(42)
    indices = np.random.permutation(len(y_t1_all))
    X_t1_all = X_t1_all[indices]
    X_t2_all = X_t2_all[indices]
    y_all = y_t1_all[indices] 

    # We will use all 10,000 rows to train the models to their final state
    tier1_rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42)
    tier2_rf = RandomForestClassifier(n_estimators=250, random_state=42)
    
    tier2_rf.fit(X_t2_all, y_all)
    tier1_rf.fit(X_t1_all, y_all)
    
    print("Models fully trained on the original 10,000-row dataset.")
    
    print("\n==================================================")
    print("PHASE 2: EXTRACTING A COMPLETELY NEW, UNSEEN DATASET")
    print("==================================================")
    # We will extract from deep within the 7GB cic.csv file to ensure 
    # totally different traffic (different attacks, different times).
    cic_path = '/Users/sohamsupekar/Desktop/MajorProject/DATASET/cic.csv'
    
    print("Extracting 2,000 brand new rows from deep within the 7GB dataset...")
    
    # Read the header
    df_head = pd.read_csv(cic_path, nrows=1)
    label_col = [c for c in df_head.columns if 'label' in c.lower()][0]
    
    benign_list = []
    attack_list = []
    
    # Skip the first 1,000,000 rows to guarantee completely unseen data
    # (The original data only used the first ~50,000 rows)
    chunk_iterator = pd.read_csv(cic_path, chunksize=50000, skiprows=range(1, 1000000))
    
    for chunk in chunk_iterator:
        # Re-apply the column names since we skipped rows
        chunk.columns = df_head.columns 
        
        if len(benign_list) < 1000:
            b_chunk = chunk[chunk[label_col] == 'Benign']
            benign_list.extend(b_chunk.to_dict('records'))
            
        if len(attack_list) < 1000:
            a_chunk = chunk[chunk[label_col] != 'Benign']
            attack_list.extend(a_chunk.to_dict('records'))
            
        if len(benign_list) >= 1000 and len(attack_list) >= 1000:
            break
            
    # Combine into a new dataframe
    new_data = benign_list[:1000] + attack_list[:1000]
    df_unseen = pd.DataFrame(new_data)
    
    print(f"Extraction complete! New Dataset Shape: {df_unseen.shape}")
    print("New Attacks in this dataset:")
    print(df_unseen[label_col].value_counts())
    
    print("\n==================================================")
    print("PHASE 3: EVALUATING FINAL MODEL ON NEW DATASET")
    print("==================================================")
    
    X_unseen = adapt_tier1_features(df_unseen)
    y_unseen = (df_unseen[label_col] != 'Benign').astype(int).values
    
    preds = tier1_rf.predict(X_unseen)
    probs = tier1_rf.predict_proba(X_unseen)
    
    accuracy = np.mean(preds == y_unseen)
    
    max_probs = np.max(probs, axis=1)
    escalation_rate = np.mean((max_probs >= 0.30) & (max_probs <= 0.70))
    
    print(f"\n--- Results on 100% UNSEEN 'WHOLE NEW BOOK' DATASET ---")
    print(f"Accuracy:        {accuracy * 100:.2f}%")
    print(f"Escalation Rate: {escalation_rate * 100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation_on_new_dataset()
