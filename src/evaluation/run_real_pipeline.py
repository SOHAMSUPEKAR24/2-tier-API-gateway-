import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier

# Ensure path is correct for imports when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.real_feature_adapter import adapt_tier1_features, adapt_tier2_features

def run_real_pipeline():
    print("Loading Tier 1 Real Corpus (CIC-IDS)...")
    df_t1 = pd.read_csv('data/generated/tier1_real_corpus.csv')
    X_t1_all = adapt_tier1_features(df_t1)
    
    # Binarize labels
    y_t1_all = (df_t1['Label'] != 'Benign').astype(int).values
    
    print("Loading Tier 2 Real Corpus (EMBER)...")
    t2_records = []
    with open('data/generated/tier2_real_corpus.jsonl', 'r') as f:
        for line in f:
            t2_records.append(json.loads(line))
            
    X_t2_all = adapt_tier2_features(t2_records)
    y_t2_all = np.array([1 if r['label'] == 1 else 0 for r in t2_records])
    
    # We combine them row-by-row since the architecture assumes 
    # a single traffic flow has both Tier 1 and Tier 2 features available during evaluation.
    # In reality, Tier 2 features require deep inspection of a payload in that flow.
    # We match them up 1-to-1 just for the simulation loop.
    
    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(len(y_t1_all))
    
    X_t1_all = X_t1_all[indices]
    X_t2_all = X_t2_all[indices]
    y_all = y_t1_all[indices] # Labels are aligned by construction
    
    # 2. Train/Test Splits (Total: 10,000 rows)
    # Test Set: 2,000 rows (20%)
    # Tier-2 Training Set: 3,000 rows (30%)
    # Tier-1 Seed Set: 300 rows (3%)
    # Feedback Pool: 4,700 rows (47%)
    
    X_t1_test = X_t1_all[:2000]
    X_t2_test = X_t2_all[:2000]
    y_test = y_all[:2000]
    
    X_t1_t2_train = X_t1_all[2000:5000]
    X_t2_t2_train = X_t2_all[2000:5000]
    y_t2_train = y_all[2000:5000]
    
    X_t1_seed = X_t1_all[5000:5300]
    y_seed = y_all[5000:5300]
    
    X_t1_pool = X_t1_all[5300:]
    X_t2_pool = X_t2_all[5300:]
    y_pool = y_all[5300:]
    
    # 3. Model Initialization
    tier1_rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42)
    tier2_rf = RandomForestClassifier(n_estimators=250, random_state=42)
    
    # 4. The Feedback Loop Execution
    print("\nTraining Tier 2 on Tier-2 Training Set (3,000 rows)...")
    tier2_rf.fit(X_t2_t2_train, y_t2_train)
    
    print("Training Tier 1 on Tier-1 Seed Set (300 rows)...")
    tier1_rf.fit(X_t1_seed, y_seed)
    
    def evaluate_tier1(model, X_test, y_test):
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        acc = np.mean(preds == y_test)
        max_probs = np.max(probs, axis=1)
        esc_rate = np.mean((max_probs >= 0.30) & (max_probs <= 0.70))
        return acc, esc_rate
        
    acc_0, esc_0 = evaluate_tier1(tier1_rf, X_t1_test, y_test)
    print(f"\n--- Cycle 0 (Baseline) ---")
    print(f"Accuracy:        {acc_0 * 100:.2f}%")
    print(f"Escalation Rate: {esc_0 * 100:.2f}%")
    
    X_t1_train = list(X_t1_seed)
    y_train = list(y_seed)
    
    pool_size = len(y_pool)
    batch_size = pool_size // 6
    
    for cycle in range(1, 7):
        start_idx = (cycle - 1) * batch_size
        end_idx = start_idx + batch_size if cycle < 6 else pool_size
        
        X_batch_t1 = X_t1_pool[start_idx:end_idx]
        X_batch_t2 = X_t2_pool[start_idx:end_idx]
        
        probs = tier1_rf.predict_proba(X_batch_t1)
        max_probs = np.max(probs, axis=1)
        
        escalate_mask = (max_probs >= 0.30) & (max_probs <= 0.70)
        
        if np.any(escalate_mask):
            X_esc_t2 = X_batch_t2[escalate_mask]
            X_esc_t1 = X_batch_t1[escalate_mask]
            
            # Tier 2 definitive prediction
            tier2_preds = tier2_rf.predict(X_esc_t2)
            
            # Add to retraining set
            X_t1_train.extend(X_esc_t1)
            y_train.extend(tier2_preds)
            
            # Retrain Tier 1
            tier1_rf.fit(np.array(X_t1_train), np.array(y_train))
            
        print(f"Cycle {cycle} complete. Escalated {np.sum(escalate_mask)} samples.")
        
    acc_6, esc_6 = evaluate_tier1(tier1_rf, X_t1_test, y_test)
    print(f"\n--- Cycle 6 (Final) ---")
    print(f"Accuracy:        {acc_6 * 100:.2f}%")
    print(f"Escalation Rate: {esc_6 * 100:.2f}%")

if __name__ == "__main__":
    run_real_pipeline()
