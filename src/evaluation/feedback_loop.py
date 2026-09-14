import sys
import os
import pandas as pd
import numpy as np

# Ensure path is correct for imports when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.pipeline import MalwarePipeline

def run_feedback_loop():
    print("Initializing Pipeline...")
    pipeline = MalwarePipeline()
    pipeline.load_and_split_data("data/generated/synthetic_corpus_9000.csv")
    
    print("Training Tier 2...")
    pipeline.train_tier2()
    
    print("Training Tier 1 (Seed only)...")
    pipeline.train_tier1()
    
    acc_0, esc_0 = pipeline.tier1_evaluate()
    print(f"\n--- Cycle 0 (Baseline) ---")
    print(f"Accuracy:        {acc_0 * 100:.2f}%")
    print(f"Escalation Rate: {esc_0 * 100:.2f}%")
    
    X_extra = pd.DataFrame()
    y_extra = pd.Series(dtype=int)
    
    pool = pipeline.feedback_pool.copy()
    
    batch_size = len(pool) // 6
    
    for cycle in range(1, 7):
        start_idx = (cycle - 1) * batch_size
        end_idx = start_idx + batch_size if cycle < 6 else len(pool)
        batch = pool.iloc[start_idx:end_idx]
        
        X_batch_fast = batch[pipeline.fast_features]
        X_batch_all = batch[pipeline.fast_features + pipeline.deep_features]
        
        # Tier 1 Prediction
        probs = pipeline.tier1_rf.predict_proba(X_batch_fast)
        max_probs = np.max(probs, axis=1)
        
        # Identify escalated samples (0.30 <= prob <= 0.70)
        escalate_mask = (max_probs >= 0.30) & (max_probs <= 0.70)
        
        if np.any(escalate_mask):
            X_escalated_all = X_batch_all[escalate_mask]
            X_escalated_fast = X_batch_fast[escalate_mask]
            
            # Tier 2 predicts the definitive label
            tier2_preds = pipeline.tier2_rf.predict(X_escalated_all)
            
            # Add to retraining set
            X_extra = pd.concat([X_extra, X_escalated_fast], ignore_index=True)
            y_extra = pd.concat([y_extra, pd.Series(tier2_preds)], ignore_index=True)
            
            # Retrain Tier 1
            pipeline.train_tier1(X_extra, y_extra)
            
        print(f"Cycle {cycle} complete. Escalated {np.sum(escalate_mask)} samples.")
        
    acc_6, esc_6 = pipeline.tier1_evaluate()
    print(f"\n--- Cycle 6 (Final) ---")
    print(f"Accuracy:        {acc_6 * 100:.2f}%")
    print(f"Escalation Rate: {esc_6 * 100:.2f}%")

if __name__ == "__main__":
    run_feedback_loop()
