import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class MalwarePipeline:
    def __init__(self):
        self.tier1_rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42)
        self.tier2_rf = RandomForestClassifier(n_estimators=250, random_state=42)
        
        self.fast_features = [
            'payload_size_bytes', 'payload_entropy', 'special_char_count',
            'sql_keyword_count', 'script_tag_count', 'base64_blob_present',
            'param_count', 'content_type_mismatch', 'request_rate_per_key'
        ]
        
        self.deep_features = [
            'decoded_blob_entropy', 'has_exe_magic_bytes', 
            'sim_registry_writes', 'sim_network_beacons', 'sim_process_injections'
        ]
        
    def load_and_split_data(self, filepath="data/generated/synthetic_corpus_9000.csv"):
        df = pd.read_csv(filepath)
        
        # 1. Test set: 1800 (20%)
        # 2. Tier-2 train: 2700 (30%)
        # 3. Tier-1 seed: 270 (3%)
        # 4. Feedback pool: 4230 (47%)
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.test_set = df_shuffled.iloc[:1800]
        self.tier2_train = df_shuffled.iloc[1800:4500]
        self.tier1_seed = df_shuffled.iloc[4500:4770]
        self.feedback_pool = df_shuffled.iloc[4770:]
        
    def train_tier2(self):
        X_t2 = self.tier2_train[self.fast_features + self.deep_features]
        y_t2 = self.tier2_train['binary_label']
        self.tier2_rf.fit(X_t2, y_t2)
        
    def train_tier1(self, X_extra=None, y_extra=None):
        X_t1 = self.tier1_seed[self.fast_features].copy()
        y_t1 = self.tier1_seed['binary_label'].copy()
        
        if X_extra is not None and y_extra is not None:
            X_t1 = pd.concat([X_t1, X_extra], ignore_index=True)
            y_t1 = pd.concat([y_t1, y_extra], ignore_index=True)
            
        self.tier1_rf.fit(X_t1, y_t1)
        
    def tier1_evaluate(self):
        X_test = self.test_set[self.fast_features]
        y_test = self.test_set['binary_label']
        
        preds = self.tier1_rf.predict(X_test)
        probs = self.tier1_rf.predict_proba(X_test)
        
        accuracy = np.mean(preds == y_test)
        
        # Escalation Policy
        # forward < 0.30, block > 0.70, escalate between 0.30 and 0.70
        max_probs = np.max(probs, axis=1)
        escalated = np.sum((max_probs >= 0.30) & (max_probs <= 0.70))
        escalation_rate = escalated / len(X_test)
        
        return accuracy, escalation_rate
