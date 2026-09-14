from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

class Tier1RF:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
