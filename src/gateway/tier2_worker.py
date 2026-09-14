import asyncio
import sqlite3
import time
from src.features.deep_features import extract_deep_features
from src.models.tier2 import Tier2Model
from src.feedback_loop import algorithm2_feedback_loop

# Using a mocked/simulated Tier 2 model in the worker for the MVP
# In a real deployed version, this loads the weights trained in Phase 2
t2_model = Tier2Model()
# Fit with dummy data so predict() doesn't throw NotFittedError
t2_model.fit([[0,0,0,0,0], [7.0,1,10,5,1]], [0, 1])

def log_tier2_verdict(path, verdict, latency_ms):
    conn = sqlite3.connect('gateway_logs.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS tier2_logs (timestamp TEXT, path TEXT, verdict TEXT, latency_ms REAL)")
    c.execute("INSERT INTO tier2_logs VALUES (datetime('now'), ?, ?, ?)",
              (path, verdict, latency_ms))
    conn.commit()
    conn.close()

async def tier2_worker_loop(escalation_queue: asyncio.Queue, feedback_threshold=10):
    """
    Background worker consuming escalated requests.
    Runs Deep Feature extraction and Tier 2 classification.
    """
    escalated_history = []
    
    while True:
        req_dict = await escalation_queue.get()
        start_time = time.time()
        
        # Simulated payload for deep features extraction
        # If real malware binary was uploaded, this is where it's parsed
        payload = req_dict.get('body', b"")
        if isinstance(payload, str):
            payload = payload.encode('utf-8', errors='ignore')
            
        is_simulated = True if req_dict['path'] == '/upload' else False
        features = extract_deep_features(payload, is_simulated=is_simulated)
        
        # Prediction
        pred = t2_model.predict([features])[0]
        verdict = "MALICIOUS" if pred == 1 else "BENIGN"
        
        latency = (time.time() - start_time) * 1000
        log_tier2_verdict(req_dict['path'], verdict, latency)
        
        # Store for feedback loop
        escalated_history.append({"features": features, "tier2_verdict": pred})
        
        # Algorithm 2 Feedback Loop Trigger
        if len(escalated_history) >= feedback_threshold:
            import queue
            fb_queue = queue.Queue()
            for h in escalated_history:
                fb_queue.put_nowait(h)
            
            # This would run asynchronously or in another worker
            algorithm2_feedback_loop(None, fb_queue) # tier1_model mocked as None here
            escalated_history = []
            
        escalation_queue.task_done()
