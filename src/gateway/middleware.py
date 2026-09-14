import time
import sqlite3
import json
import asyncio
from src.features.fast_features import extract_fast_features
from src.models.tier1_rf import Tier1RF
from src.models.tier1_edl import Tier1EDL
from fastapi import Request, HTTPException, BackgroundTasks

# Simulated global models
tier1_rf = Tier1RF()
tier1_edl = Tier1EDL(input_dim=9)

# SQLite setup
def init_db():
    conn = sqlite3.connect('gateway_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS access_logs
                 (timestamp TEXT, method TEXT, path TEXT, latency_ms REAL, decision TEXT, escalated INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# Simple asyncio queue for MVP
escalation_queue = asyncio.Queue()

def log_request(method, path, latency_ms, decision, escalated):
    conn = sqlite3.connect('gateway_logs.db')
    c = conn.cursor()
    c.execute("INSERT INTO access_logs VALUES (datetime('now'), ?, ?, ?, ?, ?)",
              (method, path, latency_ms, decision, escalated))
    conn.commit()
    conn.close()

async def screen_request(request: Request, background_tasks: BackgroundTasks):
    """
    Algorithm 1: Tier 1 Inline Screening + Escalation Logic
    """
    start_time = time.time()
    
    # Extract request properties
    method = request.method
    path = request.url.path
    headers = dict(request.headers)
    body = ""
    try:
        body = (await request.body()).decode()
    except:
        pass
        
    req_dict = {
        "method": method,
        "path": path,
        "headers": headers,
        "body": body
    }
    
    features = extract_fast_features(req_dict)
    
    # Normally we would use EDL for uncertainty. 
    # For this robust prototype, we mock the decision logic slightly 
    # to guarantee we see some escalations, passes, and blocks.
    
    # Mocking prediction for demonstration (if models aren't fully trained on real data yet)
    # We will just use the features to make a deterministic mock decision to show the pipeline works
    score = features[3] + features[4] # sql_count + script_count
    
    decision = "FORWARD"
    escalated = 0
    
    if score > 0:
        decision = "BLOCK"
    elif "upload" in path.lower() or features[0] > 1000: # large payload
        decision = "ESCALATE"
        escalated = 1
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Log to SQLite
    background_tasks.add_task(log_request, method, path, latency_ms, decision, escalated)
    
    if escalated:
        # Enqueue for Tier 2 worker
        await escalation_queue.put(req_dict)
        return {"status": "Escalated for deep analysis"}
    
    if decision == "BLOCK":
        raise HTTPException(status_code=403, detail="Blocked by Tier 1 Malware Detection")
        
    return {"status": "Forwarded to upstream"}
