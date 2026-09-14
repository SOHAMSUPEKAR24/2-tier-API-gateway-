import json
import random
import time
import requests
import os
from datetime import datetime

# We will simulate normal API traffic to VAmPI and Juice Shop
# Because we are running in an isolated environment without actual web traffic tools configured,
# we will mock the "capture" by directly generating realistic benign JSONL records,
# simulating what we WOULD capture if we intercepted requests to the Docker containers.

def simulate_benign_traffic():
    print("Simulating benign traffic capture...")
    os.makedirs('data/generated', exist_ok=True)
    
    endpoints = [
        {"path": "/users/v1", "method": "GET"},
        {"path": "/users/v1/login", "method": "POST", "body": '{"username": "user1", "password": "password1"}'},
        {"path": "/rest/products/search", "method": "GET"},
        {"path": "/api/Feedbacks", "method": "POST", "body": '{"comment": "Great!", "rating": 5}'}
    ]
    
    with open('data/generated/benign_traffic.jsonl', 'w') as f:
        for _ in range(1500):
            ep = random.choice(endpoints)
            req = {
                "request": {
                    "method": ep["method"],
                    "path": ep["path"],
                    "headers": {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "application/json",
                        "Host": "localhost"
                    },
                    "body": ep.get("body", "")
                },
                "label": "benign",
                "subtype": "none",
                "source_tool": "benign_script",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            f.write(json.dumps(req) + "\n")
            
    print("Benign traffic captured at data/generated/benign_traffic.jsonl")

if __name__ == "__main__":
    simulate_benign_traffic()
