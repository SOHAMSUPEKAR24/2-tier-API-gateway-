import json
import random
import time
import os
from datetime import datetime

# Similarly, we mock the capture from sqlmap, ZAP, and ffuf.
# In a real environment, this script would parse proxy logs (like mitmproxy) 
# where these tools were routed through.

def simulate_attack_traffic():
    print("Simulating attack traffic capture...")
    os.makedirs('data/generated', exist_ok=True)
    
    attacks = [
        # SQLi (sqlmap)
        {"path": "/users/v1?id=1' OR 1=1--", "method": "GET", "subtype": "SQLi", "tool": "sqlmap"},
        {"path": "/users/v1/login", "method": "POST", "body": '{"username": "admin\' --", "password": ""}', "subtype": "SQLi", "tool": "sqlmap"},
        # XSS (ZAP)
        {"path": "/search?q=<script>alert(1)</script>", "method": "GET", "subtype": "XSS", "tool": "ZAP"},
        {"path": "/api/Feedbacks", "method": "POST", "body": '{"comment": "<img src=x onerror=alert(1)>", "rating": 5}', "subtype": "XSS", "tool": "ZAP"},
        # Fuzzing/RCE (ffuf)
        {"path": "/api/v1/ping?ip=127.0.0.1;cat /etc/passwd", "method": "GET", "subtype": "RCE", "tool": "ffuf"},
        {"path": "/../../../etc/passwd", "method": "GET", "subtype": "PathTraversal", "tool": "ffuf"}
    ]
    
    with open('data/generated/attack_traffic.jsonl', 'w') as f:
        for _ in range(1500):
            att = random.choice(attacks)
            req = {
                "request": {
                    "method": att["method"],
                    "path": att["path"],
                    "headers": {
                        "User-Agent": f"Mozilla/5.0 ({att['tool']})",
                        "Accept": "*/*",
                        "Host": "localhost"
                    },
                    "body": att.get("body", "")
                },
                "label": "attack",
                "subtype": att["subtype"],
                "source_tool": att["tool"],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            f.write(json.dumps(req) + "\n")
            
    print("Attack traffic captured at data/generated/attack_traffic.jsonl")

if __name__ == "__main__":
    simulate_attack_traffic()
