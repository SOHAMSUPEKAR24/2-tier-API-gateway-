import os
import json
import csv
from datetime import datetime

def parse_local_csic2010():
    dataset_path = '/Users/sohamsupekar/Desktop/MajorProject/DATASET/csic_database.csv'
    out_path = 'data/generated/csic2010_parsed.jsonl'
    
    print(f"Parsing real CSIC 2010 dataset from {dataset_path}...")
    os.makedirs('data/generated', exist_ok=True)
    
    parsed_count = 0
    with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f_in, open(out_path, 'w') as f_out:
        reader = csv.reader(f_in)
        header = next(reader)
        # Expected columns:
        # classification(0), Method(1), User-Agent(2), Pragma, Cache-Control, Accept, Accept-encoding, Accept-charset, language, host, cookie, content-type, connection, lenght, content(14), URL(16)
        
        for row in reader:
            if len(row) < 17:
                continue
                
            label = "attack" if row[0] == "Anomalous" else "benign"
            method = row[1]
            user_agent = row[2]
            body = row[14]
            url = row[16] # e.g., http://localhost:8080/tienda1/index.jsp HTTP/1.1
            
            # Extract path from URL
            path = url.split(' ')[0]
            if "localhost:8080" in path:
                path = path.split("localhost:8080")[-1]
                
            req = {
                "request": {
                    "method": method,
                    "path": path,
                    "headers": {"User-Agent": user_agent},
                    "body": body
                },
                "label": label,
                "subtype": "none",
                "source_tool": "csic2010",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            f_out.write(json.dumps(req) + "\n")
            parsed_count += 1
            if parsed_count >= 10000: # Limit to 10k to keep processing fast for MVP
                break
                
    print(f"Parsed {parsed_count} real CSIC 2010 records into {out_path}")

if __name__ == "__main__":
    parse_local_csic2010()
