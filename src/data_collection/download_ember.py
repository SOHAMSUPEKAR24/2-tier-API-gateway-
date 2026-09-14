import os
import json
import random

def mock_ember_features():
    print("Mocking EMBER dataset feature extraction (real EMBER dataset is >100GB)...")
    os.makedirs('data/generated', exist_ok=True)
    
    # We create a jsonl representing the 5 deep features for 1000 binaries
    # 5 features: decoded_blob_entropy, has_exe_magic_bytes, sim_registry_writes, sim_network_beacons, sim_process_injections
    
    with open('data/generated/ember_features.jsonl', 'w') as f:
        # Generate 500 benign
        for _ in range(500):
            feats = {
                "decoded_blob_entropy": random.uniform(2.0, 5.5),
                "has_exe_magic_bytes": 1 if random.random() < 0.2 else 0, # Sometimes benign has PE header
                "sim_registry_writes": random.randint(0, 5),
                "sim_network_beacons": 0,
                "sim_process_injections": 0,
                "label": 0
            }
            f.write(json.dumps(feats) + "\n")
            
        # Generate 500 malicious
        for _ in range(500):
            feats = {
                "decoded_blob_entropy": random.uniform(6.5, 7.9), # High entropy (packed)
                "has_exe_magic_bytes": 1,
                "sim_registry_writes": random.randint(10, 50),
                "sim_network_beacons": random.randint(1, 10),
                "sim_process_injections": random.randint(1, 5),
                "label": 1
            }
            f.write(json.dumps(feats) + "\n")
            
    print("EMBER mocked features generated at data/generated/ember_features.jsonl")

if __name__ == "__main__":
    mock_ember_features()
