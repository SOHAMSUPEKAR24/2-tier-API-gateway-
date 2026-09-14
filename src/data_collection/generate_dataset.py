import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=9000, output_path="data/generated/synthetic_corpus_9000.csv"):
    print(f"Generating {num_samples} rows of synthetic dataset...")
    np.random.seed(42)
    
    # Define classes: benign (0), sqli (1), xss (2), malware_upload (3), rce_injection (4)
    # Binary label: 0 for benign, 1 for anything else (attack)
    classes = ['benign', 'sqli', 'xss', 'malware_upload', 'rce_injection']
    
    # Generate random class assignments evenly
    class_indices = np.random.choice(len(classes), size=num_samples, p=[0.5, 0.125, 0.125, 0.125, 0.125])
    class_labels = [classes[i] for i in class_indices]
    binary_labels = [0 if i == 0 else 1 for i in class_indices]
    
    # 9 Fast Features + 5 Deep Features = 14 features
    
    # Initialize arrays
    payload_size_bytes = np.zeros(num_samples)
    payload_entropy = np.zeros(num_samples)
    special_char_count = np.zeros(num_samples)
    sql_keyword_count = np.zeros(num_samples)
    script_tag_count = np.zeros(num_samples)
    base64_blob_present = np.zeros(num_samples)
    param_count = np.zeros(num_samples)
    content_type_mismatch = np.zeros(num_samples)
    request_rate_per_key = np.zeros(num_samples)
    
    decoded_blob_entropy = np.zeros(num_samples)
    has_exe_magic_bytes = np.zeros(num_samples)
    sim_registry_writes = np.zeros(num_samples)
    sim_network_beacons = np.zeros(num_samples)
    sim_process_injections = np.zeros(num_samples)
    
    # Populate with overlapping distributions based on class
    for i, c in enumerate(class_indices):
        if c == 0: # benign
            payload_size_bytes[i] = np.random.normal(500, 200)
            payload_entropy[i] = np.random.normal(3.0, 0.5)
            special_char_count[i] = np.random.poisson(2)
            sql_keyword_count[i] = np.random.poisson(0.1)
            script_tag_count[i] = np.random.poisson(0)
            base64_blob_present[i] = np.random.choice([0, 1], p=[0.95, 0.05])
            param_count[i] = np.random.poisson(2)
            content_type_mismatch[i] = np.random.choice([0, 1], p=[0.99, 0.01])
            request_rate_per_key[i] = np.random.normal(5, 2)
            
            decoded_blob_entropy[i] = np.random.normal(2.0, 1.0)
            has_exe_magic_bytes[i] = 0
            sim_registry_writes[i] = np.random.poisson(0.5)
            sim_network_beacons[i] = np.random.poisson(0.1)
            sim_process_injections[i] = 0
            
        elif c == 1: # sqli
            payload_size_bytes[i] = np.random.normal(600, 250)
            payload_entropy[i] = np.random.normal(3.5, 0.6)
            special_char_count[i] = np.random.poisson(15)
            sql_keyword_count[i] = np.random.poisson(3)
            script_tag_count[i] = 0
            base64_blob_present[i] = np.random.choice([0, 1], p=[0.9, 0.1])
            param_count[i] = np.random.poisson(4)
            content_type_mismatch[i] = np.random.choice([0, 1], p=[0.8, 0.2])
            request_rate_per_key[i] = np.random.normal(15, 5)
            
            decoded_blob_entropy[i] = np.random.normal(2.0, 1.0)
            has_exe_magic_bytes[i] = 0
            sim_registry_writes[i] = 0
            sim_network_beacons[i] = 0
            sim_process_injections[i] = 0
            
        elif c == 2: # xss
            payload_size_bytes[i] = np.random.normal(800, 300)
            payload_entropy[i] = np.random.normal(4.0, 0.5)
            special_char_count[i] = np.random.poisson(12)
            sql_keyword_count[i] = 0
            script_tag_count[i] = np.random.poisson(2)
            base64_blob_present[i] = np.random.choice([0, 1], p=[0.85, 0.15])
            param_count[i] = np.random.poisson(3)
            content_type_mismatch[i] = np.random.choice([0, 1], p=[0.8, 0.2])
            request_rate_per_key[i] = np.random.normal(10, 3)
            
            decoded_blob_entropy[i] = np.random.normal(2.0, 1.0)
            has_exe_magic_bytes[i] = 0
            sim_registry_writes[i] = 0
            sim_network_beacons[i] = 0
            sim_process_injections[i] = 0
            
        elif c == 3: # malware_upload
            payload_size_bytes[i] = np.random.normal(50000, 20000)
            payload_entropy[i] = np.random.normal(6.5, 0.8)
            special_char_count[i] = np.random.poisson(50)
            sql_keyword_count[i] = 0
            script_tag_count[i] = 0
            base64_blob_present[i] = np.random.choice([0, 1], p=[0.2, 0.8])
            param_count[i] = np.random.poisson(1)
            content_type_mismatch[i] = np.random.choice([0, 1], p=[0.4, 0.6])
            request_rate_per_key[i] = np.random.normal(5, 2)
            
            decoded_blob_entropy[i] = np.random.normal(7.2, 0.5)
            has_exe_magic_bytes[i] = np.random.choice([0, 1], p=[0.1, 0.9])
            sim_registry_writes[i] = np.random.poisson(15)
            sim_network_beacons[i] = np.random.poisson(3)
            sim_process_injections[i] = np.random.poisson(2)
            
        elif c == 4: # rce_injection
            payload_size_bytes[i] = np.random.normal(1000, 400)
            payload_entropy[i] = np.random.normal(4.5, 0.7)
            special_char_count[i] = np.random.poisson(20)
            sql_keyword_count[i] = 0
            script_tag_count[i] = 0
            base64_blob_present[i] = np.random.choice([0, 1], p=[0.5, 0.5])
            param_count[i] = np.random.poisson(5)
            content_type_mismatch[i] = np.random.choice([0, 1], p=[0.7, 0.3])
            request_rate_per_key[i] = np.random.normal(20, 10)
            
            decoded_blob_entropy[i] = np.random.normal(3.0, 1.0)
            has_exe_magic_bytes[i] = 0
            sim_registry_writes[i] = np.random.poisson(2)
            sim_network_beacons[i] = np.random.poisson(1)
            sim_process_injections[i] = np.random.poisson(0.5)
            
    # Clip absolute values for realistic ranges
    payload_size_bytes = np.clip(payload_size_bytes, 0, None)
    request_rate_per_key = np.clip(request_rate_per_key, 0, None)
    
    df = pd.DataFrame({
        'class_label': class_labels,
        'binary_label': binary_labels,
        'payload_size_bytes': payload_size_bytes,
        'payload_entropy': payload_entropy,
        'special_char_count': special_char_count,
        'sql_keyword_count': sql_keyword_count,
        'script_tag_count': script_tag_count,
        'base64_blob_present': base64_blob_present,
        'param_count': param_count,
        'content_type_mismatch': content_type_mismatch,
        'request_rate_per_key': request_rate_per_key,
        'decoded_blob_entropy': decoded_blob_entropy,
        'has_exe_magic_bytes': has_exe_magic_bytes,
        'sim_registry_writes': sim_registry_writes,
        'sim_network_beacons': sim_network_beacons,
        'sim_process_injections': sim_process_injections
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
