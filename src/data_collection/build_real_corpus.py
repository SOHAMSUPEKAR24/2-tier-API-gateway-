import pandas as pd
import json
import os

def build_tier1_corpus():
    cic_path = '/Users/sohamsupekar/Desktop/MajorProject/DATASET/cic.csv'
    out_path = 'data/generated/tier1_real_corpus.csv'
    
    print("Building Tier 1 Corpus from CIC-IDS...")
    
    benign_list = []
    attack_list = []
    
    benign_count = 0
    attack_count = 0
    
    # Read first chunk to find exact label column name
    df_head = pd.read_csv(cic_path, nrows=1)
    label_col = [c for c in df_head.columns if 'label' in c.lower()][0]
    
    chunk_size = 50000
    for chunk in pd.read_csv(cic_path, chunksize=chunk_size):
        if benign_count < 5000:
            benign_chunk = chunk[chunk[label_col] == 'Benign']
            needed = 5000 - benign_count
            benign_chunk = benign_chunk.head(needed)
            benign_list.append(benign_chunk)
            benign_count += len(benign_chunk)
            
        if attack_count < 5000:
            attack_chunk = chunk[chunk[label_col] != 'Benign']
            needed = 5000 - attack_count
            attack_chunk = attack_chunk.head(needed)
            attack_list.append(attack_chunk)
            attack_count += len(attack_chunk)
            
        if benign_count >= 5000 and attack_count >= 5000:
            break
            
    df_benign = pd.concat(benign_list)
    df_attack = pd.concat(attack_list)
    
    df_final = pd.concat([df_benign, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs('data/generated', exist_ok=True)
    df_final.to_csv(out_path, index=False)
    
    print(f"Tier 1 Corpus Shape: {df_final.shape}")
    print("Tier 1 Label Distribution:")
    print(df_final[label_col].value_counts())

def build_tier2_corpus():
    import glob
    ember_dir = '/Users/sohamsupekar/Desktop/MajorProject/DATASET/ember2018'
    out_path = 'data/generated/tier2_real_corpus.jsonl'
    
    print("\nBuilding Tier 2 Corpus from EMBER...")
    
    benign_count = 0
    malicious_count = 0
    
    results = []
    
    ember_files = sorted(glob.glob(os.path.join(ember_dir, 'train_features_*.jsonl')))
    
    for ember_path in ember_files:
        if benign_count >= 5000 and malicious_count >= 5000:
            break
        print(f"Reading {ember_path}...")
        with open(ember_path, 'r') as f:
            for line in f:
                if benign_count >= 5000 and malicious_count >= 5000:
                    break
                    
                try:
                    data = json.loads(line)
                    label = data.get('label', -1)
                    
                    # Keep only needed keys
                    extracted = {
                        'label': label,
                        'byteentropy': data.get('byteentropy', []),
                        'general': data.get('general', {}),
                        'strings': data.get('strings', {})
                    }
                    
                    if label == 0 and benign_count < 5000:
                        results.append(extracted)
                        benign_count += 1
                    elif label == 1 and malicious_count < 5000:
                        results.append(extracted)
                        malicious_count += 1
                except Exception:
                    continue
                
    import random
    random.seed(42)
    random.shuffle(results)
    
    with open(out_path, 'w') as out_f:
        for r in results:
            out_f.write(json.dumps(r) + "\n")
            
    print(f"Tier 2 Corpus Total Rows: {len(results)}")
    
    b_c = sum(1 for r in results if r['label'] == 0)
    m_c = sum(1 for r in results if r['label'] == 1)
    print("Tier 2 Label Distribution:")
    print(f"Benign (0): {b_c}")
    print(f"Malicious (1): {m_c}")

if __name__ == "__main__":
    build_tier1_corpus()
    build_tier2_corpus()
