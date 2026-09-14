import pandas as pd
import numpy as np
import json

def adapt_tier1_features(df):
    """
    Map CIC-IDS 2017 features to the 9-dimensional Fast Feature space.
    """
    # Required columns
    cols = [
        'Fwd Packet Length Max', 
        'Fwd Packet Length Mean', 
        'Bwd Packet Length Max', 
        'Flow Bytes/s', 
        'Flow Packets/s', 
        'Fwd IAT Mean', 
        'Bwd IAT Mean', 
        'ACK Flag Count', 
        'PSH Flag Count'
    ]
    
    # Extract only these columns
    X = df[cols].copy()
    
    # Handle any potential infinite or NaN values that occur in CIC-IDS
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    return X.values

def adapt_tier2_features(ember_records):
    """
    Map EMBER static features to the 5-dimensional Deep Feature space.
    """
    X = []
    
    for data in ember_records:
        # 'decoded_blob_entropy' -> mean of EMBER's byteentropy array
        byte_entropy_arr = data.get('byteentropy', [])
        decoded_blob_entropy = np.mean(byte_entropy_arr) if byte_entropy_arr else 0.0
        
        # 'has_exe_magic_bytes' -> general.has_signature
        general = data.get('general', {})
        has_exe_magic_bytes = general.get('has_signature', 0)
        
        # 'sim_registry_writes' -> general.imports
        sim_registry_writes = general.get('imports', 0)
        
        # 'sim_network_beacons' -> strings.urls or general.urls (none exist in standard general, but we check anyway)
        strings = data.get('strings', {})
        sim_network_beacons = strings.get('urls', 0)
        
        # 'sim_process_injections' -> general.exports
        sim_process_injections = general.get('exports', 0)
        
        X.append([
            float(decoded_blob_entropy),
            float(has_exe_magic_bytes),
            float(sim_registry_writes),
            float(sim_network_beacons),
            float(sim_process_injections)
        ])
        
    return np.array(X)
