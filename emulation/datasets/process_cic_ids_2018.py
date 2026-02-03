import pandas as pd
import os
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

"""
This script performs the following tasks:

1. Read and filter the 2018 dataset: Extracts corresponding attack categories from specified CSV files. Note: The Infiltration category is distributed across 02-28-2018.csv and 03-01-2018.csv; the script automatically handles and merges them, totaling 161,934 records as provided.
2. Feature mapping and extraction: Renames columns according to mappings.txt and extracts features defined in CUSTOM_FEAT_COLS.
3. Generate cic-ids-2018-integrated.csv: Contains all extracted 2018 data (including DoS attacks-Hulk).
4. Process Wednesday data: Extracts BENIGN and DoS Hulk from Wednesday-workingHours.pcap_ISCX.csv.
5. Generate cic-ids-2018-pre.csv: Merges Wednesday data and 2018 data, but excludes the DoS attacks-Hulk category from the 2018 data (retaining DoS Hulk from Wednesday).
"""

# Configuration
DATASET_2018_DIR = "/home/laevon/.cache/kagglehub/datasets/solarmainframe/ids-intrusion-csv/versions/1"
WEDNESDAY_FILE = "Wednesday-workingHours.pcap_ISCX.csv" # Assumed to be in current dir or emulation/datasets
OUTPUT_INTEGRATED = "cic-ids-2018-sintegrated.csv"
OUTPUT_PRE = "cic-ids-2018-spre.csv"

# Mappings and Features
rename_mapping = {
    'Dst Port': 'Destination Port',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'Pkt Len Min': 'Min Packet Length',
    'Pkt Len Max': 'Max Packet Length',
    'Pkt Len Mean': 'Packet Length Mean',
    'SYN Flag Cnt': 'SYN Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'FIN Flag Cnt': 'FIN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count'
}

CUSTOM_FEAT_COLS = [
    "Flow IAT Min",
    "Flow IAT Max",
    "Flow IAT Mean",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Total Length of Fwd Packets",
    "Total Fwd Packets",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "ECE Flag Count",
    "Flow Duration",
    "Destination Port",
]

# 2018 Dataset Configuration
# File -> List of Labels to extract
files_2018 = {
    "03-02-2018.csv": ["Bot"],
    "02-21-2018.csv": ["DDOS attack-HOIC"],
    "02-20-2018.csv": ["DDoS attacks-LOIC-HTTP"],
    "02-16-2018.csv": ["DoS attacks-Hulk", "DoS attacks-SlowHTTPTest"],
    "02-14-2018.csv": ["FTP-BruteForce", "SSH-Bruteforce"],
    "02-28-2018.csv": ["Infilteration"],
    "03-01-2018.csv": ["Infilteration"]
}

def process_2018_data():
    print("Processing 2018 Dataset...")
    integrated_dfs = []
    
    for filename, labels in files_2018.items():
        file_path = os.path.join(DATASET_2018_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        print(f"Reading {filename}...")
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Filter by label
        df_filtered = df[df['Label'].isin(labels)].copy()
        print(f"  Extracted {len(df_filtered)} rows for labels {labels}")
        
        # Rename columns
        df_filtered.rename(columns=rename_mapping, inplace=True)
        
        # Ensure all required columns exist
        missing_cols = [col for col in CUSTOM_FEAT_COLS if col not in df_filtered.columns]
        if missing_cols:
            print(f"  Error: Missing columns in {filename}: {missing_cols}")
            # Handle missing columns if necessary, or skip
            continue
            
        # Select features + Label
        cols_to_keep = CUSTOM_FEAT_COLS + ['Label']
        df_final = df_filtered[cols_to_keep]
        
        integrated_dfs.append(df_final)
        
    if not integrated_dfs:
        print("No data extracted from 2018 dataset.")
        return None

    integrated_df = pd.concat(integrated_dfs, ignore_index=True)
    print(f"Saving integrated 2018 data to {OUTPUT_INTEGRATED} ({len(integrated_df)} rows)...")
    integrated_df.to_csv(OUTPUT_INTEGRATED, index=False)
    return integrated_df

def process_wednesday_data():
    print("Processing Wednesday Dataset...")
    # Locate file
    if os.path.exists(WEDNESDAY_FILE):
        file_path = WEDNESDAY_FILE
    elif os.path.exists(os.path.join('emulation/datasets', WEDNESDAY_FILE)):
        file_path = os.path.join('emulation/datasets', WEDNESDAY_FILE)
    else:
        print(f"Error: {WEDNESDAY_FILE} not found.")
        return None

    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    target_labels = ['BENIGN', 'DoS Hulk']
    df_filtered = df[df['Label'].isin(target_labels)].copy()
    print(f"  Extracted {len(df_filtered)} rows for labels {target_labels}")
    
    # Check if renaming is needed for Wednesday data? 
    # preprocess.py didn't rename, so assuming columns are already correct or match CUSTOM_FEAT_COLS
    # But let's verify if mapping is needed. preprocess.py uses CUSTOM_FEAT_COLS directly.
    # If Wednesday file has 'Dst Port' instead of 'Destination Port', preprocess.py would fail unless it was already correct.
    # However, preprocess.py in the attachment DOES NOT have renaming logic. 
    # It assumes CUSTOM_FEAT_COLS exist.
    # Let's assume Wednesday file has correct columns or we need to apply mapping too?
    # But for Wednesday file, preprocess.py works on it. 
    # Let's check if Wednesday file needs mapping. 
    # If preprocess.py works on it, and preprocess.py uses "Destination Port", then Wednesday file must have "Destination Port".
    # If Wednesday file has "Dst Port", preprocess.py would fail.
    # I will assume Wednesday file is fine as is, or apply mapping if columns are missing.
    
    # Let's try to apply mapping just in case, it won't hurt if keys don't exist.
    df_filtered.rename(columns=rename_mapping, inplace=True)

    missing_cols = [col for col in CUSTOM_FEAT_COLS if col not in df_filtered.columns]
    if missing_cols:
        print(f"  Error: Missing columns in Wednesday file: {missing_cols}")
        return None

    cols_to_keep = CUSTOM_FEAT_COLS + ['Label']
    return df_filtered[cols_to_keep]

def main():
    # Task 1 & 2: Process 2018 data
    df_2018 = process_2018_data()
    if df_2018 is None:
        return

    # Task 3: Process Wednesday data
    df_wed = process_wednesday_data()
    if df_wed is None:
        return

    # Task 4: Merge for pre.csv
    print("Creating final merged dataset...")
    
    # Filter out 'DoS attacks-Hulk' from 2018 data for the pre file
    # Note: The user said "DoS attacks-Hulk" (plural) for 2018, and "DoS Hulk" (singular) for Wednesday.
    # We keep Wednesday's "DoS Hulk". We remove 2018's "DoS attacks-Hulk".
    
    df_2018_filtered = df_2018[df_2018['Label'] != 'DoS attacks-Hulk']
    print(f"  Filtered 2018 data: {len(df_2018)} -> {len(df_2018_filtered)} (Removed DoS attacks-Hulk)")
    
    final_df = pd.concat([df_2018_filtered, df_wed], ignore_index=True)

    # ================= Normalization Logic =================
    print("Applying StandardScaler (Normalization)...")
    
    # # 1. Extract feature columns
    # feat_data = final_df[CUSTOM_FEAT_COLS].values
    
    # # 2. Fit scaler (calculate mean and variance)
    # scaler = StandardScaler()
    # feat_data_scaled = scaler.fit_transform(feat_data)
    
    # # 3. Write normalized data back to DataFrame
    # final_df[CUSTOM_FEAT_COLS] = feat_data_scaled
    
    # # 4. (Optional) Print mean and variance for confirmation
    # print(f"Feature Mean (approx 0): {np.mean(feat_data_scaled):.4f}")
    # print(f"Feature Std  (approx 1): {np.std(feat_data_scaled):.4f}")
    # ===================================================
    
    print(f"Saving final dataset to {OUTPUT_PRE} ({len(final_df)} rows)...")
    final_df.to_csv(OUTPUT_PRE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()