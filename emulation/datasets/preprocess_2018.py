import pandas as pd
import os
import numpy as np
import argparse

DEFAULT_INPUT_FILE = "cic-ids-2018-deduplicated.csv"
DEFAULT_OUTPUT_FEATURE = "cic-ids-2018_X.npy"
DEFAULT_OUTPUT_LABEL = "cic-ids-2018_y.npy"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE, help="Path to input CSV file")
    parser.add_argument("--output_feature_file", default=DEFAULT_OUTPUT_FEATURE, help="Path to output feature .npy file")
    parser.add_argument("--output_label_file", default=DEFAULT_OUTPUT_LABEL, help="Path to output label .npy file")
    args = parser.parse_args()

    file_list = [args.input_file]
    
    attack_list = [
        'Bot', 
        'DDOS attack-HOIC', 
        'DDoS attacks-LOIC-HTTP', 
        'SSH-Bruteforce', 
        'Infilteration', 
        'DoS Hulk'
    ]
    all_labels = ['BENIGN'] + attack_list
    
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

    dataset_X = []
    dataset_y = []

    CUSTOM_LABEL_COL = 'Label'
    for csv_file in file_list:
        if os.path.isabs(csv_file):
            file_path = csv_file
        else:
             if os.path.exists(csv_file):
                 file_path = csv_file
             else:
                 file_path = os.path.join('emulation/datasets', csv_file)

        print(f"Reading {file_path}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        df = df[df['Label'].isin(all_labels)]

        data = df[CUSTOM_FEAT_COLS].to_numpy()
        label = df[CUSTOM_LABEL_COL].to_numpy()
        dataset_X.append(data)
        dataset_y.append(label)
        
    data_X = np.concatenate(dataset_X)
    data_y = np.concatenate(dataset_y)

    a, b = np.unique(data_y,return_counts=True)
    print("Label counts:", dict(zip(a,b)))

    # Prepare label mappings
    label_to_index_bin = {label: 0 if label == 'BENIGN' else 1 for label in all_labels}
    label_to_index_multi = {label: i for i, label in enumerate(all_labels)}

    print("Generating both binary and multi-class labels...")
    data_y_bin = np.array([label_to_index_bin[label] for label in data_y])
    data_y_multi = np.array([label_to_index_multi[label] for label in data_y])
    
    print("Binary mapping: {'BENIGN': 0, 'Attacks': 1}")
    print("Multi-class mapping:", label_to_index_multi)
    
    # Save features
    feat_path = os.path.join('emulation/datasets', args.output_feature_file) if not os.path.isabs(args.output_feature_file) else args.output_feature_file
    np.save(feat_path, data_X)
    
    # Save labels
    base_label_path = args.output_label_file
    if not os.path.isabs(base_label_path):
        base_label_path = os.path.join('emulation/datasets', base_label_path)
    
    multi_path = base_label_path.replace('.npy', '_multi.npy')
    
    np.save(base_label_path, data_y_bin)
    np.save(multi_path, data_y_multi)
    print(f"Saved binary labels to {base_label_path}")
    print(f"Saved multi-class labels to {multi_path}")
