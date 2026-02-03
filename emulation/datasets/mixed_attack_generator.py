import pandas as pd
import numpy as np
import os
import argparse
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

def get_shawarma_home():
    """
    Resolves the home directory for the project.
    """
    env_home = os.getenv("Shawarma_HOME")
    if env_home:
        return env_home
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

SHAWARMA_HOME = get_shawarma_home()
DATA_DIR = os.path.join(SHAWARMA_HOME, 'emulation', 'datasets')

def resolve_path(path):
    """
    Resolves file paths relative to the DATA_DIR if they are not absolute.
    """
    if path and not os.path.isabs(path):
        return os.path.join(DATA_DIR, path)
    return path

def select_features_for_dataset(df, dataset_name):
    """
    Select specific features for known datasets.
    """
    CIC_IDS_2018_FEATURES = [
        "Flow IAT Min", "Flow IAT Max", "Flow IAT Mean",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        "Total Length of Fwd Packets", "Total Fwd Packets",
        "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
        "FIN Flag Count", "RST Flag Count", "ECE Flag Count",
        "Flow Duration", "Destination Port"
    ]
    
    if 'cic-ids-2018' in dataset_name.lower() or 'cic_ids_2018' in dataset_name.lower():
        target_features = CIC_IDS_2018_FEATURES
        dataset_type = 'CIC-IDS-2018'
    else:
        if all(f in df.columns for f in CIC_IDS_2018_FEATURES):
            target_features = CIC_IDS_2018_FEATURES
            dataset_type = 'CIC-IDS-2018 (auto-detected)'
        else:
            logging.info("No specific feature selection applied. Using all features.")
            return df
    
    missing_features = [f for f in target_features if f not in df.columns]
    if missing_features:
        logging.warning(f"Missing features for {dataset_type}: {missing_features}")
        logging.warning("Using all available features instead.")
        return df
    
    logging.info(f"Detected dataset type: {dataset_type}")
    cols_to_keep = target_features + ['Label']
    df_selected = df[cols_to_keep]
    return df_selected

def load_csv(path, benign_source=None):
    logging.info(f"Reading CSV from {path} ...")
    df = pd.read_csv(path)
    
    if 'Label' not in df.columns:
        for col in df.columns:
            if col.lower() == 'label':
                df.rename(columns={col: 'Label'}, inplace=True)
                break
    
    df['Label'] = df['Label'].astype(str).str.strip()
    
    if 'BENIGN' not in df['Label'].values and benign_source:
        logging.info(f"BENIGN not found in input. Loading from {benign_source}...")
        if os.path.exists(benign_source):
            df_benign = pd.read_csv(benign_source)
            df_benign.columns = df_benign.columns.str.strip()
            if 'Label' in df_benign.columns:
                df_benign['Label'] = df_benign['Label'].astype(str).str.strip()
                df = pd.concat([df, df_benign[df_benign['Label'] == 'BENIGN']])
            else:
                logging.warning(f"Label column not found in {benign_source}")
        else:
            logging.warning(f"Benign source {benign_source} not found.")
    
    dataset_name = os.path.basename(path)
    df = select_features_for_dataset(df, dataset_name)
    return df

def load_npy(data_path, label_path):
    logging.info(f"Reading NPY data from {data_path} and labels from {label_path} ...")
    X = np.load(data_path)
    y = np.load(label_path)
    if len(X) != len(y):
        raise ValueError(f"Mismatch between X ({len(X)}) and y ({len(y)})")
    return X, y

def redistribute_data(data, labels, args):
    """
    Core logic to redistribute data indices based on attack phases.
    MODIFIED: First half of phase is pure attack, second half is mixed (50/50 split of malicious part).
    """
    if args.seed is not None:
        np.random.seed(args.seed)
        
    unique_labels = np.unique(labels)
    label_indices = {}

    # 1. Group indices by label and shuffle
    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]
        if args.seed is not None:
            np.random.RandomState(args.seed).shuffle(indices)
        else:
            np.random.shuffle(indices)
        label_indices[lbl] = indices
        logging.info(f"  Label {lbl}: {len(indices)} samples")

    # 2. Identify Benign label
    benign_label = args.benign_label
    if benign_label is None:
        if 'BENIGN' in label_indices:
            benign_label = 'BENIGN'
        elif 0 in label_indices:
            benign_label = 0
        elif '0' in label_indices:
            benign_label = '0'
        else:
            benign_label = max(label_indices, key=lambda k: len(label_indices[k]))
    else:
        if isinstance(unique_labels[0], (int, np.integer)):
            try: benign_label = int(benign_label)
            except: pass

    if benign_label not in label_indices:
        benign_label = max(label_indices, key=lambda k: len(label_indices[k]))
    
    logging.info(f"Using benign label: {benign_label}")
    
    attack_labels = [l for l in unique_labels if l != benign_label]
    
    # 3. Determine Attack Order
    if args.attack_order:
        preferred_order = [atk.strip() for atk in args.attack_order.split(',')]
        if isinstance(unique_labels[0], (int, np.integer)):
            try: preferred_order = [int(atk) for atk in preferred_order]
            except: pass
        final_attack_order = [atk for atk in preferred_order if atk in attack_labels]
        remaining = [atk for atk in attack_labels if atk not in final_attack_order]
        final_attack_order += remaining
    else:
        # Default: Sort by sample count Descending
        final_attack_order = sorted(
            list(attack_labels), 
            key=lambda x: len(label_indices[x]), 
            reverse=True
        )
    
    if args.skip_attacks:
        skip_list = [atk.strip() for atk in args.skip_attacks.split(',')]
        if isinstance(unique_labels[0], (int, np.integer)):
            try: skip_list = [int(atk) for atk in skip_list]
            except: pass
        final_attack_order = [atk for atk in final_attack_order if atk not in skip_list]
        
    logging.info(f"Final Attack Sequence: {final_attack_order}")

    # 4. Generate Windows (Phases)
    window_size = args.window_size
    benign_ratio = args.benign_ratio
    windows_per_attack = args.windows_per_attack
    
    total_phases = len(final_attack_order) + 1
    total_windows = total_phases * windows_per_attack
    
    logging.info(f"Total Windows: {total_windows} ({windows_per_attack} per phase)")
    
    new_indices = []
    cursors = {lbl: 0 for lbl in unique_labels}

    def get_chunk_indices(lbl, n_needed):
        indices = label_indices[lbl]
        if len(indices) == 0:
            return []
        res = []
        current_cursor = cursors[lbl]
        for _ in range(n_needed):
            res.append(indices[current_cursor % len(indices)])
            current_cursor += 1
        cursors[lbl] = current_cursor
        return res

    for w in range(total_windows):
        phase_idx = w // windows_per_attack
        # Determine current window index within the phase (0 to windows_per_attack-1)
        w_in_phase = w % windows_per_attack
        
        # --- MODIFICATION: Half-Phase Logic ---
        # Determine if we are in the second half of the phase
        is_second_half = w_in_phase >= (windows_per_attack // 2)

        if phase_idx == 0:
            current_main_attack = None
            previous_attacks = []
        else:
            attack_idx = phase_idx - 1
            if attack_idx >= len(final_attack_order):
                attack_idx = len(final_attack_order) - 1
            current_main_attack = final_attack_order[attack_idx]
            previous_attacks = final_attack_order[:attack_idx]

        n_benign = int(window_size * benign_ratio)
        n_malicious = window_size - n_benign
        
        window_indices = []
        
        # Add Benign samples
        window_indices.extend(get_chunk_indices(benign_label, n_benign))
        
        # Add Malicious samples
        if current_main_attack is None:
            # Phase 0: Fill with benign/noise
            window_indices.extend(get_chunk_indices(benign_label, n_malicious))
        else:
            # --- MODIFIED LOGIC START ---
            
            # Logic: 
            # 1. If First Half OR No previous attacks: Pure Current Attack
            # 2. If Second Half AND Previous attacks exist: Mix Current (50%) + Previous Total (50%)
            #    (Note: 50% of the malicious part equals 20% of the total window if benign_ratio is 0.6)
            
            should_mix = is_second_half and (len(previous_attacks) > 0)
            
            if not should_mix:
                # Pure Attack Phase (First half or first attack)
                n_main = n_malicious
                n_prev_each = 0
            else:
                # Mixed Phase (Second half)
                # Target: 20% Total Main, 20% Total Previous (given 40% Malicious space)
                # This implies a 0.5 split of the malicious chunk
                n_prev_total = int(n_malicious * 0.5) 
                n_main = n_malicious - n_prev_total
                
                # Distribute n_prev_total among all previous attacks
                n_prev_each = n_prev_total // len(previous_attacks)
                
                # Recalculate n_main to absorb rounding errors and ensure total sums to n_malicious
                n_main = n_malicious - (n_prev_each * len(previous_attacks))

            # Add Main Attack
            window_indices.extend(get_chunk_indices(current_main_attack, n_main))
            
            # Add Previous Attacks (if any)
            if n_prev_each > 0:
                for prev_atk in previous_attacks:
                    window_indices.extend(get_chunk_indices(prev_atk, n_prev_each))
            
            # --- MODIFIED LOGIC END ---
        
        # Shuffle within the window
        np.random.shuffle(window_indices)
        new_indices.extend(window_indices)

    return new_indices, benign_label

def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Redistribution Script")
    parser.add_argument("--input", required=True, help="Path to input file (.csv or .npy features)")
    parser.add_argument("--input_labels", help="Path to input labels (.npy only)")
    parser.add_argument("--output", help="Path to output file")
    parser.add_argument("--output_labels", help="Path to output labels")
    
    parser.add_argument("--benign_label", default=None, help="Label for benign traffic")
    parser.add_argument("--benign_ratio", type=float, default=0.6, help="Ratio of benign traffic per window")
    parser.add_argument("--window_size", type=int, default=4000, help="Samples per window")
    parser.add_argument("--windows_per_attack", type=int, default=20, help="Windows per attack phase")
    parser.add_argument("--attack_order", help="Comma-separated list of attack labels to prioritize")
    parser.add_argument("--skip_attacks", help="Comma-separated list of attack labels to skip")
    parser.add_argument("--benign_source", help="Optional CSV file to load BENIGN traffic")
    parser.add_argument("--binary", action="store_true", help="Generate binary label file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    args.input = resolve_path(args.input)
    args.input_labels = resolve_path(args.input_labels)
    args.benign_source = resolve_path(args.benign_source)

    is_csv = args.input.endswith('.csv')
    if args.output is None:
        input_basename = os.path.basename(args.input)
        dataset_name = input_basename.split('.')[0]
        dataset_name = dataset_name.replace('_train_data', '').replace('_data', '').replace('_X', '')
        
        args.output = f"{dataset_name}-redistributed_mixed_X.npy"
        if args.output_labels is None:
            args.output_labels = f"{dataset_name}-redistributed_mixed_y.npy"

    args.output = resolve_path(args.output)
    args.output_labels = resolve_path(args.output_labels)

    if is_csv:
        df = load_csv(args.input, args.benign_source)
        labels = df['Label'].values
        new_indices, final_benign_label = redistribute_data(None, labels, args)
        
        logging.info("Reconstructing redistributed DataFrame...")
        new_df = df.iloc[new_indices].reset_index(drop=True)
        
        logging.info("Converting redistributed CSV to NPY format...")
        y_raw = new_df['Label'].values
        X_df = new_df.drop(columns=['Label'])
        
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_df.fillna(0, inplace=True)
        
        new_X = X_df.values.astype(np.float32)
        
        unique_labels = np.unique(y_raw)
        other_labels = sorted([l for l in unique_labels if str(l) != str(final_benign_label)])
        label_order = [str(final_benign_label)] + other_labels
        
        label_map = {l: i for i, l in enumerate(label_order)}
        new_y = np.array([label_map[str(l)] for l in y_raw], dtype=np.int32)
        
        logging.info(f"Label mapping used: {label_map}")

        labels_json_path = args.output_labels.replace('.npy', '.json')
        if labels_json_path != args.output_labels:
            with open(labels_json_path, 'w') as f:
                json.dump(label_map, f, indent=4)

        logging.info(f"Saving to {args.output} and {args.output_labels} ...")
        np.save(args.output, new_X)
        np.save(args.output_labels, new_y)

        if args.binary:
            binary_labels_path = args.output_labels.replace('.npy', '_binary.npy')
            if binary_labels_path == args.output_labels:
                binary_labels_path = args.output_labels + ".binary.npy"
            new_y_binary = (new_y != 0).astype(np.int32)
            np.save(binary_labels_path, new_y_binary)
    else:
        if not args.input_labels:
            raise ValueError("--input_labels is required for NPY format")
        X, y = load_npy(args.input, args.input_labels)
        
        new_indices, final_benign_label = redistribute_data(X, y, args)
        
        logging.info("Creating redistributed arrays...")
        new_X = X[new_indices]
        new_y_raw = y[new_indices]
        
        unique_labels = np.unique(new_y_raw)
        if isinstance(final_benign_label, (int, np.integer)):
            benign_val = final_benign_label
        else:
            benign_val = 0 
        
        other_labels = sorted([l for l in unique_labels if l != benign_val])
        label_order = [benign_val] + other_labels if benign_val in unique_labels else list(sorted(unique_labels))
        
        label_map = {old_label: new_idx for new_idx, old_label in enumerate(label_order)}
        new_y = np.array([label_map[l] for l in new_y_raw], dtype=np.int32)
        
        logging.info(f"Label remapping: {label_map}")
        
        label_out = args.output_labels
        if not label_out:
            label_out = args.output.replace('_X.npy', '_y.npy')
            if label_out == args.output: label_out = args.output + ".labels.npy"
        
        labels_json_path = label_out.replace('.npy', '.json')
        if labels_json_path != label_out:
            label_map_json = {int(k): int(v) for k, v in label_map.items()}
            with open(labels_json_path, 'w') as f:
                json.dump(label_map_json, f, indent=4)
        
        logging.info(f"Saving to {args.output} ...")
        np.save(args.output, new_X)
        np.save(label_out, new_y)

        if args.binary:
            binary_labels_path = label_out.replace('.npy', '_binary.npy')
            if binary_labels_path == label_out:
                binary_labels_path = label_out + ".binary.npy"
            new_y_binary = (new_y != 0).astype(np.int32)
            np.save(binary_labels_path, new_y_binary)

    logging.info("Done.")

if __name__ == "__main__":
    main()