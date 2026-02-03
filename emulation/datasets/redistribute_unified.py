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
    Checks environment variable first, defaults to relative path.
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
    Select specific features for known datasets (like CIC-IDS-2018).
    This ensures we only use the correct features for each dataset.
    """
    # CIC-IDS-2018: 16 features (consistent with pForest paper)
    CIC_IDS_2018_FEATURES = [
        "Flow IAT Min", "Flow IAT Max", "Flow IAT Mean",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        "Total Length of Fwd Packets", "Total Fwd Packets",
        "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
        "FIN Flag Count", "RST Flag Count", "ECE Flag Count",
        "Flow Duration", "Destination Port"
    ]
    
    # Detect dataset type from filename or available columns
    if 'cic-ids-2018' in dataset_name.lower() or 'cic_ids_2018' in dataset_name.lower():
        target_features = CIC_IDS_2018_FEATURES
        dataset_type = 'CIC-IDS-2018'
    else:
        # Try to auto-detect based on available columns
        if all(f in df.columns for f in CIC_IDS_2018_FEATURES):
            target_features = CIC_IDS_2018_FEATURES
            dataset_type = 'CIC-IDS-2018 (auto-detected)'
        else:
            # No specific feature selection, use all features
            logging.info("No specific feature selection applied. Using all features.")
            return df
    
    # Check if all target features exist
    missing_features = [f for f in target_features if f not in df.columns]
    if missing_features:
        logging.warning(f"Missing features for {dataset_type}: {missing_features}")
        logging.warning(f"Available columns: {df.columns.tolist()}")
        logging.warning("Using all available features instead.")
        return df
    
    # Select target features + Label
    logging.info(f"Detected dataset type: {dataset_type}")
    logging.info(f"Selecting {len(target_features)} features: {target_features}")
    
    cols_to_keep = target_features + ['Label']
    df_selected = df[cols_to_keep]
    
    logging.info(f"Feature selection: {len(df.columns)-1} → {len(df_selected.columns)-1} features")
    
    return df_selected

def load_csv(path, benign_source=None):
    """
    Loads a CSV file and handles label column normalization.
    Optionally merges external benign traffic if missing in the source.
    Automatically selects appropriate features for known datasets.
    """
    logging.info(f"Reading CSV from {path} ...")
    df = pd.read_csv(path)
    
    # Normalize label column name (case-insensitive search)
    if 'Label' not in df.columns:
        for col in df.columns:
            if col.lower() == 'label':
                df.rename(columns={col: 'Label'}, inplace=True)
                break
    
    df['Label'] = df['Label'].astype(str).str.strip()
    
    # Special handling: if BENIGN is missing, try loading from benign_source
    if 'BENIGN' not in df['Label'].values and benign_source:
        logging.info(f"BENIGN not found in input. Loading from {benign_source}...")
        if os.path.exists(benign_source):
            df_benign = pd.read_csv(benign_source)
            df_benign.columns = df_benign.columns.str.strip()
            if 'Label' in df_benign.columns:
                df_benign['Label'] = df_benign['Label'].astype(str).str.strip()
                # concat only BENIGN samples
                df = pd.concat([df, df_benign[df_benign['Label'] == 'BENIGN']])
            else:
                logging.warning(f"Label column not found in {benign_source}")
        else:
            logging.warning(f"Benign source {benign_source} not found.")
    
    # Apply feature selection for known datasets
    dataset_name = os.path.basename(path)
    df = select_features_for_dataset(df, dataset_name)
            
    return df

def load_npy(data_path, label_path):
    """
    Loads features (X) and labels (y) from .npy files.
    """
    logging.info(f"Reading NPY data from {data_path} and labels from {label_path} ...")
    X = np.load(data_path)
    y = np.load(label_path)
    if len(X) != len(y):
        raise ValueError(f"Mismatch between X ({len(X)}) and y ({len(y)})")
    return X, y

def redistribute_data(data, labels, args):
    """
    Core logic to redistribute data indices based on attack phases.
    Returns a list of indices representing the new data order.
    """
    if args.seed is not None:
        np.random.seed(args.seed)
        
    unique_labels = np.unique(labels)
    label_indices = {}

    # 1. Group indices by label and shuffle them internally
    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]
        # Use a local random state for shuffling if seed is provided
        if args.seed is not None:
            np.random.RandomState(args.seed).shuffle(indices)
        else:
            np.random.shuffle(indices)
        label_indices[lbl] = indices
        logging.info(f"  Label {lbl}: {len(indices)} samples")

    # 2. Identify the Benign label
    benign_label = args.benign_label
    if benign_label is None:
        # Auto-detect benign label logic
        if 'BENIGN' in label_indices:
            benign_label = 'BENIGN'
        elif 0 in label_indices:
            benign_label = 0
        elif '0' in label_indices:
            benign_label = '0'
        else:
            # Fallback: assume the label with the most samples is benign
            benign_label = max(label_indices, key=lambda k: len(label_indices[k]))
            logging.info(f"Auto-detected benign label: {benign_label} (most samples)")
    else:
        # Try to match type with unique_labels (handle int vs str issues)
        if isinstance(unique_labels[0], (int, np.integer)):
            try: benign_label = int(benign_label)
            except: pass

    if benign_label not in label_indices:
        logging.warning(f"Benign label '{benign_label}' not found. Using label with most samples.")
        benign_label = max(label_indices, key=lambda k: len(label_indices[k]))
    
    logging.info(f"Using benign label: {benign_label}")
    
    # Get all labels that are NOT benign
    attack_labels = [l for l in unique_labels if l != benign_label]
    
    # 3. Determine Attack Order
    # Logic: If --attack_order is provided, use it.
    #        Else, sort attacks by sample count (Descending).
    if args.attack_order:
        preferred_order = [atk.strip() for atk in args.attack_order.split(',')]
        
        # Handle type conversion if labels are integers
        if isinstance(unique_labels[0], (int, np.integer)):
            try: preferred_order = [int(atk) for atk in preferred_order]
            except: pass
        
        # Filter strictly for existing labels
        final_attack_order = [atk for atk in preferred_order if atk in attack_labels]
        # Append any remaining attacks that weren't in the arguments
        remaining = [atk for atk in attack_labels if atk not in final_attack_order]
        final_attack_order += remaining
    else:
        # --- MODIFICATION START ---
        # Sort by number of samples (Descending: Most samples -> Least samples)
        # remove `reverse=True` if you want Ascending order.
        logging.info("No attack order specified. Sorting attacks by sample count (Descending).")
        final_attack_order = sorted(
            list(attack_labels), 
            key=lambda x: len(label_indices[x]), 
            reverse=True
        )
        # --- MODIFICATION END ---
    
    # Remove attacks specified in --skip_attacks
    if args.skip_attacks:
        skip_list = [atk.strip() for atk in args.skip_attacks.split(',')]
        # Handle type conversion if labels are integers
        if isinstance(unique_labels[0], (int, np.integer)):
            try: skip_list = [int(atk) for atk in skip_list]
            except: pass
        
        before_count = len(final_attack_order)
        final_attack_order = [atk for atk in final_attack_order if atk not in skip_list]
        skipped = before_count - len(final_attack_order)
        if skipped > 0:
            logging.info(f"Skipped {skipped} attack(s) from --skip_attacks: {skip_list}")
        
    logging.info(f"Final Attack Sequence: {final_attack_order}")

    # 4. Generate Windows (Phases)
    window_size = args.window_size
    benign_ratio = args.benign_ratio
    windows_per_attack = args.windows_per_attack
    
    # Total phases = 1 (Initial benign/warmup phase) + N (number of attack types)
    total_phases = len(final_attack_order) + 1
    total_windows = total_phases * windows_per_attack
    
    logging.info(f"Total Windows: {total_windows} ({windows_per_attack} per phase)")
    
    new_indices = []
    cursors = {lbl: 0 for lbl in unique_labels}

    # Helper function to get chunk of indices (wrapping around if needed)
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
        
        # Phase 0 is Warmup (No main attack), subsequent phases focus on specific attacks
        if phase_idx == 0:
            current_main_attack = None
            previous_attacks = []
        else:
            attack_idx = phase_idx - 1
            # Clamp index just in case
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
            # In warmup phase, fill malicious part with benign or noise (here: benign)
            window_indices.extend(get_chunk_indices(benign_label, n_malicious))
        else:
            if not previous_attacks:
                # First attack phase: All malicious traffic is the current attack
                n_main = n_malicious
                n_prev_each = 0
            else:
                # Subsequent phases: Mix current attack (majority) with previous attacks (minority)
                # Reserve ~20% for previous attacks replay
                n_prev_total = int(n_malicious * 0.2)
                n_main = n_malicious - n_prev_total
                
                # Distribute the previous total among all previous attacks
                n_prev_each = n_prev_total // len(previous_attacks) if previous_attacks else 0
                
                # Adjust n_main to absorb rounding errors
                n_main = n_malicious - (n_prev_each * len(previous_attacks))

            # Add Main Attack
            window_indices.extend(get_chunk_indices(current_main_attack, n_main))
            # Add Previous Attacks (Replay)
            for prev_atk in previous_attacks:
                window_indices.extend(get_chunk_indices(prev_atk, n_prev_each))
        
        # Shuffle within the window to mix benign and malicious
        np.random.shuffle(window_indices)
        new_indices.extend(window_indices)

    return new_indices, benign_label

def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Redistribution Script")
    parser.add_argument("--input", required=True, help="Path to input file (.csv or .npy features)")
    parser.add_argument("--input_labels", help="Path to input labels (.npy only)")
    parser.add_argument("--output", help="Path to output file (.csv or .npy features). Automatically generated if not provided.")
    parser.add_argument("--output_labels", help="Path to output labels (.npy only). Automatically generated if not provided.")
    
    parser.add_argument("--benign_label", default=None, help="Label for benign traffic (default: auto-detect 0 or BENIGN)")
    parser.add_argument("--benign_ratio", type=float, default=0.6, help="Ratio of benign traffic per window")
    parser.add_argument("--window_size", type=int, default=4000, help="Samples per window")
    parser.add_argument("--windows_per_attack", type=int, default=20, help="Windows per attack phase")
    parser.add_argument("--attack_order", help="Comma-separated list of attack labels to prioritize")
    parser.add_argument("--skip_attacks", help="Comma-separated list of attack labels to skip/exclude")
    parser.add_argument("--benign_source", help="Optional CSV file to load BENIGN traffic from if missing in input")
    parser.add_argument("--binary", action="store_true", help="Generate binary label file (y_binary.npy) in addition to multi-class labels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Resolve paths relative to emulation/datasets/
    args.input = resolve_path(args.input)
    args.input_labels = resolve_path(args.input_labels)
    args.benign_source = resolve_path(args.benign_source)

    # Automatic output naming if not provided
    is_csv = args.input.endswith('.csv')
    if args.output is None:
        input_basename = os.path.basename(args.input)
        # Extract meaningful part of the name
        dataset_name = input_basename.split('.')[0]
        # Common cleanups
        dataset_name = dataset_name.replace('_train_data', '').replace('_data', '').replace('_X', '')
        
        args.output = f"{dataset_name}-redistributed_X.npy"
        if args.output_labels is None:
            args.output_labels = f"{dataset_name}-redistributed_y.npy"

    args.output = resolve_path(args.output)
    args.output_labels = resolve_path(args.output_labels)

    if is_csv:
        df = load_csv(args.input, args.benign_source)
        # Convert to numpy for processing
        labels = df['Label'].values
        # Get redistributed indices
        new_indices, final_benign_label = redistribute_data(None, labels, args)
        
        logging.info("Reconstructing redistributed DataFrame...")
        # Use iloc to reorder the DataFrame based on new indices
        new_df = df.iloc[new_indices].reset_index(drop=True)
        
        logging.info("Converting redistributed CSV to NPY format...")
        # Separate labels and features
        y_raw = new_df['Label'].values
        X_df = new_df.drop(columns=['Label'])
        
        # Clean features: handle non-numeric, inf, nan
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_df.fillna(0, inplace=True)
        
        new_X = X_df.values.astype(np.float32)
        
        # Map labels to integers
        unique_labels = np.unique(y_raw)
        # Ensure benign_label is at index 0 if it exists
        other_labels = sorted([l for l in unique_labels if str(l) != str(final_benign_label)])
        label_order = [str(final_benign_label)] + other_labels
        
        label_map = {l: i for i, l in enumerate(label_order)}
        new_y = np.array([label_map[str(l)] for l in y_raw], dtype=np.int32)
        
        logging.info(f"Label mapping used: {label_map}")

        # Save label mapping to JSON
        labels_json_path = args.output_labels.replace('.npy', '.json')
        if labels_json_path != args.output_labels:
            with open(labels_json_path, 'w') as f:
                json.dump(label_map, f, indent=4)
            logging.info(f"Label mapping saved to {labels_json_path}")

        logging.info(f"Saving to {args.output} and {args.output_labels} ...")
        np.save(args.output, new_X)
        np.save(args.output_labels, new_y)

        if args.binary:
            binary_labels_path = args.output_labels.replace('.npy', '_binary.npy')
            if binary_labels_path == args.output_labels:
                binary_labels_path = args.output_labels + ".binary.npy"
            new_y_binary = (new_y != 0).astype(np.int32)
            logging.info(f"Saving binary labels to {binary_labels_path} ...")
            np.save(binary_labels_path, new_y_binary)
    else:
        # NPY Handling
        if not args.input_labels:
            raise ValueError("--input_labels is required for NPY format")
        X, y = load_npy(args.input, args.input_labels)
        
        new_indices, final_benign_label = redistribute_data(X, y, args)
        
        logging.info("Creating redistributed arrays...")
        new_X = X[new_indices]
        new_y_raw = y[new_indices]
        
        # Remap labels to consecutive integers (0, 1, 2, ..., N-1)
        # This is important when --skip_attacks is used, as it creates gaps in label values
        unique_labels = np.unique(new_y_raw)
        
        # Ensure benign_label (usually 0) stays at index 0
        if isinstance(final_benign_label, (int, np.integer)):
            benign_val = final_benign_label
        else:
            benign_val = 0  # Default assumption
        
        # Build label order: benign first, then others sorted
        other_labels = sorted([l for l in unique_labels if l != benign_val])
        label_order = [benign_val] + other_labels if benign_val in unique_labels else list(sorted(unique_labels))
        
        label_map = {old_label: new_idx for new_idx, old_label in enumerate(label_order)}
        new_y = np.array([label_map[l] for l in new_y_raw], dtype=np.int32)
        
        logging.info(f"Label remapping: {label_map}")
        logging.info(f"Final number of classes: {len(unique_labels)}")
        
        # Save label mapping to JSON for reference
        label_out = args.output_labels
        if not label_out:
            label_out = args.output.replace('_X.npy', '_y.npy')
            if label_out == args.output: label_out = args.output + ".labels.npy"
        
        labels_json_path = label_out.replace('.npy', '.json')
        if labels_json_path != label_out:
            # Convert numpy int keys to regular int for JSON serialization
            label_map_json = {int(k): int(v) for k, v in label_map.items()}
            with open(labels_json_path, 'w') as f:
                json.dump(label_map_json, f, indent=4)
            logging.info(f"Label mapping saved to {labels_json_path}")
        
        logging.info(f"Saving to {args.output} ...")
        np.save(args.output, new_X)
        
        np.save(label_out, new_y)
        logging.info(f"Labels saved to {label_out}")

        if args.binary:
            binary_labels_path = label_out.replace('.npy', '_binary.npy')
            if binary_labels_path == label_out:
                binary_labels_path = label_out + ".binary.npy"
            new_y_binary = (new_y != 0).astype(np.int32)  # 0 is now always benign after remapping
            logging.info(f"Saving binary labels to {binary_labels_path} ...")
            np.save(binary_labels_path, new_y_binary)

    logging.info("Done.")

if __name__ == "__main__":
    main()