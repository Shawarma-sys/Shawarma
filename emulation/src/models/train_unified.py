import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
import argparse
import logging
from datetime import datetime
import time
import numpy as np
import torchlens as tl

# Import datasets
from dataset.dataset import UnifiedTrafficDataset, traffic_collate_fn

# Import models
from models.cnnmodel import TextCNN1, TextCNN2
from models.teacher_cnnmodel import TextCNNTeacher
from models.rnnmodel import RNN1
from models.teacher_rnnmodel import RNNTeacher
from models.mlp import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'datasets')
DEFAULT_LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs', 'train')
DEFAULT_MODEL_ROOT = os.path.join(PROJECT_ROOT, 'models', 'checkpoint')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description='Unified Traffic Classifier (MLP/CNN/RNN)')
    
    # General Arguments
    parser.add_argument('--model_type', type=str, required=True, choices=['mlp', 'cnn', 'rnn'], help='Type of model to train')
    parser.add_argument('--model_class', type=str, required=True, help='Specific model class name')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data .npy')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels .npy')
    parser.add_argument('--test_data', type=str, help='Path to test data .npy')
    parser.add_argument('--test_labels', type=str, help='Path to test labels .npy')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/Test split ratio if test_data is not provided')
    parser.add_argument('--num_classes', type=int, help='Number of classes (automatically inferred if not provided)')
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=125)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # Model Specific Arguments (CNN/RNN)
    parser.add_argument('--window_size', type=int, default=9, help='Sequence length for CNN/RNN')
    parser.add_argument('--len_vocab', type=int, default=1501)
    parser.add_argument('--ipd_vocab', type=int, default=2561)
    parser.add_argument('--len_embedding_bits', type=int, default=10)
    parser.add_argument('--ipd_embedding_bits', type=int, default=8)
    
    # CNN Specific
    parser.add_argument('--input_size', type=int, default=4)
    parser.add_argument('--nk', type=int, default=4)
    parser.add_argument('--ebdin', type=int, default=4)
    
    # RNN Specific
    parser.add_argument('--rnnin', type=int, default=12)
    parser.add_argument('--rnnhn', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)

    # MLP Specific
    parser.add_argument('--standardize', action='store_true', default=True)
    parser.add_argument('--normalize', action='store_true', default=False)
    
    # Rule Extraction Control
    parser.add_argument('--extract_rules', type=str, default='both', 
                        choices=['linc', 'helios', 'both', 'none'],
                        help='Which rules to extract: linc, helios, both, or none (default: both)')

    args = parser.parse_args()

    # Infer dataset name from train_data path
    dataset_basename = os.path.basename(args.train_data)
    dataset_name = dataset_basename.split('.')[0].replace('_X', '').replace('_redist', '').replace('-redistributed', '')

    # Check if using binary labels (from label file name)
    labels_basename = os.path.basename(args.train_labels)
    is_binary = 'binary' in labels_basename.lower()

    # Map dataset names to num_classes (fallback only)
    dataset_class_map = {
        'cic-ids-2018': 7,
        'cicids2018': 7,
        'iscxvpn': 7,
        'ustc-tfc2016': 12,
    }

    if args.num_classes is None:
        if is_binary:
            # Binary classification
            args.num_classes = 2
        else:
            # Try to auto-detect from label file first
            try:
                labels = np.load(args.train_labels)
                args.num_classes = len(np.unique(labels))
                print(f"Auto-detected {args.num_classes} classes from label file: {args.train_labels}")
            except Exception as e:
                print(f"Warning: Could not read label file for class detection: {e}")
                # Fallback: Try to infer from dataset name
                args.num_classes = 7  # Default
                for key, val in dataset_class_map.items():
                    if key in dataset_name.lower():
                        args.num_classes = val
                        break
                print(f"Using fallback: {args.num_classes} classes for {dataset_name}")

    # Setup Directories
    save_dir = os.path.join(DEFAULT_MODEL_ROOT, args.model_type)
    log_dir = os.path.join(DEFAULT_LOG_ROOT, args.model_type)
    ensure_dir(save_dir)
    ensure_dir(log_dir)

    # Logging - setup before any logging calls
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{dataset_name}_{args.model_class}_{timestamp}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.info(f"Starting unified training for {args.model_class} ({args.model_type})")
    logging.info(f"Arguments: {args}")
    if is_binary:
        logging.info(f"Detected binary classification from label file: {labels_basename}")

    # Dataset Loading
    full_dataset = UnifiedTrafficDataset(
        args.train_data, 
        args.train_labels, 
        sequence_length=args.window_size if args.model_type in ['cnn', 'rnn'] else None,
        standardize=args.standardize if args.model_type == 'mlp' else False,
        normalize=args.normalize if args.model_type == 'mlp' else False,
        device=DEVICE
    )

    if 'Teacher' not in args.model_class:
        # Select the first 160000 samples (windows 1-40) for training
        # This simulates a model trained only on initial traffic (mostly benign)
        subset_size = 160000
        full_dataset = torch.utils.data.Subset(full_dataset, range(subset_size))
    logging.info(f"Loaded dataset with {len(full_dataset)} samples.")

    # Splitting
    if args.test_data and args.test_labels:
        train_dataset = full_dataset
        test_dataset = UnifiedTrafficDataset(
            args.test_data, 
            args.test_labels, 
            sequence_length=args.window_size if args.model_type in ['cnn', 'rnn'] else None,
            standardize=args.standardize if args.model_type == 'mlp' else False,
            normalize=args.normalize if args.model_type == 'mlp' else False,
            device=DEVICE
        )
    else:
        train_size = int(args.split_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        logging.info(f"Split dataset: {train_size} train, {test_size} test (ratio {args.split_ratio})")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=traffic_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=traffic_collate_fn)

    # Model Initialization
    if args.model_type == 'cnn':
        model_cls = globals()[args.model_class]
        model = model_cls(input_size=args.input_size, num_classes=args.num_classes,
                          len_vocab=args.len_vocab, ipd_vocab=args.ipd_vocab,
                          len_embedding_bits=args.len_embedding_bits, ipd_embedding_bits=args.ipd_embedding_bits,
                          nk=args.nk, ebdin=args.ebdin, device=DEVICE).to(DEVICE)
        dummy_input = torch.ones((2, args.window_size, args.input_size), dtype=torch.long).to(DEVICE)
    elif args.model_type == 'rnn':
        if args.model_class == 'RNNTeacher' and args.dropout == 0.0: args.dropout = 0.1
        model_cls = globals()[args.model_class]
        model = model_cls(rnn_in=args.rnnin, hidden_size=args.rnnhn, labels_num=args.num_classes,
                          len_vocab=args.len_vocab, ipd_vocab=args.ipd_vocab,
                          len_embedding_bits=args.len_embedding_bits, ipd_embedding_bits=args.ipd_embedding_bits,
                          device=DEVICE, droprate=args.dropout).to(DEVICE)
        dummy_input = torch.ones((2, args.window_size, 2), dtype=torch.long).to(DEVICE)
    else: # mlp
        model_cls = globals()[args.model_class]
        
        # Infer input size from first sample
        sample_x, _ = full_dataset[0]
        input_dim = sample_x.shape[0]
        
        # Check model signature and create model accordingly
        import inspect
        sig = inspect.signature(model_cls.__init__)
        
        if 'num_classes' in sig.parameters:
            # Models that take num_classes (e.g., MLP1Teacher_multi)
            model = model_cls(num_classes=args.num_classes).to(DEVICE)
        else:
            # Simple models without parameters (e.g., MLP1, MLP1Teacher)
            model = model_cls().to(DEVICE)
        
        dummy_input = torch.ones((2, input_dim)).to(DEVICE)

    # TorchLens Verification (Optional, mainly for sequence models)
    if args.model_type in ['cnn', 'rnn']:
        model.eval()
        with torch.no_grad():
            tl_model = tl.log_forward_pass(model, dummy_input, vis_opt='unrolled', save_function_args=True,
                                        vis_outpath=f'{PROJECT_ROOT}/scripts/results/{dataset_name}_{args.model_class}',
                                        vis_save_only=True)
            logging.info("TorchLens verification pass completed.")
    
    model.train()

    # Class Weights
    def get_labels(ds):
        if isinstance(ds, torch.utils.data.Subset):
            # Recursively get labels from the base dataset using indices
            base_labels = get_labels(ds.dataset)
            return base_labels[ds.indices]
        return ds.labels

    train_labels_tensor = get_labels(train_dataset)
    class_counts = torch.bincount(train_labels_tensor)
    
    # Ensure class_counts matches num_classes
    if len(class_counts) > args.num_classes:
        logging.warning(f"Data contains {len(class_counts)} classes, but --num_classes is set to {args.num_classes}. Truncating weights.")
        class_counts = class_counts[:args.num_classes]
    elif len(class_counts) < args.num_classes:
        new_counts = torch.zeros(args.num_classes, dtype=class_counts.dtype, device=class_counts.device)
        new_counts[:len(class_counts)] = class_counts
        class_counts = new_counts

    C = 1.01
    class_weights = 1.0 / torch.log(C + class_counts.float())
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(DEVICE)

    # Training Setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    def run_epoch(model, loader, criterion, optimizer=None):
        if optimizer: model.train()
        else: model.eval()
        total_loss = 0
        all_labels, all_preds = [], []
        context = torch.enable_grad() if optimizer else torch.no_grad()
        with context:
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                # Handle MLP models that might have run_inference
                if hasattr(model, 'run_inference') and not optimizer:
                    outputs, _ = model.run_inference(batch_x)
                else:
                    outputs = model(batch_x)
                
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss = criterion(outputs, batch_y)
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(batch_y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Training Loop
    best_f1 = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(save_dir, f"best_{args.model_class}_{dataset_name}_{timestamp}.pth")
    # Also save a version without timestamp for easy loading
    best_model_path_no_timestamp = os.path.join(save_dir, f"best_{args.model_class}_{dataset_name}.pth")

    for epoch in range(args.epochs):
        t_loss, t_acc, t_f1 = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, v_f1 = run_epoch(model, test_loader, criterion)
        logging.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {t_loss:.4f}, F1: {t_f1:.4f} | Val Loss: {v_loss:.4f}, F1: {v_f1:.4f}")
        
        scheduler.step(v_f1)
        if v_f1 > best_f1:
            best_f1 = v_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), best_model_path_no_timestamp)
            logging.info(f"Saved new best model (F1: {best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    logging.info(f"Training complete. Best Val F1: {best_f1:.4f}")
    
    # Extract LINC/Helios rules only for inference models (not Teacher/labeler models)
    if 'Teacher' not in args.model_class:
        # Load best model for rule extraction
        model.load_state_dict(torch.load(best_model_path))
        
        if args.extract_rules in ['linc', 'both']:
            extract_linc_rules(model, train_loader, args, save_dir, dataset_name, timestamp)
        else:
            logging.info("Skipping LINC rule extraction (extract_rules != 'linc' or 'both')")
        
        if args.extract_rules in ['helios', 'both']:
            extract_helios_rules(model, train_loader, args, save_dir, dataset_name, timestamp)
        else:
            logging.info("Skipping Helios rule extraction (extract_rules != 'helios' or 'both')")
    else:
        logging.info("Skipping LINC/Helios rule extraction for Teacher/labeler model")


def extract_embedding_features(model, features, args):
    """
    Extract embedding features from sequence models for rule learning.
    
    For sequence models (RNN1, TextCNN1, etc.), raw token IDs are not suitable
    for threshold-based rules. We extract the learned embedding representations.
    
    OPTIMIZED: For RNN1, extract the final hidden state which contains richer
    sequential information than just mean-pooled embeddings.
    
    Args:
        model: Trained PyTorch model
        features: Input tensor (batch_size, seq_len, feature_dim) for sequence models
                  or (batch_size, feature_dim) for MLP models
        args: Training arguments containing model_type and model_class
    
    Returns:
        numpy array of shape (batch_size, embedding_dim) suitable for rule learning
    """
    if args.model_type not in ['cnn', 'rnn']:
        # For MLP models, just flatten and return
        if torch.is_tensor(features):
            return features.cpu().numpy().reshape(features.shape[0], -1)
        return np.array(features).reshape(len(features), -1)
    
    # For sequence models, extract embeddings using the trained model
    with torch.no_grad():
        features = features.to(DEVICE)
        if features.dtype != torch.long:
            features = features.long()
        
        # Extract length and IPD tokens
        len_x = features[:, :, 0]  # (batch_size, seq_len)
        ipd_x = features[:, :, 1]  # (batch_size, seq_len)
        
        # Clamp inputs to valid vocabulary range
        if hasattr(model, 'len_embedding'):
            len_vocab = model.len_embedding.num_embeddings
            len_x = torch.clamp(len_x, 0, len_vocab - 1)
        if hasattr(model, 'ipd_embedding'):
            ipd_vocab = model.ipd_embedding.num_embeddings
            ipd_x = torch.clamp(ipd_x, 0, ipd_vocab - 1)
        
        # Use the trained embedding layers from the model
        len_embedded = model.len_embedding(len_x)  # (batch_size, seq_len, len_emb_bits)
        ipd_embedded = model.ipd_embedding(ipd_x)  # (batch_size, seq_len, ipd_emb_bits)
        
        # Concatenate embeddings
        combined = torch.cat([len_embedded, ipd_embedded], dim=-1)
        
        # OPTIMIZED: For RNN models, extract only embedding features (Input Projection)
        if args.model_type == 'rnn' and hasattr(model, 'rnn') and hasattr(model, 'fc1'):
            # Modified: Extract only embedding features, using FC1 projection, then FLATTEN (no pooling)
            # Apply fc1 transformation: (batch_size, seq_len, rnn_in)
            x = model.fc1(combined)
            
            # Flatten the sequence: (batch_size, seq_len * rnn_in)
            result = x.reshape(x.shape[0], -1)
            
            return result.cpu().numpy()
        
        elif args.model_type == 'cnn' and hasattr(model, 'conv'):
            # For TextCNN, apply convolution and pooling
            x = model.fc1(combined) if hasattr(model, 'fc1') else combined
            
            # Apply convolution layers if available
            x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
            conv_out = model.conv(x)
            # Global max pooling
            pooled = torch.max(conv_out, dim=2)[0]
            return pooled.cpu().numpy()
        
        # Fallback: use mean and max pooling with fc1 transformation
        if hasattr(model, 'fc1'):
            combined = model.fc1(combined)
        
        # Aggregate across sequence: use both mean and max pooling for richer features
        mean_pooled = combined.mean(dim=1)
        max_pooled = combined.max(dim=1)[0]
        
        # Concatenate mean and max pooled features
        result = torch.cat([mean_pooled, max_pooled], dim=-1)
        
        return result.cpu().numpy()


def extract_linc_rules(model, train_loader, args, save_dir, dataset_name, timestamp, 
                       rule_threshold=0.5, max_rules=10000):
    """
    Extract LINC rules from trained model using training data.
    Generates TWO sets of rules:
    1. Original LINC rules using SoftTreeClassifier (linc_rules_*.json)
    2. Improved LINC rules using sklearn DecisionTree (linc_rules_improved_*.json)
    
    Args:
        model: Trained PyTorch model
        train_loader: DataLoader for training data
        args: Training arguments
        save_dir: Directory to save rules
        dataset_name: Name of dataset
        timestamp: Timestamp string
        rule_threshold: Minimum confidence threshold for rule extraction
        max_rules: Maximum number of rules to keep
    """
    import json
    import sys
    from pathlib import Path
    
    # Add benchmarks to path to import LincManager
    benchmarks_dir = Path(__file__).resolve().parents[2] / 'benchmarks'
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))
    
    from linc_manager import LincManager
    
    logging.info("Extracting LINC rules from trained model (both original and improved)...")
    model.eval()
    
    # Collect high-confidence correct predictions for rule extraction
    all_features = []
    all_labels = []
    class_stats = {}
    
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Get model predictions
            if hasattr(model, 'run_inference'):
                outputs, _ = model.run_inference(batch_x)
            else:
                outputs = model(batch_x)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get predictions and confidences
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = probs.max(dim=1)
            
            # Convert labels to numpy for statistics
            labels_np = batch_y.cpu().numpy()
            preds_np = predictions.cpu().numpy()
            confs_np = confidences.cpu().numpy()
            
            # Track statistics per class
            unique_classes = np.unique(labels_np)
            for c in unique_classes:
                if c not in class_stats:
                    class_stats[c] = {'total': 0, 'correct': 0, 'high_conf': 0}
                mask = labels_np == c
                class_stats[c]['total'] += mask.sum()
                class_stats[c]['correct'] += ((preds_np == labels_np) & mask).sum()
                class_stats[c]['high_conf'] += ((preds_np == labels_np) & (confs_np >= rule_threshold) & mask).sum()
            
            # Filter: only correct, high-confidence predictions
            valid_mask = (preds_np == labels_np) & (confs_np >= rule_threshold)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # Extract embedding features for sequence models, or flatten for MLP
                valid_batch_x = batch_x[valid_indices]
                valid_features = extract_embedding_features(model, valid_batch_x, args)
                valid_labels = labels_np[valid_indices]
                all_features.append(valid_features)
                all_labels.append(valid_labels)
    
    # Log per-class statistics
    for c in sorted(class_stats.keys()):
        total = class_stats[c]['total']
        correct = class_stats[c]['correct']
        high_conf = class_stats[c]['high_conf']
        if total > 0:
            logging.info(f"Class {c}: total={total}, correct={correct} ({correct/total*100:.1f}%), "
                        f"high_conf_correct={high_conf} ({high_conf/total*100:.1f}%)")
    
    if not all_features:
        logging.warning("No valid samples for LINC rule extraction")
        return []
    
    # Concatenate all valid samples
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    logging.info(f"Collected {len(X)} high-confidence correct samples for rule extraction")
    
    # Determine feature type for metadata
    uses_embedding = args.model_type in ['cnn', 'rnn']
    if uses_embedding:
        sample_x, _ = next(iter(train_loader))
        sample_emb = extract_embedding_features(model, sample_x[:1].to(DEVICE), args)
        feature_dim = sample_emb.shape[1]
        logging.info(f"Rules use embedding features with dimension {feature_dim}")
    else:
        feature_dim = None
    
    # Helper function to convert rules to serializable format
    def rules_to_json(linc_manager):
        rules = []
        for rule in linc_manager.rules:
            imp_feats = []
            threshs = []
            for idx, thresh, op in rule.conditions:
                imp_feats.append(int(idx))
                imp_feats.append(int(op))
                threshs.append(float(thresh))
            rules.append({
                'pattern': threshs,
                'important_features': imp_feats,
                'class': int(rule.class_label),
                'confidence': float(rule.confidence)
            })
        return rules
    
    # Helper function to save rules
    def save_rules(rules, mode_name, suffix=""):
        class_counts = {}
        for r in rules:
            c = r['class']
            class_counts[c] = class_counts.get(c, 0) + 1
        
        filename = f"linc_rules{suffix}_{args.model_class}_{dataset_name}"
        rules_path = os.path.join(save_dir, f"{filename}_{timestamp}.json")
        best_path = os.path.join(save_dir, f"{filename}.json")
        
        data = {
            'model_class': args.model_class,
            'dataset_name': dataset_name,
            'mode': mode_name,
            'num_rules': len(rules),
            'class_distribution': class_counts,
            'rule_threshold': rule_threshold,
            'uses_embedding_features': uses_embedding,
            'embedding_feature_dim': feature_dim,
            'rules': rules
        }
        
        with open(rules_path, 'w') as f:
            json.dump(data, f)
        with open(best_path, 'w') as f:
            json.dump(data, f)
        
        logging.info(f"LINC {mode_name} rules: {len(rules)}, class distribution: {class_counts}")
        logging.info(f"Saved to {best_path}")
        return rules
    
    # ========== Generate LINC Rules (sklearn DecisionTree) ==========
    logging.info("=" * 50)
    logging.info("Generating LINC rules (sklearn DecisionTree)...")
    linc_manager = LincManager(max_rules=max_rules, improved=True)
    linc_manager.fit(X, y)
    rules = rules_to_json(linc_manager)
    save_rules(rules, "improved", suffix="")
    
    logging.info("=" * 50)
    logging.info(f"LINC rule extraction complete: {len(rules)} rules")
    
    return rules


def extract_helios_rules(model, train_loader, args, save_dir, dataset_name, timestamp, 
                         rule_threshold=0.5, max_rules=2000):
    """
    Extract Helios rules using Prototype Network.
    
    This function trains a separate Helios Prototype Network on the training data
    and generates hypercube matching rules. The rules are saved to a JSON file.
    
    Args:
        model: Trained PyTorch model (used for filtering high-confidence samples)
        train_loader: DataLoader for training data
        args: Training arguments
        save_dir: Directory to save rules
        dataset_name: Name of dataset
        timestamp: Timestamp string
        rule_threshold: Minimum confidence threshold for sample filtering
        max_rules: Maximum number of rules to keep
    """
    import json
    import sys
    from pathlib import Path
    
    logging.info("Extracting Helios rules using Prototype Network...")
    model.eval()
    
    # Collect high-confidence correct predictions for rule extraction
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Get model predictions
            if hasattr(model, 'run_inference'):
                outputs, _ = model.run_inference(batch_x)
            else:
                outputs = model(batch_x)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get predictions and confidences
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = probs.max(dim=1)
            
            # Convert labels to numpy for filtering
            labels_np = batch_y.cpu().numpy()
            preds_np = predictions.cpu().numpy()
            confs_np = confidences.cpu().numpy()
            
            # Filter: only correct, high-confidence predictions
            valid_mask = (preds_np == labels_np) & (confs_np >= rule_threshold)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # Extract embedding features for sequence models, or flatten for MLP
                valid_batch_x = batch_x[valid_indices]
                valid_features = extract_embedding_features(model, valid_batch_x, args)
                valid_labels = labels_np[valid_indices]
                all_features.append(valid_features)
                all_labels.append(valid_labels)
    
    if not all_features:
        logging.warning("No valid samples for Helios rule extraction")
        return []
    
    # Concatenate all valid samples
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    logging.info(f"Collected {len(X)} high-confidence correct samples for Helios rule extraction")
    
    # Use HeliosSystem with Prototype Network
    try:
        from helios_model import HeliosSystem
        
        input_dim = X.shape[1]
        
        # Remap labels to compact 0..N-1 range for Helios
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        reverse_label_map = {new: old for new, old in enumerate(unique_labels)}
        
        y_mapped = np.array([label_map[label] for label in y])
        num_classes = len(unique_labels)
        
        # Determine device string
        device_str = str(DEVICE)
        
        helios_system = HeliosSystem(
            input_dim=input_dim,
            num_classes=num_classes,
            num_prototypes_per_class=200,  # More prototypes for better coverage
            max_rules=max_rules,
            radio=1.0,  # Tighter bounding boxes (was 1.2)
            boost_num=8,  # More boosting iterations (was 4)
            prune_threshold=2,  # Lower threshold to keep more rules (was 5)
            device=device_str
        )
        
        # Train prototype network and generate rules
        helios_system.fit(X, y_mapped, epochs=30)  # More epochs for better convergence
        
        logging.info(f"HeliosSystem extracted {len(helios_system.rules)} rules")
        
        # Verify coverage on training data
        coverage_count = 0
        correct_count = 0
        for i in range(len(X)):
            for rule in helios_system.rules:
                if rule.match(X[i]):
                    coverage_count += 1
                    if rule.class_label == y_mapped[i]:
                        correct_count += 1
                    break
        
        coverage_rate = coverage_count / len(X) if len(X) > 0 else 0
        accuracy_rate = correct_count / len(X) if len(X) > 0 else 0
        logging.info(f"Helios rules coverage on training data: {coverage_count}/{len(X)} ({coverage_rate:.1%})")
        logging.info(f"Helios rules accuracy on training data: {correct_count}/{len(X)} ({accuracy_rate:.1%})")
        
        # If coverage is too low, generate more rules
        if coverage_rate < 0.95:
            logging.warning(f"Coverage too low ({coverage_rate:.1%}), generating additional rules...")
            # Find uncovered samples
            uncovered_mask = np.ones(len(X), dtype=bool)
            for i in range(len(X)):
                for rule in helios_system.rules:
                    if rule.match(X[i]) and rule.class_label == y_mapped[i]:
                        uncovered_mask[i] = False
                        break
            
            X_uncovered = X[uncovered_mask]
            y_uncovered = y_mapped[uncovered_mask]
            
            if len(X_uncovered) > 0:
                # Use incremental update to add more rules
                helios_system.incremental_update(X_uncovered, y_uncovered)
                logging.info(f"After incremental update: {len(helios_system.rules)} rules")
        
        # Restore original class labels in rules
        for rule in helios_system.rules:
            if rule.class_label in reverse_label_map:
                rule.class_label = int(reverse_label_map[rule.class_label])

        # Convert rules to serializable format
        rules = helios_system.to_dict_list()
        
        # Save prototype network model
        proto_model_path = os.path.join(save_dir, f"helios_proto_{args.model_class}_{dataset_name}.pth")
        helios_system.save_model(proto_model_path)
        
    except Exception as e:
        logging.warning(f"HeliosSystem failed: {e}. Using fallback HeliosManager.")
        import traceback
        traceback.print_exc()
        
        # Fallback to HeliosManager (clustering-based)
        benchmarks_dir = Path(__file__).resolve().parents[2] / 'benchmarks'
        if str(benchmarks_dir) not in sys.path:
            sys.path.insert(0, str(benchmarks_dir))
        
        from helios_manager import HeliosManager
        
        helios_manager = HeliosManager(
            max_rules=max_rules, 
            radio=1.0,  # Tighter bounding boxes
            boost_num=8,
            prune_rule_threshold=2,
            use_prototype_network=False  # Use clustering fallback
        )
        helios_manager.fit(X, y)
        
        logging.info(f"HeliosManager extracted {len(helios_manager.rules)} rules")
        
        # Verify coverage
        coverage_count = 0
        correct_count = 0
        for i in range(len(X)):
            for rule in helios_manager.rules:
                if rule.match(X[i]):
                    coverage_count += 1
                    if rule.class_label == y[i]:
                        correct_count += 1
                    break
        
        coverage_rate = coverage_count / len(X) if len(X) > 0 else 0
        accuracy_rate = correct_count / len(X) if len(X) > 0 else 0
        logging.info(f"Helios rules coverage on training data: {coverage_count}/{len(X)} ({coverage_rate:.1%})")
        logging.info(f"Helios rules accuracy on training data: {correct_count}/{len(X)} ({accuracy_rate:.1%})")
        
        rules = [r.to_dict() for r in helios_manager.rules]
    
    # Log class distribution
    final_class_counts = {}
    for r in rules:
        c = r['class_label']
        final_class_counts[c] = final_class_counts.get(c, 0) + 1
    logging.info(f"Final Helios rules: {len(rules)}, class distribution: {final_class_counts}")
    
    # Determine feature type for metadata
    uses_embedding = args.model_type in ['cnn', 'rnn']
    feature_dim = X.shape[1] if len(X) > 0 else None
    if uses_embedding:
        logging.info(f"Rules use embedding features with dimension {feature_dim}")
    
    # Save rules to JSON file
    rules_path = os.path.join(save_dir, f"helios_rules_{args.model_class}_{dataset_name}_{timestamp}.json")
    with open(rules_path, 'w') as f:
        json.dump({
            'model_class': args.model_class,
            'dataset_name': dataset_name,
            'num_rules': len(rules),
            'class_distribution': final_class_counts,
            'rule_threshold': rule_threshold,
            'uses_embedding_features': uses_embedding,
            'embedding_feature_dim': feature_dim,
            'rules': rules
        }, f)
    
    logging.info(f"Helios rules saved to {rules_path}")
    
    # Also save a "best" version without timestamp for easy loading
    best_rules_path = os.path.join(save_dir, f"helios_rules_{args.model_class}_{dataset_name}.json")
    with open(best_rules_path, 'w') as f:
        json.dump({
            'model_class': args.model_class,
            'dataset_name': dataset_name,
            'num_rules': len(rules),
            'class_distribution': final_class_counts,
            'rule_threshold': rule_threshold,
            'uses_embedding_features': uses_embedding,
            'embedding_feature_dim': feature_dim,
            'rules': rules
        }, f)
    
    logging.info(f"Helios rules also saved to {best_rules_path}")
    
    return rules


if __name__ == "__main__":
    main()
