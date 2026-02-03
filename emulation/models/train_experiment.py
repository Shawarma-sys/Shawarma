# *************************************************************************
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *************************************************************************

import os
import sys
import subprocess
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

def _get_shawarma_home():
    env_home = os.getenv("Shawarma_HOME")
    if env_home:
        return env_home
    # Fallback to repository root if env var is missing (models -> emulation -> Shawarma)
    return str(Path(__file__).resolve().parents[2])

Shawarma_home = _get_shawarma_home()
os.environ.setdefault("Shawarma_HOME", Shawarma_home)

def run_train_experiment(model_type="mlp", model_class="MLP1", dataset_name="cic-ids-2018", 
                         train_data=None, train_labels=None, num_classes=None,
                         extract_rules="both"):
    """
    Train a model and extract LINC/Helios rules.
    
    Args:
        model_type: Type of model (mlp, cnn, rnn)
        model_class: Specific model class (MLP1, MLP1Teacher, TextCNN1, RNN1, etc.)
        dataset_name: Name of dataset for file naming
        train_data: Path to training data .npy file
        train_labels: Path to training labels .npy file
        num_classes: Number of classes (auto-detected from labels if not provided)
        extract_rules: Which rules to extract - "linc", "helios", "both", or "none" (default: "both")
    """
    if train_data is None:
        train_data = f"{Shawarma_home}/emulation/datasets/{dataset_name}-redistributed_X.npy"
    if train_labels is None:
        # Use binary labels for cic-ids-2018
        if 'cic-ids-2018' in dataset_name:
            train_labels = f"{Shawarma_home}/emulation/datasets/{dataset_name}-redistributed_y_binary.npy"
        else:
            train_labels = f"{Shawarma_home}/emulation/datasets/{dataset_name}-redistributed_y.npy"
    
    cmd = [
        sys.executable,
        f"{Shawarma_home}/emulation/src/models/train_unified.py",

        "--model_type",
        model_type,
        "--model_class",
        model_class,
        "--train_data",
        train_data,
        "--train_labels",
        train_labels,
    ]
    
    # Add num_classes if specified
    if num_classes is not None:
        cmd.extend(["--num_classes", str(num_classes)])
    
    # Add extract_rules parameter
    if extract_rules != "both":
        cmd.extend(["--extract_rules", extract_rules])

    print(f"Training {model_class} on {dataset_name}...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=Shawarma_home)

    return

def extract_rules_from_model(model_path, model_type="mlp", model_class="MLP1", 
                             dataset_name="cic-ids-2018", train_data=None, 
                             train_labels=None, num_classes=None, 
                             extract_rules="both", batch_size=256):
    """
    Extract LINC/Helios rules from an existing trained model.
    
    Args:
        model_path: Path to the trained model checkpoint (.pth file)
        model_type: Type of model (mlp, cnn, rnn)
        model_class: Specific model class (MLP1, TextCNN1, RNN1, etc.)
        dataset_name: Name of dataset for file naming
        train_data: Path to training data .npy file
        train_labels: Path to training labels .npy file
        num_classes: Number of classes (auto-detected from labels if not provided)
        extract_rules: Which rules to extract - "linc", "helios", or "both" (default: "both")
        batch_size: Batch size for processing data
    """
    # Add src to path
    src_path = f"{Shawarma_home}/emulation/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Import necessary modules
    from models.train_unified import extract_linc_rules, extract_helios_rules, ensure_dir
    from dataset.dataset import UnifiedTrafficDataset, traffic_collate_fn
    from torch.utils.data import DataLoader
    import logging
    import importlib
    
    # Import model classes dynamically
    if model_type == 'cnn':
        from models.cnnmodel import TextCNN1, TextCNN2
        model_module = sys.modules['models.cnnmodel']
    elif model_type == 'rnn':
        from models.rnnmodel import RNN1
        model_module = sys.modules['models.rnnmodel']
    else:  # mlp
        import models.mlp
        model_module = models.mlp
    
    # Setup device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup default paths
    if train_data is None:
        train_data = f"{Shawarma_home}/emulation/datasets/{dataset_name}-redistributed_X.npy"
    if train_labels is None:
        if 'cic-ids-2018' in dataset_name:
            train_labels = f"{Shawarma_home}/emulation/datasets/{dataset_name}-redistributed_y_binary.npy"
        else:
            train_labels = f"{Shawarma_home}/emulation/datasets/{dataset_name}-redistributed_y.npy"
    
    # Auto-detect num_classes
    if num_classes is None:
        labels_basename = os.path.basename(train_labels)
        is_binary = 'binary' in labels_basename.lower()
        if is_binary:
            num_classes = 2
        else:
            try:
                labels = np.load(train_labels)
                num_classes = len(np.unique(labels))
                print(f"Auto-detected {num_classes} classes from label file")
            except Exception as e:
                print(f"Warning: Could not read label file: {e}")
                num_classes = 7  # Default
    
    # Setup directories
    DEFAULT_MODEL_ROOT = os.path.join(Shawarma_home, 'emulation', 'models', 'checkpoint')
    DEFAULT_LOG_ROOT = os.path.join(Shawarma_home, 'emulation', 'logs', 'train')
    save_dir = os.path.join(DEFAULT_MODEL_ROOT, model_type)
    log_dir = os.path.join(DEFAULT_LOG_ROOT, model_type)
    ensure_dir(save_dir)
    ensure_dir(log_dir)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"extract_rules_{dataset_name}_{model_class}_{timestamp}.log")
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    
    logging.info(f"Extracting rules from existing model: {model_path}")
    logging.info(f"Model type: {model_type}, Model class: {model_class}")
    logging.info(f"Dataset: {dataset_name}, Extract rules: {extract_rules}")
    
    # Load dataset
    window_size = 9  # Default for sequence models
    full_dataset = UnifiedTrafficDataset(
        train_data, 
        train_labels, 
        sequence_length=window_size if model_type in ['cnn', 'rnn'] else None,
        standardize=True if model_type == 'mlp' else False,
        normalize=False,
        device=DEVICE
    )
    
    # Use first 160000 samples (same as training)
    subset_size = min(160000, len(full_dataset))
    full_dataset = torch.utils.data.Subset(full_dataset, range(subset_size))
    logging.info(f"Loaded dataset with {len(full_dataset)} samples")
    
    train_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=traffic_collate_fn
    )
    
    # Initialize model
    if model_type == 'cnn':
        model_cls = getattr(model_module, model_class)
        model = model_cls(
            input_size=4, 
            num_classes=num_classes,
            len_vocab=1501, 
            ipd_vocab=2561,
            len_embedding_bits=10, 
            ipd_embedding_bits=8,
            nk=4, 
            ebdin=4, 
            device=DEVICE
        ).to(DEVICE)
    elif model_type == 'rnn':
        model_cls = getattr(model_module, model_class)
        model = model_cls(
            rnn_in=12, 
            hidden_size=16, 
            labels_num=num_classes,
            len_vocab=1501, 
            ipd_vocab=2561,
            len_embedding_bits=10, 
            ipd_embedding_bits=8,
            device=DEVICE, 
            droprate=0.0
        ).to(DEVICE)
    else:  # mlp
        model_cls = getattr(model_module, model_class)
        import inspect
        sig = inspect.signature(model_cls.__init__)
        
        if 'num_classes' in sig.parameters:
            model = model_cls(num_classes=num_classes).to(DEVICE)
        else:
            model = model_cls().to(DEVICE)
    
    # Load model weights
    logging.info(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Create args object for rule extraction functions
    class Args:
        pass
    
    args = Args()
    args.model_type = model_type
    args.model_class = model_class
    args.window_size = window_size
    args.num_classes = num_classes
    
    # Extract rules
    if extract_rules in ['linc', 'both']:
        logging.info("=" * 60)
        logging.info("Extracting LINC rules...")
        extract_linc_rules(model, train_loader, args, save_dir, dataset_name, timestamp)
    
    if extract_rules in ['helios', 'both']:
        logging.info("=" * 60)
        logging.info("Extracting Helios rules...")
        extract_helios_rules(model, train_loader, args, save_dir, dataset_name, timestamp)
    
    logging.info("=" * 60)
    logging.info(f"Rule extraction complete! Check {save_dir} for output files.")
    print(f"\n✓ Rule extraction complete! Check {save_dir} for output files.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model or extract rules from existing model")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train", choices=["train", "extract"],
                        help="Mode: 'train' to train new model, 'extract' to extract rules from existing model")
    
    # Common arguments
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "cnn", "rnn"],
                        help="Type of model to train")
    parser.add_argument("--model_class", type=str, default="MLP1",
                        help="Specific model class (MLP1, MLP1Teacher, TextCNN1, RNN1, etc.)")
    parser.add_argument("--dataset_name", type=str, default="cic-ids-2018",
                        help="Name of dataset")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data .npy file")
    parser.add_argument("--train_labels", type=str, default=None,
                        help="Path to training labels .npy file")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (auto-detected from labels if not provided)")
    parser.add_argument("--extract_rules", type=str, default="both", 
                        choices=["linc", "helios", "both", "none"],
                        help="Which rules to extract: linc, helios, both, or none (default: both)")
    
    # Extract mode specific arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint (required for extract mode)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for rule extraction (extract mode only)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Training mode
        run_train_experiment(
            model_type=args.model_type,
            model_class=args.model_class,
            dataset_name=args.dataset_name,
            train_data=args.train_data,
            train_labels=args.train_labels,
            num_classes=args.num_classes,
            extract_rules=args.extract_rules,
        )
    else:
        # Extract mode
        if args.model_path is None:
            # Try to find the best model automatically
            DEFAULT_MODEL_ROOT = os.path.join(Shawarma_home, 'emulation', 'models', 'checkpoint')
            model_dir = os.path.join(DEFAULT_MODEL_ROOT, args.model_type)
            model_filename = f"best_{args.model_class}_{args.dataset_name}.pth"
            args.model_path = os.path.join(model_dir, model_filename)
            
            if not os.path.exists(args.model_path):
                print(f"Error: Model not found at {args.model_path}")
                print("Please specify --model_path explicitly")
                sys.exit(1)
            else:
                print(f"Using model: {args.model_path}")
        
        extract_rules_from_model(
            model_path=args.model_path,
            model_type=args.model_type,
            model_class=args.model_class,
            dataset_name=args.dataset_name,
            train_data=args.train_data,
            train_labels=args.train_labels,
            num_classes=args.num_classes,
            extract_rules=args.extract_rules,
            batch_size=args.batch_size,
        )
