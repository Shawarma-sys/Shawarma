# Data Plane (Client)
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
from pathlib import Path
import logging
import sys

# Add src to sys.path to allow imports from models, dataset, utils
# This file is in emulation/benchmarks/
# src is in emulation/src/
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from utils.metrics import compute_acc
from dataset.dataset import UnifiedTrafficDataset, traffic_collate_fn
from models.mlp import *
from models.cnnmodel import TextCNN1, TextCNN2
from models.rnnmodel import RNN1
import json
from queue import Queue
from threading import Event
import grpc
import control_plane_pb2
import control_plane_pb2_grpc
import copy
import argparse
import time
import io
from concurrent import futures
import threading
import asyncio
import torchlens as tl
from utils.model_utils import *
import numpy as np
import random


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set global RNG seeds for reproducibility."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _get_shawarma_home():
    env_home = os.getenv("Shawarma_HOME")
    if env_home:
        return env_home
    # Fallback to repository root if env var is missing (benchmarks -> emulation -> Shawarma)
    return str(Path(__file__).resolve().parents[2])


Shawarma_home = _get_shawarma_home()
os.environ.setdefault("Shawarma_HOME", Shawarma_home)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

class DataPlaneClient:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = control_plane_pb2_grpc.ControlPlaneStub(self.channel)
        self.sequence_models = {"TextCNN1", "TextCNN2", "RNN1"}

        self.sequence_length = args.sequence_length
        self.sequence_feature_dim = args.sequence_feature_dim
        self.sequence_dataset_root = args.sequence_dataset_root
        self.sequence_dataset_name = args.sequence_dataset_name
        self.sequence_dataset_split = args.sequence_dataset_split

        # LINC mode support
        self.linc_mode = False  # Will be set to True when rules are received
        self.linc_rules = []  # List of LINC rules
        self._compiled_rules = None # Compiled rules for fast inference
        self._linc_lock = threading.Lock()  # Thread-safe access to rules

        self.my_dnn, dummy_input = self._init_model(args.model_class, args.model_input_shape)
        if args.base_model_path:
            self.my_dnn.load_state_dict(torch.load(args.base_model_path))
        self.my_dnn.eval()
        
        # Save a reference to the original model for embedding extraction (before torchlens conversion)
        # This is needed for LINC/Helios to extract meaningful features from sequence models
        if args.model_class in self.sequence_models:
            self._original_model_for_embedding = copy.deepcopy(self.my_dnn)
            self._original_model_for_embedding.eval()
            logging.info(f"Saved original {args.model_class} model for embedding extraction")
        
        # Keep buffers (e.g., BatchNorm running stats) available for forward_eval replay.
        # Note: passing layers_to_save=None can cause TorchLens to omit buffer contents.
        my_dnn_history = tl.log_forward_pass(self.my_dnn, dummy_input, vis_save_only=True)
        self.my_original_dnn = my_dnn_history.layer_list
        self.label_index = label2index(self.my_original_dnn)
        self.max_partition_index = get_max_partition_index(self.my_original_dnn)
        self.partition_index = 0  # self.max_partition_index-1
        self._retraining_executor = futures.ThreadPoolExecutor(max_workers=getattr(args, "retraining_workers", 4))
        self._retraining_shutdown = threading.Event()
        
        # Load pre-trained LINC rules if path is provided
        linc_rules_path = getattr(args, 'linc_rules_path', None)
        if linc_rules_path:
            self._load_linc_rules(linc_rules_path)

    def _load_linc_rules(self, rules_path):
        """Load pre-trained LINC rules from JSON file."""
        if not os.path.exists(rules_path):
            logging.warning(f"LINC rules file not found: {rules_path}")
            return
        
        try:
            with open(rules_path, 'r') as f:
                rules_data = json.load(f)
            
            rules = []
            for rule in rules_data.get('rules', []):
                rules.append({
                    'pattern': np.array(rule['pattern'], dtype=np.float32),
                    'important_features': rule['important_features'],
                    'class': rule['class'],
                    'confidence': rule['confidence']
                })
            
            # Update rules using the existing method
            self.update_linc_rules(rules)
            logging.info(f"Loaded {len(rules)} pre-trained LINC rules from {rules_path}")
        except Exception as e:
            logging.error(f"Failed to load LINC rules from {rules_path}: {e}")

    def _validate_sequence_args(self):
        if self.args.num_classes is None:
            raise ValueError("--num_classes must be provided for sequential models.")
        if not self.sequence_length or not self.sequence_feature_dim:
            raise ValueError("--sequence_length and --sequence_feature_dim are required for sequential models.")

    def _build_dummy_sequence(self, batch_size=1):
        seq_shape = (batch_size, self.sequence_length, self.sequence_feature_dim)
        dummy = torch.zeros(seq_shape, dtype=torch.long, device=self.device)
        len_vocab = max(1, int(self.args.len_vocab))
        ipd_vocab = max(1, int(self.args.ipd_vocab))
        dummy[..., 0] = torch.randint(0, len_vocab, dummy[..., 0].shape, device=self.device)
        if self.sequence_feature_dim > 1:
            dummy[..., 1] = torch.randint(0, ipd_vocab, dummy[..., 1].shape, device=self.device)
        return dummy

    def _get_dataset(self):
        """
        Unified dataset loader for both sequence and flat models.
        """
        dataset_root = self.args.list_flow_input_csv_file_path[0] if self.args.list_flow_input_csv_file_path else ""
        
        # Case 1: Explicit feature/label files provided
        if self.args.feature_file and self.args.label_file:
            data_file = self.args.feature_file
            labels_file = self.args.label_file
            
            if not os.path.isabs(data_file):
                data_file = os.path.join(dataset_root, data_file)
            if not os.path.isabs(labels_file):
                labels_file = os.path.join(dataset_root, labels_file)
                
            if os.path.exists(data_file) and os.path.exists(labels_file):
                logging.info(f"Loading dataset from explicit files: {data_file}, {labels_file}")
                return UnifiedTrafficDataset(
                    data_file=data_file,
                    labels_file=labels_file,
                    sequence_length=self.sequence_length if self.args.model_class in self.sequence_models else None,
                    standardize=True if self.args.model_class not in self.sequence_models else False,
                    device=self.device
                )

        # Case 2: Sequence-specific dataset name (fallback for legacy sequence experiments)
        if self.args.model_class in self.sequence_models:
            dataset_name = self.args.sequence_dataset_name
            if not dataset_name:
                raise ValueError("--sequence_dataset_name or --feature_file/--label_file is required for sequential models.")
            
            split = self.args.sequence_dataset_split or "test"
            base_root = self.args.sequence_dataset_root or os.path.join(Shawarma_home, "emulation", "datasets")
            dataset_base = os.path.join(base_root, dataset_name)
            split_dir = os.path.join(dataset_base, split)
            data_file = os.path.join(split_dir, f"{dataset_name}_{split}_data.npy")
            labels_file = os.path.join(split_dir, f"{dataset_name}_{split}_labels.npy")
            
            return UnifiedTrafficDataset(
                data_file=data_file,
                labels_file=labels_file,
                sequence_length=self.sequence_length,
                device=self.device
            )
        
        raise ValueError("Could not determine dataset files. Please provide --feature_file and --label_file.")

    def _init_model(self, model_class, model_input_shape):
        if model_class in ["TextCNN1", "TextCNN2"]:
            self._validate_sequence_args()
            cnn_class = TextCNN1 if model_class == "TextCNN1" else TextCNN2
            model = cnn_class(
                input_size=self.sequence_feature_dim,
                num_classes=self.args.num_classes,
                len_vocab=self.args.len_vocab,
                ipd_vocab=self.args.ipd_vocab,
                len_embedding_bits=self.args.len_embedding_bits,
                ipd_embedding_bits=self.args.ipd_embedding_bits,
                nk=self.args.textcnn_nk,
                ebdin=self.args.textcnn_ebdin,
                device=self.device,
            ).to(self.device)
            dummy = self._build_dummy_sequence(batch_size=int(getattr(self.args, "batch_size", 1) or 1))
            return model, dummy
        if model_class == "RNN1":
            self._validate_sequence_args()
            model = RNN1(
                rnn_in=self.args.rnn_in,
                hidden_size=self.args.rnn_hidden,
                labels_num=self.args.num_classes,
                len_vocab=self.args.len_vocab,
                ipd_vocab=self.args.ipd_vocab,
                len_embedding_bits=self.args.len_embedding_bits,
                ipd_embedding_bits=self.args.ipd_embedding_bits,
                device=self.device,
                droprate=self.args.rnn_dropout,
            ).to(self.device)
            dummy = self._build_dummy_sequence(batch_size=int(getattr(self.args, "batch_size", 1) or 1))
            return model, dummy
        my_dnn_class = globals()[model_class]
        model = my_dnn_class(input_shape=model_input_shape).to(self.device) if model_input_shape else my_dnn_class().to(self.device)
        feature_dim = model_input_shape if model_input_shape else 16
        dummy = torch.zeros((1, feature_dim), device=self.device)
        return model, dummy

    def _prepare_features(self, features):
        if self.args.model_class in self.sequence_models:
            if features.dim() == 2:
                expected = self.sequence_length * self.sequence_feature_dim
                if features.shape[1] != expected:
                    raise ValueError(
                        f"Expected flattened features with width {expected} but received {features.shape[1]}"
                    )
                features = features.view(features.shape[0], self.sequence_length, self.sequence_feature_dim)
            return features.to(device=self.device, dtype=torch.long)
        return features.to(self.device)

    def _compile_rules(self, parsed_rules):
        """
        Compile parsed rules into tensor format for vectorized inference.
        Supports only rules with Op 0 (<) and Op 1 (>=).
        Fallback to None if complex ops (==, !=) are found.
        """
        if not parsed_rules:
            return None

        try:
            # Determine feature dimension
            max_idx = 0
            for r in parsed_rules:
                for idx, _, op in r['conditions']:
                    if op not in (0, 1):
                        # Complex op found, disable optimization
                        return None
                    max_idx = max(max_idx, int(idx))
            
            feature_dim = max_idx + 1
            num_rules = len(parsed_rules)
            
            if feature_dim == 0:
                return None

            # Initialize bounds: Lower = -inf, Upper = +inf
            lower_bounds = torch.full((num_rules, feature_dim), -float('inf'), device=self.device)
            upper_bounds = torch.full((num_rules, feature_dim), float('inf'), device=self.device)
            classes = torch.zeros(num_rules, dtype=torch.long, device=self.device)
            
            for i, r in enumerate(parsed_rules):
                classes[i] = int(r['class'])
                for idx, thresh, op in r['conditions']:
                    idx = int(idx)
                    val = float(thresh)
                    if op == 0: # < thresh -> Upper Bound
                        if val < upper_bounds[i, idx]:
                            upper_bounds[i, idx] = val
                    elif op == 1: # >= thresh -> Lower Bound
                        if val > lower_bounds[i, idx]:
                            lower_bounds[i, idx] = val
                            
            logging.info(f"Compiled {num_rules} rules into tensors for fast inference")
            return (lower_bounds, upper_bounds, classes)
            
        except Exception as e:
            logging.warning(f"Failed to compile rules for fast inference: {e}")
            return None

    def update_linc_rules(self, rules):
        """
        Update LINC rules from control plane.
        Uses copy-on-write pattern to avoid blocking predictions during update.
        """
        parsed_rules = []
        if rules:
            for r in rules:
                if isinstance(r, dict):
                    threshs = r.get('pattern', [])
                    imp_feats = r.get('important_features', [])
                    lbl = r.get('class')
                    conf = r.get('confidence')
                else:
                    threshs = list(r.pattern)
                    imp_feats = list(r.important_features)
                    lbl = r.class_label
                    conf = r.confidence
                
                conditions = []
                # Ensure length matching
                num_conds = len(threshs)
                if len(imp_feats) >= 2 * num_conds:
                    for i in range(num_conds):
                        idx = imp_feats[2*i]
                        op = imp_feats[2*i+1]
                        thresh = threshs[i]
                        conditions.append((idx, thresh, op))
                
                parsed_rules.append({
                    'conditions': conditions,
                    'class': lbl,
                    'confidence': conf
                })
        
        # Compile rules off critical path
        compiled = self._compile_rules(parsed_rules)
        
        # Atomic swap - minimal lock time
        with self._linc_lock:
            self.linc_rules = parsed_rules
            self._compiled_rules = compiled
            self.linc_mode = True
        logging.info(f"Updated LINC rules: {len(parsed_rules)} rules received")

    def _extract_embedding_features(self, features_tensor):
        """
        Extract embedding features from sequence models for rule-based prediction.
        
        For sequence models (RNN1, TextCNN1, etc.), raw token IDs are not suitable
        for threshold-based rules. We extract embedding representations which are
        continuous and meaningful for rule matching.
        
        OPTIMIZED: For RNN1, extract the final hidden state which contains richer
        sequential information than just mean-pooled embeddings. This must match
        the feature extraction used during rule training (train_unified.py) and
        control plane (control_plane_retrain_torchlens.py).
        
        Args:
            features_tensor: Input tensor of shape (batch_size, seq_len, feature_dim)
        
        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        if self.args.model_class not in self.sequence_models:
            # Non-sequence model: flatten and return
            if torch.is_tensor(features_tensor):
                return features_tensor.cpu().numpy().reshape(features_tensor.shape[0], -1)
            return np.array(features_tensor).reshape(len(features_tensor), -1)
        
        with torch.no_grad():
            features_tensor = features_tensor.to(self.device)
            if features_tensor.dtype != torch.long:
                features_tensor = features_tensor.long()
            
            # Extract length and IPD tokens
            len_x = features_tensor[:, :, 0]  # (batch_size, seq_len)
            ipd_x = features_tensor[:, :, 1]  # (batch_size, seq_len)
            
            # Use the saved original model for embedding extraction (preferred)
            if hasattr(self, '_original_model_for_embedding') and self._original_model_for_embedding is not None:
                original_model = self._original_model_for_embedding
                original_model.eval()
                
                # Clamp inputs to valid vocabulary range
                if hasattr(original_model, 'len_embedding'):
                    len_vocab = original_model.len_embedding.num_embeddings
                    len_x = torch.clamp(len_x, 0, len_vocab - 1)
                if hasattr(original_model, 'ipd_embedding'):
                    ipd_vocab = original_model.ipd_embedding.num_embeddings
                    ipd_x = torch.clamp(ipd_x, 0, ipd_vocab - 1)
                
                # Use the trained embedding layers from the original model
                len_embedded = original_model.len_embedding(len_x)  # (batch_size, seq_len, len_emb_bits)
                ipd_embedded = original_model.ipd_embedding(ipd_x)  # (batch_size, seq_len, ipd_emb_bits)
                
                # Concatenate embeddings
                combined = torch.cat([len_embedded, ipd_embedded], dim=-1)
                
                # OPTIMIZED: For RNN1, extract only embedding features (Input Projection)
                if self.args.model_class == "RNN1" and hasattr(original_model, 'rnn') and hasattr(original_model, 'fc1'):
                    # Modified: Extract only embedding features, using FC1 projection, then FLATTEN (no pooling)
                    # Apply fc1 transformation: (batch_size, seq_len, rnn_in)
                    x = original_model.fc1(combined)
                    
                    # Flatten the sequence: (batch_size, seq_len * rnn_in)
                    features = x.reshape(x.shape[0], -1)
                    
                    return features.cpu().numpy()
                
                elif self.args.model_class in ["TextCNN1", "TextCNN2"] and hasattr(original_model, 'conv'):
                    # For TextCNN, apply convolution and pooling
                    x = original_model.fc1(combined) if hasattr(original_model, 'fc1') else combined
                    
                    # Apply convolution layers
                    x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
                    conv_out = original_model.conv(x)
                    # Global max pooling
                    pooled = torch.max(conv_out, dim=2)[0]
                    return pooled.cpu().numpy()
                
                # Fallback: use mean and max pooling with fc1 transformation
                if hasattr(original_model, 'fc1'):
                    combined = original_model.fc1(combined)
                
                # Aggregate across sequence: use both mean and max pooling
                mean_pooled = combined.mean(dim=1)
                max_pooled = combined.max(dim=1)[0]
                
                # Concatenate mean and max pooled features
                features = torch.cat([mean_pooled, max_pooled], dim=-1)
                
                return features.cpu().numpy()
            
            # Fallback: create embedding layers (less effective without trained weights)
            logging.warning("Original model not available for embedding extraction, using cached embeddings")
            
            len_vocab = getattr(self.args, 'len_vocab', 1501)
            ipd_vocab = getattr(self.args, 'ipd_vocab', 2561)
            len_emb_bits = getattr(self.args, 'len_embedding_bits', 10)
            ipd_emb_bits = getattr(self.args, 'ipd_embedding_bits', 8)
            
            # Create embedding layers (or reuse if cached)
            if not hasattr(self, '_len_emb_cache'):
                self._len_emb_cache = torch.nn.Embedding(len_vocab, len_emb_bits).to(self.device)
                self._ipd_emb_cache = torch.nn.Embedding(ipd_vocab, ipd_emb_bits).to(self.device)
            
            # Apply embeddings
            len_embedded = self._len_emb_cache(len_x)
            ipd_embedded = self._ipd_emb_cache(ipd_x)
            
            # Concatenate and pool
            combined = torch.cat([len_embedded, ipd_embedded], dim=-1)
            pooled = combined.mean(dim=1)
            
            return pooled.cpu().numpy()

    def linc_predict(self, features):
        """
        Predict using LINC/Helios rules with DNN fallback for unmatched samples.
        
        For Helios rules (interval-based), conditions are:
        - (feature_idx, min_val - epsilon, 1) meaning feature > (min_val - epsilon)
        - (feature_idx, max_val, 0) meaning feature <= max_val
        
        For LINC rules (tree-based), conditions are:
        - (feature_idx, threshold, op) where op=0 means <=, op=1 means >
        
        For sequence models (RNN1, TextCNN1), features are first converted to
        embedding representations before rule matching.
        
        Uses snapshot of rules to avoid blocking updates during prediction.
        First matching rule wins (matching original Helios logic).
        
        Returns:
            Tuple of (predictions, unmatched_mask) where unmatched_mask indicates
            samples that need DNN fallback, or None if no rules available.
        """
        # Quick check without lock
        if not getattr(self, 'linc_mode', False):
            return None
        
        # Take a snapshot of rules (atomic read)
        with self._linc_lock:
            rules_snapshot = self.linc_rules
            compiled_rules_snapshot = self._compiled_rules
            if not rules_snapshot:
                return None
        
        # For sequence models, extract embedding features
        if self.args.model_class in self.sequence_models:
            features_np = self._extract_embedding_features(features)
        else:
            # All prediction logic runs WITHOUT holding the lock
            if torch.is_tensor(features):
                features_np = features.cpu().numpy()
            else:
                features_np = np.array(features)
            
            batch_size = features_np.shape[0]
            # Ensure 2D
            if len(features_np.shape) > 2:
                features_np = features_np.reshape(batch_size, -1)
        
        # --- Fast Path: Vectorized Inference ---
        if compiled_rules_snapshot is not None:
            lower_bounds, upper_bounds, rule_classes = compiled_rules_snapshot
            
            # Prepare features tensor
            if isinstance(features_np, np.ndarray):
                features_t = torch.from_numpy(features_np).to(self.device).float()
            else:
                features_t = features_np.to(self.device).float()
            
            # Handle dimension mismatch
            feat_dim = features_t.shape[1]
            rule_dim = lower_bounds.shape[1]
            
            # Pad or slice to match rule dimensions
            if feat_dim < rule_dim:
                # Pad features with 0 (or careful choice? 0 might trigger rules incorrectly)
                # If feature is missing, it's 0.
                padding = torch.zeros(features_t.shape[0], rule_dim - feat_dim, device=self.device)
                features_t = torch.cat([features_t, padding], dim=1)
            elif feat_dim > rule_dim:
                # Slice features
                features_t = features_t[:, :rule_dim]
                
            # Broadcasting: (Batch, 1, Feat) vs (1, Rules, Feat)
            # Use chunks to save memory if batch * rules is large
            batch_size = features_t.shape[0]
            num_rules = lower_bounds.shape[0]
            
            # Result buffer
            preds = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            unmatched_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            
            # Chunking rules to avoid OOM with large rule sets * large batches
            chunk_size = 1000 # Process 1000 rules at a time
            
            # We need to find the FIRST match.
            # So we process chunks. If a sample is matched by a previous chunk, we skip it?
            # Or we find matches in chunk, and fill.
            
            # Better approach: Process all rules in chunks, accumulate "first match".
            # Indices of rules are 0..N.
            # We want min(rule_idx) where match is True.
            
            # Initialize min_match_idx with infinity
            min_match_idx = torch.full((batch_size,), num_rules, dtype=torch.long, device=self.device)
            
            X_expanded = features_t.unsqueeze(1) # (Batch, 1, Feat)
            
            for i in range(0, num_rules, chunk_size):
                end = min(i + chunk_size, num_rules)
                
                # Slicing rules
                lb_chunk = lower_bounds[i:end].unsqueeze(0) # (1, Chunk, Feat)
                ub_chunk = upper_bounds[i:end].unsqueeze(0)
                
                # Check (Batch, Chunk, Feat) -> (Batch, Chunk)
                # Optimize: Check Lower first, then Upper.
                match_chunk = (X_expanded >= lb_chunk).all(dim=2) & (X_expanded < ub_chunk).all(dim=2)
                
                # Find matches in this chunk
                # any() check per sample
                has_match_chunk = match_chunk.any(dim=1)
                
                if has_match_chunk.any():
                    # Find first match index relative to chunk start
                    # Argmax returns first True
                    rel_idx = match_chunk.int().argmax(dim=1)
                    abs_idx = rel_idx + i
                    
                    # Update min_match_idx only if (found match in this chunk) AND (current min > abs_idx)
                    # Actually, if we process chunks in order 0..N, the first time we see a match for a sample, it IS the first match.
                    # So we only update if min_match_idx is still 'num_rules'
                    
                    update_mask = has_match_chunk & (min_match_idx == num_rules)
                    min_match_idx[update_mask] = abs_idx[update_mask]
            
            matched_mask = (min_match_idx < num_rules)
            unmatched_mask = ~matched_mask
            
            if matched_mask.any():
                preds[matched_mask] = rule_classes[min_match_idx[matched_mask]]
                
            return preds, unmatched_mask
            
        # --- Slow Path: Iterative Loop (Fallback) ---
        batch_size = features_np.shape[0]
        preds = np.zeros(batch_size, dtype=np.int64)
        unmatched_mask = np.ones(batch_size, dtype=bool)
        
        # First matching rule wins (matching original Helios logic)
        for rule in rules_snapshot:
            if not unmatched_mask.any():
                break
            
            candidate_mask = unmatched_mask.copy()
            for idx, thresh, op in rule['conditions']:
                if not candidate_mask.any():
                    break
                # Handle case where rule feature index exceeds embedding dimension
                if int(idx) >= features_np.shape[1]:
                    candidate_mask[:] = False
                    break
                col = features_np[:, int(idx)]
                if op == 0:  # <
                    candidate_mask &= (col < thresh)
                elif op == 1:  # >=
                    candidate_mask &= (col >= thresh)
                elif op == 2:  # ==
                    candidate_mask &= (col == thresh)
                elif op == 3:  # !=
                    candidate_mask &= (col != thresh)
                else:  # Fallback
                    candidate_mask &= (col > thresh)
            
            if candidate_mask.any():
                preds[candidate_mask] = rule['class']
                unmatched_mask[candidate_mask] = False
        
        # Return predictions and unmatched mask for DNN fallback
        return (torch.tensor(preds, device=self.device, dtype=torch.long), 
                torch.tensor(unmatched_mask, device=self.device, dtype=torch.bool))

    def send_data(self, features, labels, generated_labels, window_index):
        request = control_plane_pb2.DataRequest(
            features=features.tolist(),
            labels=labels.tolist(),
            window_index=window_index
        )
        self.stub.ReceiveData(request)

    def warmup_gpu(self):
        logging.info("Requesting GPU warm-up via gRPC...")
        try:
            self.stub.WarmupGPU(control_plane_pb2.Empty())
            logging.info("GPU warm-up completed via gRPC.")
        except Exception as e:
            logging.error(f"GPU warm-up failed: {e}")

    def _send_retraining_request(self, window_index, features_tensor, labels_tensor, my_window_f1: float, ground_truth_labels_tensor=None):
        if self._retraining_shutdown.is_set():
            logging.debug("Skipping retraining request for window %s because shutdown has started.", window_index)
            return
        try:
            logging.info(f"Requesting retraining for window index {window_index}...")
            request = control_plane_pb2.RetrainingRequest(
                window_index=int(window_index),
                features=control_plane_pb2.Tensor(
                    data=features_tensor.flatten().tolist(),
                    shape=list(features_tensor.shape)
                ),
                my_labels=labels_tensor.tolist(),
                my_window_f1=float(my_window_f1),
                ground_truth_labels=ground_truth_labels_tensor.tolist() if ground_truth_labels_tensor is not None else [],
            )
            self.stub.RequestRetraining(request)
            logging.info("Retraining request sent with online learning features.")
        except grpc.RpcError as e:
            if self._retraining_shutdown.is_set():
                logging.debug("Retraining RPC cancelled during shutdown for window %s: %s", window_index, e)
            else:
                logging.error(f"Retraining request failed: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while sending retraining request: {e}")

    def request_retraining(self, window_index, online_learning_features, online_learning_labels, my_window_f1: float, ground_truth_labels=None):
        if self._retraining_shutdown.is_set():
            logging.debug("Skipping retraining enqueue for window %s due to shutdown.", window_index)
            return
        features_tensor = online_learning_features.detach().cpu()
        labels_tensor = online_learning_labels.detach().cpu()
        ground_truth_labels_tensor = ground_truth_labels.detach().cpu() if ground_truth_labels is not None else None
        try:
            self._retraining_executor.submit(
                self._send_retraining_request,
                int(window_index),
                features_tensor,
                labels_tensor,
                float(my_window_f1),
                ground_truth_labels_tensor,
            )
        except RuntimeError as e:
            logging.debug("Executor rejected retraining task for window %s: %s", window_index, e)
            self._send_retraining_request(int(window_index), features_tensor, labels_tensor, float(my_window_f1), ground_truth_labels_tensor)

    def wait_for_pending_retraining(self):
        if self._retraining_executor:
            self._retraining_shutdown.set()
            self._retraining_executor.shutdown(wait=True)
            self._retraining_executor = None
            logging.info("Completed all pending retraining requests.")

    def run_inference(self):
        self.warmup_gpu()
        batch_size = self.args.batch_size

        # Load dataset using unified method
        dataset = self._get_dataset()
        streaming_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=traffic_collate_fn,
            drop_last=(self.args.model_class in self.sequence_models)
        )

        self.my_dnn = new_model(self.my_original_dnn, partition_index=-1)
        static_dnn = new_model(self.my_dnn, partition_index=-1)
        static_dnn = eval_mode(static_dnn)
        
        self.my_dnn = frozen_model(self.my_dnn, partition_index=self.partition_index)
        self.my_dnn = eval_mode(self.my_dnn)
        my_model_labels_list, static_model_labels_list, gt_labels_list = [], [], []
        online_learning_features_list, online_learning_labels_list = [], []
        experiment_data = {
            "labeling_window_size": self.args.eval_frequency,
            "my_f1s": [],
            "static_f1s": [],
            "my_accuracy": [],
            "static_accuracy": [],
            "my_improvement": 0.0,
        }
        for k, (features, gt_label) in enumerate(streaming_dataloader):
            features = self._prepare_features(features)
            
            # Use LINC/Helios rules for prediction if available
            # No DNN fallback - rules should cover all samples
            if self.linc_mode and self.linc_rules:
                linc_result = self.linc_predict(features)
                if linc_result is not None:
                    linc_preds, unmatched_mask = linc_result
                    my_label = linc_preds
                    # Log if there are unmatched samples (for debugging)
                    unmatched_count = unmatched_mask.sum().item()
                    if unmatched_count > 0 and k % 100 == 0:
                        logging.debug(f"Batch {k}: {unmatched_count}/{len(features)} samples unmatched by rules")
                else:
                    my_score = forward_eval(self.my_dnn, self.label_index, features)
                    my_label = torch.argmax(my_score, dim=1)
            else:
                my_score = forward_eval(self.my_dnn, self.label_index, features)
                my_label = torch.argmax(my_score, dim=1)

            online_learning_features_list.append(features)
            online_learning_labels_list.append(my_label)

            static_score = forward_eval(static_dnn, self.label_index, features)
            static_label = torch.argmax(static_score, dim=1)
            my_model_labels_list.append(my_label)
            static_model_labels_list.append(static_label)
            gt_labels_list.append(gt_label)
            # time.sleep(0.0005) # simulate familiar traffic
            if (k + 1) * batch_size % self.args.eval_frequency == 0:
                window_index = (k + 1) * batch_size / self.args.eval_frequency
                my_model_labels = torch.cat(my_model_labels_list, 0)
                static_model_labels = torch.cat(static_model_labels_list, 0)
                gt_labels = torch.cat(gt_labels_list, 0)
                my_window_f1 = f1_score(gt_labels.cpu(), my_model_labels.cpu(), average='macro')
                my_window_accuracy = compute_acc(my_model_labels, gt_labels).item()
                static_window_f1 = f1_score(gt_labels.cpu(), static_model_labels.cpu(), average='macro')
                static_window_accuracy = compute_acc(static_model_labels, gt_labels).item()
                experiment_data["my_accuracy"].append(my_window_accuracy)
                experiment_data["my_f1s"].append(my_window_f1)
                experiment_data['static_accuracy'].append(static_window_accuracy)
                experiment_data["static_f1s"].append(static_window_f1)
                improvement = max(0.0, my_window_f1 - static_window_f1)
                experiment_data["my_improvement"] += improvement
                logging.info(f"Window {window_index}: My F1 = {my_window_f1}, Static F1 = {static_window_f1}, My Acc = {my_window_accuracy}, Static Acc = {static_window_accuracy}, Improvement = {improvement}")
                online_learning_features = torch.cat(online_learning_features_list, 0)
                online_learning_labels = torch.cat(online_learning_labels_list, 0)
                self.request_retraining(window_index, online_learning_features, online_learning_labels, my_window_f1, gt_labels)
                logging.info(f"Retraining requested for the last window index {window_index}.")
                my_model_labels_list.clear()
                static_model_labels_list.clear()
                gt_labels_list.clear()
                online_learning_features_list.clear()
                online_learning_labels_list.clear()
        self.wait_for_pending_retraining()
        try:
            self.stub.RecordExperimentData(control_plane_pb2.ExperimentDataRequest(
                my_improvement=experiment_data["my_improvement"],
                my_f1s=experiment_data["my_f1s"],
                static_f1s=experiment_data["static_f1s"],
                my_accuracy=experiment_data["my_accuracy"],
                static_accuracy=experiment_data["static_accuracy"],
                window_index=int(self.args.eval_frequency)
            ))
            logging.info("All experiment data sent to control plane.")
        except Exception as e:
            logging.error(f"Failed to send experiment data: {e}")
        if getattr(self.args, "shutdown_control_plane", False):
            try:
                self.stub.Shutdown(control_plane_pb2.Empty())
                logging.info("Sent shutdown request to control plane.")
            except Exception as e:
                logging.error(f"Failed to shutdown control plane: {e}")

class DataPlaneServicer(control_plane_pb2_grpc.DataPlaneServicer):
    def __init__(self, client):
        self.client = client

    def UpdateModel(self, request, context):
        try:
            logging.info("Received model update from control plane.")
            model_state_buffer = io.BytesIO(request.model_state)
            self.client.my_dnn = load_weights(self.client.my_dnn, torch.load(model_state_buffer, map_location=self.client.device))
            eval_mode(self.client.my_dnn)
            logging.info("Local model updated successfully.")
            return control_plane_pb2.Empty()
        except Exception as e:
            logging.error(f"Exception in UpdateModel: {e}", exc_info=True)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return control_plane_pb2.Empty()

    def UpdateLincRules(self, request, context):
        """Receive LINC rules from control plane."""
        try:
            logging.info(f"Received LINC rules update from control plane: {len(request.rules)} rules")
            
            # Convert protobuf rules to dict format
            rules = []
            for rule_proto in request.rules:
                rule = {
                    'pattern': np.array(rule_proto.pattern),
                    'important_features': list(rule_proto.important_features),
                    'class': rule_proto.class_label,
                    'confidence': rule_proto.confidence
                }
                rules.append(rule)
            
            # Update client's LINC rules
            self.client.update_linc_rules(rules)
            logging.info(f"LINC rules updated successfully: {len(rules)} rules")
            return control_plane_pb2.Empty()
        except Exception as e:
            logging.error(f"Exception in UpdateLincRules: {e}", exc_info=True)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return control_plane_pb2.Empty()

def start_data_plane_server(client):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    control_plane_pb2_grpc.add_DataPlaneServicer_to_server(DataPlaneServicer(client), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    logging.info("Data Plane Server started on port 50052")
    server.wait_for_termination()

def start_data_plane_server_in_thread(client):
    def server_thread():
        start_data_plane_server(client)

    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    logging.info("Data Plane Server started in a separate thread.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Plane Client")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--application_type", type=str, required=True, help="Application type")
    parser.add_argument("-n", "--job_name", type=str, required=True, help="Job name")
    parser.add_argument("--device_list_path", type=str, help="Path to device list")
    parser.add_argument("--eval_frequency", type=int, help="Evaluation frequency")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-i", "--list_flow_input_csv_file_path", nargs="+", required=True, help="List of input CSV file paths")
    parser.add_argument("--feature_names", nargs="+", help="Feature names")
    parser.add_argument("--model_class", type=str, required=True, help="Model class")
    parser.add_argument("--model_input_shape", type=int, help="Model input shape")
    parser.add_argument("-m", "--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--shutdown_control_plane", action="store_true", help="Shutdown control plane after experiment (only set True in the last script)")
    parser.add_argument("--num_classes", type=int, help="Number of output classes for sequential models")
    parser.add_argument("--sequence_length", type=int, default=9, help="Sequence/window length for sequential models")
    parser.add_argument("--sequence_feature_dim", type=int, default=2, help="Per-step feature width for sequential models")
    parser.add_argument("--len_vocab", type=int, default=1501, help="Length vocabulary size")
    parser.add_argument("--ipd_vocab", type=int, default=2561, help="IPD vocabulary size")
    parser.add_argument("--len_embedding_bits", type=int, default=10, help="Length embedding dimension")
    parser.add_argument("--ipd_embedding_bits", type=int, default=8, help="IPD embedding dimension")
    parser.add_argument("--textcnn_nk", type=int, default=4, help="TextCNN convolution filter count per kernel size")
    parser.add_argument("--textcnn_ebdin", type=int, default=4, help="TextCNN intermediate fully-connected dimension")
    parser.add_argument("--rnn_in", type=int, default=12, help="RNN projection dimension")
    parser.add_argument("--rnn_hidden", type=int, default=16, help="RNN hidden size")
    parser.add_argument("--rnn_dropout", type=float, default=0.0, help="RNN dropout rate")
    parser.add_argument("--sequence_dataset_root", type=str, default=os.path.join(Shawarma_home, "emulation", "datasets"), help="Root directory containing sequence npy datasets")
    parser.add_argument("--sequence_dataset_name", type=str, help="Name of sequence dataset (e.g., iscxvpn)")
    parser.add_argument("--sequence_dataset_split", type=str, default="test", help="Split name to load (train/test)")
    parser.add_argument("--feature_file", default="experiment_2018_0.6_X.npy", help="Feature file name")
    parser.add_argument("--label_file", default="experiment_2018_0.6_y.npy", help="Label file name")
    parser.add_argument("--labeler_type", type=str, help="Type of labeler")
    parser.add_argument("--labeler_dnn_class", type=str, help="Class of the labeler DNN model")
    parser.add_argument("--labeler_dnn_path", type=str, help="Path to the labeler DNN model")
    parser.add_argument("--linc_rules_path", type=str, default=None, help="Path to pre-trained LINC rules JSON file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA behavior (may reduce performance)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    env_seed = os.getenv("SHAWARMA_DP_SEED")
    if args.seed is None and env_seed is not None:
        try:
            args.seed = int(env_seed)
        except ValueError:
            logging.warning(f"Invalid SHAWARMA_DP_SEED={env_seed}, ignoring.")

    set_global_seed(args.seed, deterministic=args.deterministic)
    client = DataPlaneClient(args)
    start_data_plane_server_in_thread(client)
    client.run_inference()

if __name__ == "__main__":
    main()
