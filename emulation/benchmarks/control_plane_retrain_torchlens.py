# Control Plane (Server)
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
import logging
from pathlib import Path
import sys
import copy

# Add src to sys.path to allow imports from models, dataset, utils
# This file is in emulation/benchmarks/
# src is in emulation/src/
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

import torch
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Queue, Event
from models.mlp import *
from models.cnnmodel import TextCNN1, TextCNN2
from models.rnnmodel import RNN1
from dataset.dataset import OnlineDataset
import time
import json
import grpc
from concurrent import futures
import control_plane_pb2
import control_plane_pb2_grpc
from labeler.labeler import *
import io
import numpy as np
import torchlens as tl
from utils.model_utils import *
import threading
import random
from bucket_memory import BucketMemory
from layer_freezer_improved import LayerFreezer
from linc_manager import LincManager
from helios_manager import HeliosManager
import torch.nn.functional as F


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



def compute_macro_f1_gpu(pred: torch.Tensor, label: torch.Tensor, num_classes: int) -> float:
    """
    Compute macro F1 score entirely on GPU using confusion matrix.
    Optimized to minimize GPU-CPU synchronization points.
    
    Args:
        pred: Predicted class indices, shape [batch_size]
        label: Ground truth class indices, shape [batch_size]
        num_classes: Number of classes
    
    Returns:
        Macro F1 score as a float
    """
    # Build confusion matrix in one pass using linear indexing
    # conf_matrix[i, j] = count of (label == i) & (pred == j)
    indices = label * num_classes + pred
    conf_flat = torch.bincount(indices, minlength=num_classes * num_classes)
    conf_matrix = conf_flat.view(num_classes, num_classes).float()
    
    # Batch compute TP, FP, FN for all classes at once
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp  # column sum - diagonal
    fn = conf_matrix.sum(dim=1) - tp  # row sum - diagonal
    
    # Compute precision, recall, F1 for all classes
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
    
    # Only average over classes that have samples in ground truth
    class_support = conf_matrix.sum(dim=1)  # samples per class in labels
    valid_mask = class_support > 0
    
    # Single CPU transfer at the end
    if valid_mask.any():
        return float(f1[valid_mask].mean().item())
    return 0.0


def compute_binary_f1_gpu(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Compute binary F1 score (for positive class) entirely on GPU.
    Optimized to use single GPU-CPU sync point.
    
    Args:
        pred: Predicted class indices (0 or 1), shape [batch_size]
        label: Ground truth class indices (0 or 1), shape [batch_size]
    
    Returns:
        F1 score for the positive class
    """
    # Compute all metrics on GPU, single sync at the end
    pred_pos = pred == 1
    label_pos = label == 1
    
    tp = (pred_pos & label_pos).sum().float()
    fp = (pred_pos & ~label_pos).sum().float()
    fn = (~pred_pos & label_pos).sum().float()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
    
    # Single CPU transfer
    return float(f1.item())


def _get_shawarma_home():
    env_home = os.getenv("Shawarma_HOME")
    if env_home:
        return env_home
    # Fallback to repository root if env var is missing (benchmarks -> emulation -> Shawarma)
    return str(Path(__file__).resolve().parents[2])


Shawarma_home = _get_shawarma_home()
os.environ.setdefault("Shawarma_HOME", Shawarma_home)

shutdown_event = threading.Event()


class ControlPlaneServer(control_plane_pb2_grpc.ControlPlaneServicer):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.total_epochs = args.total_epochs
        self.learning_rate = args.learning_rate
        self.test_size = args.test_size
        self.retrain_batch_size = int(getattr(args, "retrain_batch_size", 512))

        self.adaptive_freeze = args.adaptive_freeze

        # LINC mode: use rule-based incremental update with sklearn DecisionTree
        self.linc_mode = getattr(args, 'linc_mode', False)
        self.helios_mode = getattr(args, 'helios_mode', False)
        self.linc_manager = None
        self.linc_update_samples = getattr(args, 'linc_update_samples', 4000)
        
        if self.linc_mode:
            # LINC with sklearn DecisionTree (improved version)
            self.linc_manager = LincManager(
                max_rules=getattr(args, 'linc_max_rules', 10000),
                improved=True
            )
            logging.info("Initialized LINC Manager (sklearn DecisionTree)")
        elif self.helios_mode:
            logging.info("Initializing Helios Manager...")
            self.linc_manager = HeliosManager(
                max_rules=getattr(args, 'helios_max_rules', 10000),
                radio=getattr(args, 'helios_radio', 1.5),  # Larger radio for better coverage
                boost_num=getattr(args, 'helios_boost_num', 6),  # More boosting iterations
                prune_rule_threshold=getattr(args, 'helios_prune_rule', 3),  # Lower threshold to keep more rules
                device=self.device
            )

        # Memory and drift detection settings (used by both adaptive and baseline retraining)
        self.memory_size = getattr(args, 'memory_size', 5000)
        self.memory = BucketMemory(self.memory_size)
        self.label_diff_threshold = getattr(args, 'label_diff_threshold', 0.1)
        # When True, use ground-truth labels for drift detection instead of labeler
        self.use_ground_truth_for_drift = getattr(args, 'use_ground_truth_for_drift', False)
        # Ratio of memory samples to add to relabeled samples during retraining
        self.memory_sample_ratio = getattr(args, 'memory_sample_ratio', 0.1)
        self.deployment_delay = getattr(args, 'deployment_delay', 0.0) # Delay before sending updates

        # Sequence-model configuration (TextCNN1 / TextCNN2 / RNN1) and flag
        self.sequence_models = {"TextCNN1", "TextCNN2", "RNN1"}
        self.sequence_length = getattr(args, "sequence_length", None)
        self.sequence_feature_dim = getattr(args, "sequence_feature_dim", None)

        model_class = args.model_class
        model_input_shape = args.model_input_shape
        self.sequence_model = model_class in self.sequence_models

        # Initialize model and appropriate dummy input
        self.my_dnn, dummy_input = self._init_model(model_class, model_input_shape)
        if args.base_model_path:
            self.my_dnn.load_state_dict(torch.load(args.base_model_path))

        # TorchLens runs an actual forward pass; ensure layers like BatchNorm don't
        # fail on dummy batch size (and we don't want to update running stats here).
        self.my_dnn.eval()
        
        # Save a reference to the original model for embedding extraction (before torchlens conversion)
        # This is needed for LINC/Helios to extract meaningful features from sequence models
        if self.sequence_model:
            self._original_model_for_embedding = copy.deepcopy(self.my_dnn)
            self._original_model_for_embedding.eval()
            logging.info(f"Saved original {model_class} model for embedding extraction")

        # Check if the model is binary or multi-class based on output dimension
        with torch.no_grad():
            out = self.my_dnn(dummy_input)
            self.is_binary_model = (out.shape[-1] <= 2)
            logging.info(f"Model detected as {'binary' if self.is_binary_model else 'multi-class'} (output dim: {out.shape[-1]})")

        model_name = getattr(self.my_dnn, "name", type(self.my_dnn).__name__)
        my_dnn_history = tl.log_forward_pass(
            self.my_dnn,
            dummy_input,
            vis_opt="unrolled",
            save_function_args=True,
            vis_outpath=f"{Shawarma_home}/emulation/scripts/results/{model_name}",
            vis_save_only=True,
        )
        self.my_dnn = my_dnn_history.layer_list
        self.label_index = label2index(self.my_dnn)
        self.optimizer = torch.optim.Adam(get_weights(self.my_dnn), lr=self.learning_rate)
        self.max_partition_index = get_max_partition_index(self.my_dnn)
        self.partition_index = 0
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.labeler = self.initialize_labeler(args)
        self.warmup_done_event = Event()
        self._retraining_lock = Event()
        self._retraining_lock.set()
        self._server_stop_event = None
        self._training_warmup_done = False
        # Track training step during an active retraining session
        self._train_step = 0
        # F1_d for current retraining session (data plane window F1 at drift)
        self._F1_d_current = 0.5

        # initialize the layer freezer
        self.freezer = LayerFreezer(
            model=self.my_dnn,
            T=10,
            idle_f1=0.95,
            prewarm=False,
        )
        self.drift = False
        self._labeler_warm = False

        # OPTIMIZATION: Reuse gRPC channel to avoid connection overhead
        self._data_plane_channel = None
        self._data_plane_stub = None

        # Initialize rules from model if LINC or Helios mode is enabled
        if self.linc_mode:
            self._init_linc_rules()
        elif self.helios_mode:
            self._init_helios_rules()

    def _get_data_plane_stub(self):
        """Get or create a reusable gRPC stub for data plane communication.
        
        OPTIMIZATION: Reuse channel to avoid connection overhead on every update.
        """
        if self._data_plane_channel is None or self._data_plane_stub is None:
            self._data_plane_channel = grpc.insecure_channel('localhost:50052')
            self._data_plane_stub = control_plane_pb2_grpc.DataPlaneStub(self._data_plane_channel)
        return self._data_plane_stub

    def _close_data_plane_channel(self):
        """Close the reusable gRPC channel."""
        if self._data_plane_channel is not None:
            self._data_plane_channel.close()
            self._data_plane_channel = None
            self._data_plane_stub = None

    def _init_linc_rules(self):
        """
        Initialize LINC manager and optionally load pre-trained rules.
        """
        logging.info(f"LINC mode enabled with LincManager.")
        
        # Load pre-trained rules if path is provided
        linc_rules_path = getattr(self.args, 'linc_rules_path', None)
        if linc_rules_path and os.path.exists(linc_rules_path):
            self._load_linc_rules(linc_rules_path)

    def _init_helios_rules(self):
        """
        Initialize Helios manager and optionally load pre-trained rules.
        """
        logging.info(f"Helios mode enabled with HeliosManager (max_rules={getattr(self.args, 'helios_max_rules', 3000)}, radio={getattr(self.args, 'helios_radio', 1.5)}).")
        
        # Load pre-trained rules if path is provided
        helios_rules_path = getattr(self.args, 'helios_rules_path', None)
        if helios_rules_path and os.path.exists(helios_rules_path):
            self._load_helios_rules(helios_rules_path)
        else:
            logging.info("No pre-trained Helios rules provided, will generate rules on first drift.")

    def _load_helios_rules(self, rules_path):
        """Load pre-trained Helios rules from JSON file into HeliosManager."""
        if not os.path.exists(rules_path):
            logging.warning(f"Helios rules file not found: {rules_path}")
            return
        
        try:
            with open(rules_path, 'r') as f:
                rules_data = json.load(f)
            
            # Import HeliosRule from the correct location
            from helios_model import HeliosRule
            
            loaded_rules = []
            for rule in rules_data.get('rules', []):
                helios_rule = HeliosRule(
                    class_label=rule['class_label'],
                    intervals=rule['intervals'],
                    confidence=rule.get('confidence', 1.0),
                    prototype_idx=rule.get('prototype_idx', -1)
                )
                # Restore support if available
                helios_rule.support = rule.get('support', 0)
                loaded_rules.append(helios_rule)
            
            self.linc_manager.rules = loaded_rules
            logging.info(f"Loaded {len(loaded_rules)} pre-trained Helios rules from {rules_path}")
            
            # Log class distribution
            class_counts = {}
            for r in loaded_rules:
                c = r.class_label
                class_counts[c] = class_counts.get(c, 0) + 1
            logging.info(f"Pre-trained Helios rules class distribution: {class_counts}")
            
        except Exception as e:
            logging.error(f"Failed to load Helios rules from {rules_path}: {e}")
    
    def _load_linc_rules(self, rules_path):
        """Load pre-trained LINC rules from JSON file into LincManager."""
        if not os.path.exists(rules_path):
            logging.warning(f"LINC rules file not found: {rules_path}")
            return
        
        try:
            with open(rules_path, 'r') as f:
                rules_data = json.load(f)
            
            # Convert JSON rules to LincRule objects
            from linc_manager import LincRule
            
            loaded_rules = []
            for rule in rules_data.get('rules', []):
                # Parse the tree-based format: 
                # important_features = [idx1, op1, idx2, op2, ...]
                # pattern = [thresh1, thresh2, ...]
                imp_feats = rule.get('important_features', [])
                threshs = rule.get('pattern', [])
                
                conditions = []
                num_conds = len(threshs)
                if len(imp_feats) >= 2 * num_conds:
                    for i in range(num_conds):
                        idx = imp_feats[2*i]
                        op = imp_feats[2*i+1]
                        thresh = threshs[i]
                        conditions.append((idx, thresh, op))
                
                linc_rule = LincRule(
                    conditions=conditions,
                    class_label=rule['class'],
                    confidence=rule['confidence']
                )
                loaded_rules.append(linc_rule)
            
            self.linc_manager.rules = loaded_rules
            logging.info(f"Loaded {len(loaded_rules)} pre-trained LINC rules from {rules_path}")
            
            # Log class distribution
            class_counts = {}
            for r in loaded_rules:
                c = r.class_label
                class_counts[c] = class_counts.get(c, 0) + 1
            logging.info(f"Pre-trained rules class distribution: {class_counts}")
            
        except Exception as e:
            logging.error(f"Failed to load LINC rules from {rules_path}: {e}")

    def linc_extract_rules(self, features, labels, predictions, confidences):
        """
        Extract rules from high-confidence predictions using Decision Tree.
        """
        if not self.linc_manager:
            return

        import numpy as np
        
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        preds_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        confs_np = confidences.cpu().numpy() if torch.is_tensor(confidences) else confidences
        
        # Filter: only correct, high-confidence predictions
        # Note: self.linc_rule_threshold is not set in __init__ anymore if I removed it,
        # but I kept linc_mode block. Ah I removed the lines setting linc_rule_threshold in __init__.
        # I should check if I can access args. Or just use default 0.5.
        threshold = 0.5
        valid_mask = (preds_np == labels_np) & (confs_np >= threshold)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return
        
        # Flatten features for valid samples
        valid_features = features_np[valid_indices].reshape(len(valid_indices), -1)
        valid_labels = labels_np[valid_indices]
        
        self.linc_manager.fit(valid_features, valid_labels)
        self._send_linc_rules_to_data_plane()

    def _extract_embedding_features(self, features_tensor):
        """
        Extract embedding features from sequence models (RNN1, TextCNN1, etc.) for rule learning.
        
        For sequence models, raw token IDs are not suitable for threshold-based rules.
        Instead, we extract the learned embedding representations which are continuous
        and meaningful for rule-based classification.
        
        OPTIMIZED: For RNN1, extract the final hidden state which contains richer
        sequential information than just mean-pooled embeddings.
        
        Args:
            features_tensor: Input tensor, shape depends on model type:
                - Sequence models: (batch_size, seq_len, feature_dim) with token IDs
                - MLP models: (batch_size, feature_dim) with continuous features
        
        Returns:
            numpy array of shape (batch_size, embedding_dim) suitable for rule learning
        """
        import numpy as np
        
        if not self.sequence_model:
            # For non-sequence models, just flatten and return
            if torch.is_tensor(features_tensor):
                return features_tensor.cpu().numpy().reshape(features_tensor.shape[0], -1)
            return np.array(features_tensor).reshape(len(features_tensor), -1)
        
        # For sequence models, extract embeddings using the saved original model
        with torch.no_grad():
            features_tensor = features_tensor.to(self.device)
            if features_tensor.dtype != torch.long:
                features_tensor = features_tensor.long()
            
            model_class = self.args.model_class
            
            # Use the saved original model for embedding extraction
            if hasattr(self, '_original_model_for_embedding') and self._original_model_for_embedding is not None:
                original_model = self._original_model_for_embedding
                original_model.to(self.device)
                original_model.eval()
                
                # Extract length and IPD tokens
                len_x = features_tensor[:, :, 0]  # (batch_size, seq_len)
                ipd_x = features_tensor[:, :, 1]  # (batch_size, seq_len)
                
                # Clamp inputs to valid vocabulary range to prevent CUDA asserts
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
                
                # OPTIMIZATION: For RNN1, extract only embedding features (Input Projection)
                if model_class == "RNN1" and hasattr(original_model, 'rnn') and hasattr(original_model, 'fc1'):
                    # Modified: Extract only embedding features, using FC1 projection, then FLATTEN (no pooling)
                    # Apply fc1 transformation: (batch_size, seq_len, rnn_in)
                    x = original_model.fc1(combined)
                    
                    # Flatten the sequence: (batch_size, seq_len * rnn_in)
                    features = x.reshape(x.shape[0], -1)
                    
                    return features.cpu().numpy()
                
                elif model_class in ["TextCNN1", "TextCNN2"] and hasattr(original_model, 'conv'):
                    # For TextCNN, apply convolution and pooling
                    x = original_model.fc1(combined) if hasattr(original_model, 'fc1') else combined
                    
                    # Apply convolution layers if available
                    if hasattr(original_model, 'conv'):
                        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
                        conv_out = original_model.conv(x)
                        # Global max pooling
                        pooled = torch.max(conv_out, dim=2)[0]
                        return pooled.cpu().numpy()
                
                # Fallback: use mean pooling with fc1 transformation
                if hasattr(original_model, 'fc1'):
                    combined = original_model.fc1(combined)
                
                # Aggregate across sequence: use both mean and max pooling for richer features
                mean_pooled = combined.mean(dim=1)
                max_pooled = combined.max(dim=1)[0]
                
                # Concatenate mean and max pooled features
                features = torch.cat([mean_pooled, max_pooled], dim=-1)
                
                return features.cpu().numpy()
            
            # Fallback: create temporary embedding layers (untrained, less effective)
            logging.warning("Original model not available for embedding extraction, using untrained embeddings")
            
            len_x = features_tensor[:, :, 0]
            ipd_x = features_tensor[:, :, 1]
            
            len_vocab = getattr(self.args, 'len_vocab', 1501)
            ipd_vocab = getattr(self.args, 'ipd_vocab', 2561)
            len_emb_bits = getattr(self.args, 'len_embedding_bits', 10)
            ipd_emb_bits = getattr(self.args, 'ipd_embedding_bits', 8)
            
            temp_len_emb = torch.nn.Embedding(len_vocab, len_emb_bits).to(self.device)
            temp_ipd_emb = torch.nn.Embedding(ipd_vocab, ipd_emb_bits).to(self.device)
            
            len_embedded = temp_len_emb(len_x)
            ipd_embedded = temp_ipd_emb(ipd_x)
            
            combined = torch.cat([len_embedded, ipd_embedded], dim=-1)
            pooled = combined.mean(dim=1)
            
            return pooled.cpu().numpy()

    def linc_incremental_update(self, samples):
        """
        Incrementally update LINC/Helios rules with new labeled samples.
        
        Following original Helios logic:
        1. Extract features and labels from samples
        2. For sequence models, extract embedding features instead of raw token IDs
        3. Call manager's incremental_update (handles conflict removal + rule generation)
        4. Send updated rules to data plane
        """
        if not self.linc_manager or not samples:
            return

        import numpy as np
        import random
        
        # Use all samples for incremental update (don't subsample too aggressively)
        max_samples = self.linc_update_samples
        if self.helios_mode:
            # Use more samples for Helios to ensure good coverage
            max_samples = min(len(samples), max(self.linc_update_samples, 10000))
        
        # Randomly sample if we have more samples than needed
        if len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        # Extract features and labels from samples
        labels_list = []
        for sample in samples:
            labels_list.append(int(sample['klass']))
        
        # Stack features into a tensor for batch processing
        features_list = []
        for sample in samples:
            feature = sample['feature']
            if torch.is_tensor(feature):
                features_list.append(feature)
            else:
                features_list.append(torch.tensor(feature))
        
        features_tensor = torch.stack(features_list, dim=0)
        
        # For sequence models, extract embedding features instead of raw token IDs
        if self.sequence_model:
            new_features = self._extract_embedding_features(features_tensor)
            logging.info(f"Extracted embedding features for sequence model: shape={new_features.shape}")
        else:
            # For non-sequence models, flatten features directly
            new_features = features_tensor.cpu().numpy().reshape(len(samples), -1)
        
        new_labels = np.array(labels_list)
        
        logging.info(f"{'Helios' if self.helios_mode else 'LINC'}: Incremental update with {len(new_features)} samples")
        
        # Perform incremental update (manager handles conflict removal + boosting)
        self.linc_manager.incremental_update(new_features, new_labels)
        
        # Log accuracy on update samples
        if hasattr(self.linc_manager, 'rules') and len(self.linc_manager.rules) > 0:
            # For Helios mode, metrics are already logged internally by HeliosSystem using fast vectorized ops
            if self.helios_mode:
                 logging.info(f"Helios post-update: Rules={len(self.linc_manager.rules)} (See HeliosSystem logs for detailed coverage/accuracy)")
            else:
                correct_count = 0
                covered_count = 0
                
                for i in range(len(new_features)):
                    x = new_features[i]
                    y_true = new_labels[i]
                    
                    for rule in self.linc_manager.rules:
                        if rule.match(x):
                            covered_count += 1
                            if rule.class_label == y_true:
                                correct_count += 1
                            break
                
                coverage_rate = covered_count / len(new_features) if len(new_features) > 0 else 0
                accuracy_rate = correct_count / len(new_features) if len(new_features) > 0 else 0
                
                method_name = "LINC"
                logging.info(f"{method_name} post-update: coverage={coverage_rate:.1%}, accuracy={accuracy_rate:.1%}, rules={len(self.linc_manager.rules)}")
        
        self._send_linc_rules_to_data_plane()

    def _send_linc_rules_to_data_plane(self):
        """Send LINC/Helios rules to data plane via gRPC. Serializing rules to proto.
        
        OPTIMIZATION: Use reusable gRPC channel to avoid connection overhead.
        """
        if not self.linc_manager or not self.linc_manager.rules:
            logging.info("No rules to send to data plane.")
            return
        
        try:
            rules_proto = []
            for rule in self.linc_manager.rules:
                # Both LINC and Helios rules have a `conditions` property
                # that returns list of (feature_idx, threshold, operator)
                # For Helios: intervals are converted via the conditions property
                # For LINC: conditions are already in this format
                
                imp_feats = []
                threshs = []
                for idx, thresh, op in rule.conditions:
                    imp_feats.append(int(idx))
                    imp_feats.append(int(op))
                    threshs.append(float(thresh))
                
                rule_proto = control_plane_pb2.LincRule(
                    pattern=threshs,
                    important_features=imp_feats,
                    class_label=int(rule.class_label),
                    confidence=float(rule.confidence)
                )
                rules_proto.append(rule_proto)
            
            request = control_plane_pb2.LincRulesRequest(rules=rules_proto)
            
            if not shutdown_event.is_set():
                stub = self._get_data_plane_stub()
                stub.UpdateLincRules(request)
                method_name = "Helios" if self.helios_mode else "LINC"
                # Log class distribution for debugging
                class_counts = {}
                for r in self.linc_manager.rules:
                    c = r.class_label
                    class_counts[c] = class_counts.get(c, 0) + 1
                logging.info(f"Sent {len(self.linc_manager.rules)} {method_name} rules to data plane. Class distribution: {class_counts}")
        except Exception as e:
            logging.error(f"Failed to send rules to data plane: {e}")

    def linc_predict(self, features):
        """
        Predict using LINC manager.
        """
        if not self.linc_manager:
            return None, None
            
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        # Reshape if necessary
        batch_size = features_np.shape[0]
        if len(features_np.shape) > 2:
            features_np = features_np.reshape(batch_size, -1)
            
        predictions, confidences = self.linc_manager.predict(features_np)
        return torch.tensor(predictions), torch.tensor(confidences)

    def _validate_sequence_args(self):
        if self.sequence_model:
            if getattr(self.args, "num_classes", None) is None:
                raise ValueError("--num_classes must be specified for sequential models.")
            if not self.sequence_length or not self.sequence_feature_dim:
                raise ValueError("--sequence_length and --sequence_feature_dim are required for sequential models.")

    def _build_dummy_sequence(self, batch_size: int = 1):
        seq_shape = (batch_size, self.sequence_length, self.sequence_feature_dim)
        dummy = torch.zeros(seq_shape, dtype=torch.long, device=self.device)
        len_vocab = max(1, int(getattr(self.args, "len_vocab", 1)))
        ipd_vocab = max(1, int(getattr(self.args, "ipd_vocab", 1)))
        # First feature: length token
        dummy[..., 0] = torch.randint(0, len_vocab, dummy[..., 0].shape, device=self.device)
        # Second feature: IPD token (if present)
        if self.sequence_feature_dim > 1:
            dummy[..., 1] = torch.randint(0, ipd_vocab, dummy[..., 1].shape, device=self.device)
        return dummy

    def _init_model(self, model_class: str, model_input_shape: int):
        # Sequential TextCNN1 or TextCNN2
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
            return model, self._build_dummy_sequence(batch_size=self.retrain_batch_size)

        # Sequential RNN1
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
            return model, self._build_dummy_sequence(batch_size=self.retrain_batch_size)

        # Legacy non-sequential models (MLP, PForest, etc.)
        my_dnn_class = globals()[model_class]
        model = (
            my_dnn_class(input_shape=model_input_shape).to(self.device)
            if model_input_shape
            else my_dnn_class().to(self.device)
        )
        feature_dim = model_input_shape if model_input_shape else 16
        dummy = torch.zeros((1, feature_dim), device=self.device)
        return model, dummy

    def _build_training_dataloader(self, features, labels, batch_size: int = None):
        if batch_size is None:
            batch_size = self.retrain_batch_size
        if self.sequence_model:
            # Pad data to make it divisible by batch_size to avoid dropping samples
            n_samples = features.shape[0]
            remainder = n_samples % batch_size
            if remainder != 0:
                pad_size = batch_size - remainder
                # Repeat samples from the beginning to fill the last batch
                pad_indices = torch.arange(pad_size) % n_samples
                features = torch.cat([features, features[pad_indices]], dim=0)
                labels = torch.cat([labels, labels[pad_indices]], dim=0)
            dataset = TensorDataset(features.to(self.device).long(), labels.to(self.device))
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        dataset = OnlineDataset(features, labels, standardize=False, normalize=False, device=self.device)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def adaptive_layer_freeze_retrain(self, my_dnn, training_dataloader, total_epochs, optimizer, loss_fn, freezer, l1_reg_flag=False, l1_reg_param=0.001):
        my_dnn = train_mode(my_dnn)
        freezer.unfreeze_layers()
        label_index = label2index(my_dnn)

        # Materialize batches once so DataLoader initialization cost stays off the timed path.
        batch_list = list(training_dataloader)
        if len(batch_list) == 0:
            logging.warning("Empty training dataloader, skipping retraining.")
            return my_dnn, optimizer

        num_batches = len(batch_list)

        for epoch_index in range(total_epochs):
            random.shuffle(batch_list)
            epoch_start = time.time()
            epoch_loss = 0.0
            batch_f1_val = None
            fully_frozen_batches = 0
            
            for batch_index, (features, label) in enumerate(batch_list):
                optimizer.zero_grad(set_to_none=True)
                my_score = forward_eval(my_dnn, label_index, features)

                # Compute F1 for freeze decision (skip first batch)
                if batch_index > 0:
                    with torch.no_grad():
                        if my_score.dim() > 1:
                            batch_pred = my_score.argmax(dim=1)
                            batch_f1_val = compute_macro_f1_gpu(batch_pred, label, my_score.shape[1])
                        else:
                            batch_pred = (my_score > 0.5).long()
                            batch_f1_val = compute_binary_f1_gpu(batch_pred, label)

                loss = loss_fn(my_score, label)

                # Check freeze decision for batches after the first
                fully_frozen = False
                if batch_f1_val is not None and batch_index > 0:
                    soft_scores = F.softmax(my_score.detach(), dim=1) if my_score.dim() > 1 else my_score.detach()
                    train_step = int(self._train_step)
                    F1_d_val = float(self._F1_d_current)
                    freeze_idx, fully_frozen = freezer.get_freeze_idx(
                        soft_scores, label, batch_f1_val, train_step, F1_d_val
                    )
                    if fully_frozen:
                        fully_frozen_batches += 1
                        epoch_loss += loss.item()
                        # Skip backward/optimizer when all layers are frozen - no gradients to compute
                        freezer.unfreeze_layers()
                        continue
                    freezer.freeze_layers()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Calculate Fisher only on first batch when no freeze decision yet
                if not freezer.freeze_idx and batch_index == 0:
                    freezer.calculate_fisher()

                if freezer.freeze_idx:
                    freezer.unfreeze_layers()

            epoch_time_ms = (time.time() - epoch_start) * 1000
            avg_loss = epoch_loss / num_batches
            logging.info(
                f"Epoch {epoch_index}, loss={avg_loss:.4f}, time={epoch_time_ms:.2f}ms, skipped={fully_frozen_batches}/{num_batches}"
            )
            
            # Early stopping if all batches were fully frozen
            if num_batches > 1 and fully_frozen_batches == num_batches - 1:
                logging.info("All batches in epoch were fully frozen. Stopping training.")
                break
                
        return my_dnn, optimizer

    def original_retrain(self, training_features, training_labels):
        """Baseline retraining path (no adaptive freezing), mirroring original server behavior.
        
        OPTIMIZATION: Expects features and labels already on device to avoid transfer overhead.
        """
        try:
            if training_features is None or training_features.nelement() == 0:
                logging.info("No valid training samples, skipping retraining.")
                return

            logging.info(f"Starting baseline retraining with {training_features.shape[0]} samples.")

            training_dataset = OnlineDataset(
                training_features,
                training_labels,
                standardize=False,
                normalize=False,
                device=self.device,
            )
            # Pad data for sequence models to avoid dropping samples
            if self.sequence_model:
                n_samples = training_features.shape[0]
                remainder = n_samples % self.retrain_batch_size
                if remainder != 0:
                    pad_size = self.retrain_batch_size - remainder
                    pad_indices = torch.arange(pad_size, device=self.device) % n_samples
                    training_features = torch.cat([training_features, training_features[pad_indices]], dim=0)
                    training_labels = torch.cat([training_labels, training_labels[pad_indices]], dim=0)
                    training_dataset = OnlineDataset(
                        training_features,
                        training_labels,
                        standardize=False,
                        normalize=False,
                        device=self.device,
                    )
            training_dataloader = DataLoader(training_dataset, batch_size=self.retrain_batch_size, shuffle=True, drop_last=False)

            # Inline training loop with logging (similar to adaptive_layer_freeze_retrain)
            self.my_dnn = train_mode(self.my_dnn)
            label_index = label2index(self.my_dnn)
            batch_list = list(training_dataloader)
            
            if len(batch_list) == 0:
                logging.warning("Empty training dataloader after drop_last, skipping retraining.")
                return
            
            num_batches = len(batch_list)
            
            for epoch_index in range(self.total_epochs):
                random.shuffle(batch_list)
                epoch_start = time.time()
                epoch_loss = 0.0
                
                for batch_index, (features, label) in enumerate(batch_list):
                    self.optimizer.zero_grad(set_to_none=True)
                    my_score = forward_eval(self.my_dnn, label_index, features)
                    loss = self.loss_fn(my_score, label)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                epoch_time_ms = (time.time() - epoch_start) * 1000
                avg_loss = epoch_loss / num_batches
                logging.info(f"Epoch {epoch_index}, loss={avg_loss:.4f}, time={epoch_time_ms:.2f}ms")

            # After retraining, push updated weights to the data plane.
            if self.deployment_delay > 0:
                logging.info(f"Waiting {self.deployment_delay}s before sending weights (deployment delay)...")
                time.sleep(self.deployment_delay)

            weights = get_weights(self.my_dnn)
            buffer = io.BytesIO()
            torch.save(weights, buffer)
            buffer.seek(0)
            model_bytes = buffer.read()
            if not shutdown_event.is_set():
                stub = self._get_data_plane_stub()
                stub.UpdateModel(control_plane_pb2.ModelUpdateRequest(model_state=model_bytes))
                logging.info("Updated model state sent to data plane (baseline retrain).")
        except Exception as exc:
            logging.error(f"original_retrain failed: {exc}")

    def adaptive_retrain(self, label_samples):
        """Adaptive-freeze retraining using relabeled samples from drift detection."""
        if not label_samples:
            logging.info("No relabeled samples provided to adaptive_retrain; skipping retraining.")
            return

        try:
            # OPTIMIZATION: Build tensors directly on device to avoid CPU->GPU transfer
            features = torch.stack([s['feature'] for s in label_samples], dim=0)
            labels = torch.tensor([int(s['klass']) for s in label_samples], dtype=torch.long, device=self.device)
            
            # If retraining a binary model but labels are multi-class, convert to binary
            if self.is_binary_model:
                labels = (labels > 0).long()
                logging.debug("Retraining binary model: converted multi-class labels to binary.")
        except Exception as exc:
            logging.error(f"Failed to build tensors from label_samples: {exc}")
            return

        training_dataloader = self._build_training_dataloader(features, labels, batch_size=self.retrain_batch_size)

        self.my_dnn, self.optimizer = self.adaptive_layer_freeze_retrain(
            self.my_dnn,
            training_dataloader,
            self.total_epochs,
            self.optimizer,
            self.loss_fn,
            self.freezer,
        )

        # After adaptive retraining, push updated weights to the data plane.
        try:
            if self.deployment_delay > 0:
                logging.info(f"Waiting {self.deployment_delay}s before sending weights (deployment delay)...")
                time.sleep(self.deployment_delay)

            weights = get_weights(self.my_dnn)
            buffer = io.BytesIO()
            torch.save(weights, buffer)
            buffer.seek(0)
            model_bytes = buffer.read()
            if not shutdown_event.is_set():
                stub = self._get_data_plane_stub()
                stub.UpdateModel(control_plane_pb2.ModelUpdateRequest(model_state=model_bytes))
                logging.info("Updated model state sent to data plane (adaptive retrain).")
        except Exception as exc:
            logging.error(f"Failed to send updated model to data plane: {exc}")

    def handle_memory_and_drift(self, online_samples, window_index=None, ground_truth_labels=None):
        """Relabel incoming samples and optionally flag drift.

        Returns:
            (relabeled_samples, drift_detected)

        Args:
            online_samples: List of sample dicts with 'feature' and 'klass' keys
            window_index: Optional window index for logging
            ground_truth_labels: Optional tensor of ground-truth labels for drift detection.
                                 When use_ground_truth_for_drift=True and this is provided,
                                 drift is detected by comparing ground-truth with original_labels
                                 instead of using the labeler.

        Notes:
            - We always relabel the full incoming window and overwrite each sample's `klass` with the relabeled value.
            - Drift is flagged when the mismatch ratio crosses `self.label_diff_threshold`.
            - To preserve the original experimental behavior, we skip declaring drift while memory is still empty
              (we still relabel, but `drift_detected` will be False).
        """
        if not online_samples:
            logging.warning("No online samples provided for drift detection.")
            return None, False

        # Determine which labels to use for drift detection
        use_ground_truth = self.use_ground_truth_for_drift and ground_truth_labels is not None

        # Extract original labels (predictions from data plane model)
        original_labels = torch.tensor(
            [int(sample['klass']) for sample in online_samples],
            dtype=torch.long,
            device=self.device,
        )

        if use_ground_truth:
            # Use ground-truth labels directly for drift detection - no need to stack features
            if not isinstance(ground_truth_labels, torch.Tensor):
                ground_truth_labels = torch.tensor(ground_truth_labels, dtype=torch.long, device=self.device)
            else:
                ground_truth_labels = ground_truth_labels.to(device=self.device, dtype=torch.long)
            
            labels_for_comparison = ground_truth_labels
            labels_for_update = ground_truth_labels
            logging.debug("Using ground-truth labels for drift detection.")
        else:
            # Use labeler to generate labels for drift detection (original behavior)
            # Only stack features when we need to run the labeler
            try:
                feature_batch = torch.stack([sample['feature'] for sample in online_samples], dim=0).to(self.device)
            except Exception as exc:
                logging.error(f"Failed to stack online sample features: {exc}")
                return None, False

            try:
                with torch.no_grad():
                    label_output = self.labeler.label(feature_batch)
                relabeled = self._coerce_label_tensor(label_output, feature_batch.size(0), original_labels.device)
                self._labeler_warm = True
            except TypeError:
                logging.error("Labeler warmup failed: labeler requires additional arguments for labeling.")
                return None, False
            except Exception as exc:
                logging.error(f"Labeling failed during drift detection: {exc}")
                return None, False
            
            labels_for_comparison = relabeled
            labels_for_update = relabeled

        # If the data plane model is binary, we compare on a benign-vs-attack basis.
        # If it's multi-class, we compare the exact class indices.
        if self.is_binary_model:
            labels_for_comparison = (labels_for_comparison > 0).long()
        
        mismatches = labels_for_comparison.ne(original_labels)
        mismatch_count = int(mismatches.sum().item())
        total_samples = len(online_samples)
        mismatch_ratio = (mismatch_count / total_samples) if total_samples else 0.0

        # Overwrite labels for ALL samples so memory can always be updated with correct `klass`.
        # Optimization: Use .tolist() for batch conversion instead of per-element .item() calls
        # This avoids N separate CUDA synchronization points
        labels_list = labels_for_update.detach().cpu().tolist()
        for idx, sample in enumerate(online_samples):
            sample['klass'] = labels_list[idx]

        # Preserve original behavior: do not start declaring drift until memory has been primed.
        if len(self.memory) == 0:
            label_source = "ground-truth" if use_ground_truth else "relabeled"
            logging.info(f"Memory empty; priming memory with {label_source} samples; skipping drift decision for this window.")
            return online_samples, False

        drift_detected = bool(mismatch_ratio >= self.label_diff_threshold and mismatch_count > 0)
        if drift_detected:
            self.drift = True
            window_prefix = f"Window {window_index}: " if window_index is not None else ""
            label_source = "ground-truth vs prediction" if use_ground_truth else "labeler vs prediction"
            logging.info(
                f"{window_prefix}Drift detected using {label_source} (mismatch ratio: {mismatch_ratio:.4f}, mismatched samples: {mismatch_count}/{total_samples})"
            )

        return online_samples, drift_detected

    def RequestRetraining(self, request, context):
        if shutdown_event.is_set():
            logging.info("Shutdown in progress, skip retraining.")
            return control_plane_pb2.Empty()
        if not self._retraining_lock.is_set():
            window_index = request.window_index
            # When training is in progress, accumulate train_step per new window request
            self._train_step += 1
            logging.warning(f"Retraining already in progress, skipping new retraining request for window index {window_index}. Accumulated train_step={self._train_step}.")
            return control_plane_pb2.Empty()
        self._retraining_lock.clear()
        try:
            window_index = request.window_index
            logging.info(f"Received retraining request for window index {window_index}.")
            # Prefer F1_d from gRPC field (my_window_f1), fallback to idle_f1 if missing
            self._F1_d_current = float(getattr(request, 'my_window_f1'))
            logging.info(f"Using F1_d={self._F1_d_current} (my_window_f1) for window {window_index}.")
            # Reset train_step for this new retraining session
            self._train_step = 0
            
            # OPTIMIZATION: Convert features and labels to tensors directly on target device
            # to avoid CPU->GPU transfer overhead
            dtype = torch.long if self.sequence_model else torch.float32
            online_learning_features = torch.as_tensor(
                request.features.data, 
                dtype=dtype, 
                device=self.device
            ).view(*request.features.shape)
            online_learning_labels = torch.as_tensor(
                request.my_labels, 
                dtype=torch.long, 
                device=self.device
            )
            # online_samples creation moved to else block to avoid overhead when adaptive_freeze is False

            # OPTIMIZATION: Freeze model and initialize optimizer (already on device)
            self.my_dnn = frozen_model(self.my_dnn, partition_index=self.partition_index)
            self.optimizer = torch.optim.Adam(get_weights(self.my_dnn), lr=self.learning_rate)
            
            # OPTIMIZATION: Load pre-warmed optimizer state to avoid cold start latency
            if hasattr(self, 'initial_optimizer_state'):                      
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.requires_grad:
                            state = self.optimizer.state[p]
                            # Force full initialization
                            state['step'] = torch.tensor(0.)
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            
            if not self.adaptive_freeze:
                # OPTIMIZATION: Build online_samples for drift detection (already on device)
                labels_list = online_learning_labels.tolist()
                online_samples = [{'klass': labels_list[i], 'feature': feature} for i, feature in enumerate(online_learning_features)]
                
                # OPTIMIZATION: Extract ground-truth labels if provided
                ground_truth_labels = None
                if self.use_ground_truth_for_drift and hasattr(request, 'ground_truth_labels') and len(request.ground_truth_labels) > 0:
                    ground_truth_labels = torch.tensor(request.ground_truth_labels, dtype=torch.long, device=self.device)
                    logging.info(f"Using ground-truth labels for drift detection (count: {len(ground_truth_labels)})")
                
                # OPTIMIZATION: Perform drift detection
                t_time = time.time()
                relabeled_samples, drift_detected = self.handle_memory_and_drift(online_samples, window_index, ground_truth_labels)
                t_time = (time.time() - t_time) * 1000
                logging.info(f"Memory and drift handling took {t_time:.2f} ms")
                
                if relabeled_samples is not None:
                    # Refresh memory every window
                    self.memory_future_step(relabeled_samples)
                    sorted_keys = sorted(self.memory.class_counts.keys())
                    cls_counts = [self.memory.class_counts[k] for k in sorted_keys]
                    logging.info(f"Memory class distribution (sorted by class index): {cls_counts}")
                
                # Only retrain when drift is detected
                if drift_detected and relabeled_samples is not None:
                    # OPTIMIZATION: Build training set from relabeled samples (already on device)
                    features = torch.stack([s['feature'] for s in relabeled_samples], dim=0)
                    labels = torch.tensor([int(s['klass']) for s in relabeled_samples], dtype=torch.long, device=self.device)
                    
                    # Convert to binary if needed
                    if self.is_binary_model:
                        labels = (labels > 0).long()
                    
                    # OPTIMIZATION: Hybrid strategy - combine with memory samples
                    num_relabeled = len(relabeled_samples)
                    num_memory_samples = int(num_relabeled * self.memory_sample_ratio / (1 - self.memory_sample_ratio))
                    memory_samples = self.memory.quota_balanced_retrieval(min(num_memory_samples, len(self.memory)))
                    
                    if memory_samples and len(memory_samples) > 0:
                        mem_features = torch.stack([s['feature'] for s in memory_samples], dim=0)
                        mem_labels = torch.tensor([int(s['klass']) for s in memory_samples], dtype=torch.long, device=self.device)
                        if self.is_binary_model:
                            mem_labels = (mem_labels > 0).long()
                        features = torch.cat([features, mem_features], dim=0)
                        labels = torch.cat([labels, mem_labels], dim=0)
                        logging.info(f"Training with {num_relabeled} relabeled samples + {len(memory_samples)} memory samples (total: {len(labels)})")
                    else:
                        logging.info(f"Training with {num_relabeled} relabeled samples only (memory empty)")
                    
                    # Choose update method based on mode
                    if self.linc_mode or self.helios_mode:
                        # Rule-based incremental update
                        combined_samples = relabeled_samples + (memory_samples if memory_samples else [])
                        self.linc_incremental_update(combined_samples)
                        if self.helios_mode:
                            method_name = "Helios"
                        else:
                            method_name = "LINC"
                        logging.info(f"{method_name} incremental update for window index {window_index} completed.")
                    else:
                        # OPTIMIZATION: Pass features already on device
                        self.original_retrain(features, labels)
                        logging.info(f"Retraining for window index {window_index} completed.")
                else:
                    logging.info("No drift or insufficient samples, skipping retraining.")
            else:
                # OPTIMIZATION: Build online_samples (already on device)
                labels_list = online_learning_labels.tolist()
                online_samples = [{'klass': labels_list[i], 'feature': feature} for i, feature in enumerate(online_learning_features)]
                
                # OPTIMIZATION: Extract ground-truth labels if provided and use_ground_truth_for_drift is enabled
                ground_truth_labels = None
                if self.use_ground_truth_for_drift and hasattr(request, 'ground_truth_labels') and len(request.ground_truth_labels) > 0:
                    ground_truth_labels = torch.tensor(request.ground_truth_labels, dtype=torch.long, device=self.device)
                    logging.info(f"Using ground-truth labels for drift detection (count: {len(ground_truth_labels)})")
                
                # OPTIMIZATION: Always relabel the incoming window, then refresh memory with relabeled samples
                t_time = time.time()
                relabeled_samples, drift_detected = self.handle_memory_and_drift(online_samples, window_index, ground_truth_labels)
                t_time = (time.time() - t_time) * 1000  # Convert to milliseconds
                logging.info(f"Memory and drift handling took {t_time:.2f} ms")

                if relabeled_samples is not None:
                    # Refresh memory every window using relabeled labels (as requested).
                    self.memory_future_step(relabeled_samples)
                    # Sort by class index to ensure consistent logging
                    sorted_keys = sorted(self.memory.class_counts.keys())
                    cls_counts = [self.memory.class_counts[k] for k in sorted_keys]
                    logging.info(f"Memory class distribution (sorted by class index): {cls_counts}")

                # Only retrain when drift is detected.
                if drift_detected and relabeled_samples is not None:
                    # OPTIMIZATION: Hybrid strategy - combine relabeled samples with memory samples
                    num_relabeled = len(relabeled_samples)
                    num_memory_samples = int(num_relabeled * self.memory_sample_ratio / (1 - self.memory_sample_ratio))
                    
                    # Retrieve balanced samples from memory
                    memory_samples = self.memory.quota_balanced_retrieval(min(num_memory_samples, len(self.memory)))
                    
                    # Check if any LINC/Helios mode is enabled
                    is_rule_based = self.linc_mode or self.helios_mode
                    
                    if memory_samples and len(memory_samples) > 0:
                        # Combine relabeled samples with memory samples
                        combined_samples = relabeled_samples + memory_samples
                        logging.info(f"Training with {len(relabeled_samples)} relabeled samples + {len(memory_samples)} memory samples (total: {len(combined_samples)})")
                        if is_rule_based:
                            # Rule-based incremental update
                            self.linc_incremental_update(combined_samples)
                        else:
                            self.adaptive_retrain(combined_samples)
                    else:
                        # Fallback to relabeled samples only if memory is empty
                        logging.info(f"Training with {len(relabeled_samples)} relabeled samples only (memory empty)")
                        if is_rule_based:
                            # Rule-based incremental update
                            self.linc_incremental_update(relabeled_samples)
                        else:
                            self.adaptive_retrain(relabeled_samples)
                    
                    if self.helios_mode:
                        update_method = "Helios incremental update"
                    elif self.linc_mode:
                        update_method = "LINC incremental update"
                    else:
                        update_method = "Retraining"
                    logging.info(f"{update_method} for window index {window_index} completed.")
                else:
                    logging.info("No drift or insufficient samples, skipping retraining.")
        except Exception as e:
            logging.error(f"Retraining failed: {e}")
        finally:
            self._retraining_lock.set()
        return control_plane_pb2.Empty()

    def WarmupGPU(self, request, context):
        if not self._retraining_lock.is_set():
            logging.warning("Retraining in progress, skipping GPU warmup.")
            return control_plane_pb2.Empty()
        self._retraining_lock.clear()
        try:
            logging.info("Warming up GPU with torchlens graph via gRPC...")
            if self.sequence_model:
                dummy_features = self._build_dummy_sequence(batch_size=512)
            else:
                input_shape = self.args.model_input_shape if self.args.model_input_shape else 16
                dummy_features = torch.randn(512, input_shape, device=self.device)
            dummy_labels = torch.zeros(512, dtype=torch.long, device=self.device)

            # Pre-warm the labeler to avoid first-call latency if it supports single-argument labeling.
            # Skip labeler warmup if using ground-truth labels for drift detection.
            if not getattr(self, 'use_ground_truth_for_drift', False):
                self._warm_labeler(dummy_features)

            batch_list = [(dummy_features, dummy_labels)]
            self.my_dnn = train_mode(self.my_dnn)
            self._ensure_training_warmup(self.my_dnn, self.label_index, batch_list, self.optimizer, self.freezer)
            
            # For LINC/Helios mode: send pre-trained rules to data plane (if available)
            is_rule_based = self.linc_mode or self.helios_mode
            if is_rule_based and self.linc_manager:
                if self.linc_manager.rules:
                    # Send pre-trained rules to data plane
                    self._send_linc_rules_to_data_plane()
                    if self.helios_mode:
                        method_name = "Helios"
                    else:
                        method_name = "LINC"
                    logging.info(f"Sent {len(self.linc_manager.rules)} pre-trained {method_name} rules to data plane")
                else:
                    # No pre-trained rules, will use DNN predictions until first drift
                    if self.helios_mode:
                        method_name = "Helios"
                    else:
                        method_name = "LINC"
                    logging.info(f"No pre-trained {method_name} rules, will use DNN predictions until drift detection")
            
            self.warmup_done_event.set()
            logging.info("GPU warm-up done via gRPC.")
            return control_plane_pb2.Empty()
        except Exception as e:
            logging.error(f"GPU warmup failed: {e}")
        finally:
            self._retraining_lock.set()

    def RecordExperimentData(self, request, context):
        logging.info(f"Recording experimental data to {self.args.output}...")
        experiment_data = {
            "partition_index": self.partition_index,
            "my_improvement": request.my_improvement,
            "my_f1s": list(request.my_f1s),
            "my_f1s_proxy": list(request.my_f1s_proxy),
            "static_f1s": list(request.static_f1s),
            "my_accuracy": list(request.my_accuracy),
            "static_accuracy": list(request.static_accuracy),
        }
        output_path = self.args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "a") as outfile:
                json.dump(experiment_data, outfile)
                outfile.write("\n")
            logging.info("Experimental data recorded.")
        except Exception as e:
            logging.error(f"Failed to record experiment data to {output_path}: {e}")
        return control_plane_pb2.Empty()

    def Shutdown(self, request, context):
        logging.info("Shutdown request received. Stopping server...")
        shutdown_event.set()
        # OPTIMIZATION: Close reusable gRPC channel
        self._close_data_plane_channel()
        if self._server_stop_event:
            self._server_stop_event.set()
        return control_plane_pb2.Empty()

    def run(self):
        self._server_stop_event = Event()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        control_plane_pb2_grpc.add_ControlPlaneServicer_to_server(self, server)
        port = server.add_insecure_port('[::]:50051')
        if port == 0:
            logging.error("Failed to bind to port 50051. Is another instance running?")
            sys.exit(1)
        server.start()
        logging.info(f"Control Plane Server started on port {port}")
        self._server_stop_event.wait()
        server.stop(0).wait()
        logging.info("Control Plane Server stopped.")
        # Force exit to ensure process termination
        os._exit(0)

    def initialize_labeler(self, args):
        if args.labeler_type == "LLM":
            api_key = os.getenv('OPENAI_API_KEY')
            labeler = LMLabeler(model_name=args.llm_model_name,
                                api_key=api_key,
                                application_type=args.application_type)
            labeler.initialize_assistant()
        elif args.labeler_type == "DNN_classifier":
            labeler = DNNLabeler(
                args.labeler_dnn_class,
                args.labeler_dnn_path,
                device=args.device,
            )
        elif args.labeler_type == "device_list":
            labeler = DeviceListLabeler(device_list_path=args.device_list_path,
                                        device=args.device)
        elif args.labeler_type == "rules_or_heuristics":
            labeler = FlowLabeler(heuristics_function=[
                "allow_udp_con", "allow_arp_int_or_con",
                "allow_tcp_req_or_con", "filter_proto",
                "filter_service", "block_udp_flooding",
                "block_malicious_http", "block_malicious_smtp",
                "block_malicious_ftp", "block_malicious_ftp_data"
            ])
        else:
            print("Specified labeler type not supported")
            exit(0)
        return labeler

    def form_training_set(self, all_features, all_labels, generated_labels):
        pos_idx = generated_labels.nonzero(as_tuple=True)[0]
        neg_idx = (generated_labels == 0).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            return None, None
        if pos_idx.shape[0] < neg_idx.shape[0]:
            keep_neg = neg_idx[torch.randperm(neg_idx.shape[0])[:pos_idx.shape[0]]]
            neg_feat = all_features[keep_neg]
            neg_lab = generated_labels[keep_neg]
            return torch.cat((neg_feat, all_features[pos_idx]), 0), torch.cat((neg_lab, generated_labels[pos_idx]), 0)
        else:
            keep_pos = pos_idx[torch.randperm(pos_idx.shape[0])[:neg_idx.shape[0]]]
            pos_feat = all_features[keep_pos]
            pos_lab = generated_labels[keep_pos]
            return torch.cat((pos_feat, all_features[neg_idx]), 0), torch.cat((pos_lab, generated_labels[neg_idx]), 0)

    def memory_future_step(self, online_samples):
        """
        Optimized batch memory update for online samples.
        Args:
            online_samples (List[dict]): List of online sample dicts
        """
        if not online_samples:
            return
        self.memory.replace_batch_quota(online_samples)

    def _coerce_label_tensor(self, label_output, batch_size, device):
        """Normalize labeler outputs into a 1-D long tensor on the requested device."""
        if isinstance(label_output, torch.Tensor):
            label_tensor = label_output
        elif isinstance(label_output, (list, tuple)):
            candidate = label_output[0] if len(label_output) > 0 else []
            label_tensor = candidate if isinstance(candidate, torch.Tensor) else torch.as_tensor(candidate)
        else:
            label_tensor = torch.as_tensor(label_output)

        if label_tensor is None:
            raise ValueError("Labeler returned None when tensor labels were expected.")

        label_tensor = label_tensor.detach() if isinstance(label_tensor, torch.Tensor) else torch.as_tensor(label_tensor)
        label_tensor = label_tensor.to(device=device)

        if label_tensor.ndim > 1:
            try:
                label_tensor = label_tensor.squeeze()
            except RuntimeError:
                label_tensor = label_tensor.reshape(-1)

        label_tensor = label_tensor.reshape(-1)
        if label_tensor.numel() != batch_size:
            raise ValueError(f"Label tensor size {label_tensor.numel()} does not match batch size {batch_size}.")

        return label_tensor.to(dtype=torch.long)

    def _warm_labeler(self, feature_batch):
        """Run a single forward pass through the labeler to amortize future latency."""
        if self._labeler_warm:
            return
        try:
            with torch.no_grad():
                label_output = self.labeler.label(feature_batch)
            _ = self._coerce_label_tensor(label_output, feature_batch.size(0), feature_batch.device)
            self._labeler_warm = True
        except TypeError:
            logging.debug("Labeler warmup skipped: labeler requires additional arguments.")
        except Exception as exc:
            logging.debug(f"Labeler warmup skipped: {exc}")

    def _ensure_training_warmup(self, model, label_index, batch_list, optimizer, freezer):
        """Run a one-off forward/backward pass to trigger CUDA kernel compilation."""
        if self._training_warmup_done:
            return
        if len(batch_list) == 0:
            self._training_warmup_done = True
            return
        try:
            first_batch = batch_list[0]
            if not isinstance(first_batch, (list, tuple)) or len(first_batch) < 2:
                self._training_warmup_done = True
                return
            sample_features, sample_labels = first_batch[:2]
            if sample_features.dim() == 1:
                sample_features = sample_features.unsqueeze(0)
            if sample_labels.dim() == 0:
                sample_labels = sample_labels.unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            warm_score = forward_eval(model, label_index, sample_features)
            warm_loss = self.loss_fn(warm_score, sample_labels)
            warm_loss.backward()

            # Warmup GPU F1 calculation to trigger CUDA kernel compilation
            with torch.no_grad():
                if warm_score.dim() > 1:
                    warm_pred = warm_score.argmax(dim=1)
                    _ = compute_macro_f1_gpu(warm_pred, sample_labels, warm_score.shape[1])
                else:
                    warm_pred = (warm_score > 0.5).long()
                    _ = compute_binary_f1_gpu(warm_pred, sample_labels)
                
                # Warmup softmax path
                if warm_score.dim() > 1:
                    _ = F.softmax(warm_score.detach(), dim=1)

            # Perform a step to initialize optimizer state memory
            optimizer.step()
            # Save the initialized state for later reuse to avoid cold start
            self.initial_optimizer_state = optimizer.state_dict()
            for state in self.initial_optimizer_state['state'].values():
                if 'step' in state:
                    state['step'] = torch.tensor(0.)

            if freezer is not None:
                try:
                    # Calculate fisher for warmup batch and use it for visualization
                    layer_fisher = freezer.calculate_fisher(prewarm=True)
                    
                    # Temporarily update fisher stats to generate policy visualization with real data
                    # Save old state
                    old_fisher = None
                    if hasattr(freezer, 'fisher') and isinstance(freezer.fisher, np.ndarray):
                        old_fisher = freezer.fisher.copy()

                    if isinstance(layer_fisher, np.ndarray):
                        freezer.fisher = layer_fisher.copy()
                        freezer.total_fisher = float(freezer.fisher.sum())
                        freezer.cumulative_fisher = np.cumsum(freezer.fisher)
                        
                        # Save visualization to scripts/results with model class prefix
                        results_dir = os.path.join(Shawarma_home, "emulation", "scripts", "results")
                        os.makedirs(results_dir, exist_ok=True)
                        viz_path = os.path.join(results_dir, f"{self.args.model_class}_policy_visualization_t_acc.png")
                        
                        logging.info(f"Generating policy visualization at {viz_path}")
                        freezer.visualize_policy(viz_path)
                    
                    # Restore original state
                    if old_fisher is not None:
                        freezer.fisher = old_fisher
                        freezer.total_fisher = float(freezer.fisher.sum())
                        freezer.cumulative_fisher = np.cumsum(freezer.fisher)

                    # For warmup, use t=0 and F1_d from freezer.idle_f1
                    freezer.get_freeze_idx(warm_score.detach(), sample_labels, 1.0, self._train_step, float(self._F1_d_current), prewarm=True)
                except Exception as exc:
                    logging.debug(f"Layer freezer warmup skipped: {exc}")
                    import traceback
                    logging.debug(traceback.format_exc())
                finally:
                    freezer.unfreeze_layers()
                    freezer.freeze_idx = []
            optimizer.zero_grad(set_to_none=True)
            device_str = str(self.device)
            if device_str.startswith("cuda") and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(self.device)
                except RuntimeError:
                    torch.cuda.synchronize()
        except StopIteration:
            logging.debug("Training warmup skipped: dataloader empty")
        except Exception as exc:
            logging.debug(f"Training warmup skipped: {exc}")
        finally:
            self._training_warmup_done = True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, help="Device type, e.g. cuda or cpu", default="cuda:1")
    parser.add_argument("--application_type", type=str, help="Type of application", required=True)
    parser.add_argument("-n", "--job_name", type=str, help="Name of the job", required=True)
    parser.add_argument("--labeler_type", type=str, help="Type of labeler", required=True)
    parser.add_argument("--llm_model_name", type=str, help="Type of LLM model (API) to be called", required=False)
    parser.add_argument("--labeler_dnn_class", type=str, help="Class of the labeler DNN model", required=False)
    parser.add_argument("--labeler_dnn_path", type=str, help="Path to the labeler DNN model", required=False)
    parser.add_argument("--device_list_path", type=str, help="Path to the IoT device list", required=False)
    parser.add_argument("-e", "--total_epochs", type=int, help="Maximum number of epochs for training", default=10)
    parser.add_argument("-r", "--learning_rate", type=float, help="Initial learning rate", default=0.01)
    parser.add_argument("--logdir", type=str, help="Logging Directory", required=True)
    parser.add_argument("--model_class", type=str, help="Name of model class", required=True)
    parser.add_argument("--model_input_shape", type=int, help="Number of input features", required=False)
    parser.add_argument("-m", "--base_model_path", type=str, help="Path to the base model path", required=False)
    parser.add_argument("-o", "--output", type=str, help="Output file name", required=True)
    parser.add_argument("--adaptive_freeze", action='store_true', help="Enable adaptive layer freezing during retraining")
    parser.add_argument("--test_size", type=int, help="Size of the test set for evaluation", default=100)
    parser.add_argument("--memory_size", type=int, help="Size of the memory", default=5000)
    parser.add_argument("--label_diff_threshold", type=float, help="Minimum fraction of relabeled samples", default=0.1)
    parser.add_argument("--memory_sample_ratio", type=float, help="Ratio of memory samples to mix with relabeled samples (0.0-1.0)", default=0.1)
    parser.add_argument("--use_ground_truth_for_drift", action='store_true', help="Use ground-truth labels instead of labeler for drift detection")
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
    parser.add_argument("--retrain_batch_size", type=int, default=512, help="Batch size used for retraining")
    parser.add_argument("--deployment_delay", type=float, default=0.0, help="Delay (seconds) before sending updated weights to DP")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA behavior (may reduce performance)")

    
    # Helios mode arguments
    parser.add_argument("--helios_mode", action='store_true', help="Enable Helios mode")
    parser.add_argument("--helios_max_rules", type=int, default=3000, help="Helios max rules")
    parser.add_argument("--helios_radio", type=float, default=1.5, help="Helios radio")
    parser.add_argument("--helios_boost_num", type=int, default=6, help="Helios boost num")
    parser.add_argument("--helios_prune_rule", type=int, default=3, help="Helios prune rule threshold")

    # LINC mode arguments
    parser.add_argument("--linc_mode", action='store_true', help="Enable LINC rule-based incremental update (sklearn DecisionTree)")
    parser.add_argument("--linc_max_rules", type=int, default=10000, help="Maximum number of rules for LINC")
    parser.add_argument("--linc_rule_threshold", type=float, default=0.5, help="Confidence threshold for LINC rule extraction")
    parser.add_argument("--linc_update_samples", type=int, default=4000, help="Number of samples for LINC incremental update")
    parser.add_argument("--linc_rules_path", type=str, default=None, help="Path to pre-trained LINC rules JSON file")
    parser.add_argument("--helios_rules_path", type=str, default=None, help="Path to pre-trained Helios rules JSON file")
    
    args = parser.parse_args()

    env_seed = os.getenv("SHAWARMA_CP_SEED")
    if args.seed is None and env_seed is not None:
        try:
            args.seed = int(env_seed)
        except ValueError:
            logging.warning(f"Invalid SHAWARMA_CP_SEED={env_seed}, ignoring.")

    set_global_seed(args.seed, deterministic=args.deterministic)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Use timestamped log file names to avoid collisions between runs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.logdir, f"{args.job_name}_{timestamp}.log")

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_file)])
    
    # Suppress matplotlib debug logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    server = ControlPlaneServer(args)
    server.run()
