"""
Helios Manager - Wrapper for HeliosSystem to integrate with control plane.

This module provides a simplified interface compatible with the existing
LINC manager pattern, while using the full Helios Prototype Network implementation.
"""

import numpy as np
import logging
import sys
import torch
from pathlib import Path
from typing import List, Tuple, Optional

# Add models to path to import HeliosSystem
models_dir = Path(__file__).resolve().parents[1] / 'src' / 'models'
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

try:
    from helios_model import HeliosSystem, HeliosRule
except ImportError:
    logging.error("Could not import HeliosSystem from helios_model. Ensure helios_model.py exists.")
    raise

class HeliosManager:
    """
    Manager for Helios Prototype Network based rule generation.
    
    This class provides a simplified interface compatible with LincManager,
    while internally using the full HeliosSystem with Prototype Network.
    
    For lightweight deployment (no prototype network training), set use_prototype_network=False
    to use simple clustering-based rule generation.
    """
    
    def __init__(self, max_rules: int = 1000, radio: float = 1.0, boost_num: int = 6,
                 prune_rule_threshold: int = 3, use_prototype_network: bool = True,
                 embedding_dim: int = 64, hidden_dim: int = 128, 
                 num_prototypes_per_class: int = 5, device: str = 'cpu'):
        
        self.max_rules = max_rules
        self.radio = radio
        self.boost_num = boost_num
        self.prune_rule_threshold = prune_rule_threshold
        self.use_prototype_network = use_prototype_network
        self.device = device
        
        # Prototype network parameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        
        # Rules storage
        self.rules: List[HeliosRule] = []
        
        # HeliosSystem instance (lazy initialization)
        self._helios_system = None
        self._input_dim = None
        self._num_classes = None
    
    def _init_helios_system(self, input_dim: int, num_classes: int):
        """Initialize HeliosSystem with proper dimensions."""
        if self._helios_system is not None:
            return
        
        try:
            self._helios_system = HeliosSystem(
                input_dim=input_dim,
                num_classes=num_classes,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_prototypes_per_class=self.num_prototypes_per_class,
                max_rules=self.max_rules,
                radio=self.radio,
                boost_num=self.boost_num,
                prune_threshold=self.prune_rule_threshold,
                device=self.device if self.device != 'cpu' else 'cuda' if torch.cuda.is_available() else 'cpu'
            )
            self._input_dim = input_dim
            self._num_classes = num_classes
            logging.info(f"Initialized HeliosSystem: input_dim={input_dim}, num_classes={num_classes}, device={self._helios_system.device}")
        except ImportError as e:
            logging.warning(f"Could not import HeliosSystem: {e}. Using fallback clustering.")
            self.use_prototype_network = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """
        Initial training using Prototype Network and rule generation.
        
        Args:
            X: Training features (N, D)
            y: Training labels (N,)
            epochs: Training epochs for prototype network
        """
        logging.info(f"Helios: Initial fitting with {len(X)} samples")
        
        input_dim = X.shape[1]
        # Fix: Use max(y) + 1 to handle sparse labels (e.g. 0, 5) without crashing
        num_classes = int(np.max(y)) + 1
        
        if self.use_prototype_network:
            self._init_helios_system(input_dim, num_classes)
            
            if self._helios_system is not None:
                # Use full HeliosSystem
                self._helios_system.fit(X, y, epochs=epochs)
                self._sync_rules_from_system()
                return
        
        # Fallback: simple clustering-based rule generation
        self._fit_clustering(X, y)
    
    def _fit_clustering(self, X: np.ndarray, y: np.ndarray):
        """Fallback clustering-based rule generation (no prototype network)."""
        from sklearn.cluster import KMeans
        
        # Don't clear existing rules - append to them
        classes = np.unique(y)
        new_rules = []
        
        for c in classes:
            X_c = X[y == c]
            if len(X_c) < 2:
                continue
            
            # More clusters for better coverage - at least 1 per 15 samples
            n_clusters = min(len(X_c) // 15, 50)
            n_clusters = max(3, n_clusters)  # At least 3 clusters
            
            clustering = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit(X_c)
            
            for label in range(n_clusters):
                cluster_mask = (clustering.labels_ == label)
                X_cluster = X_c[cluster_mask]
                
                if len(X_cluster) < 1:
                    continue
                
                mins = np.min(X_cluster, axis=0)
                maxs = np.max(X_cluster, axis=0)
                
                intervals = []
                for k in range(len(mins)):
                    span = maxs[k] - mins[k]
                    if span == 0:
                        span = 1e-6
                    center = (maxs[k] + mins[k]) / 2.0
                    half_width = (span * self.radio) / 2.0
                    intervals.append([center - half_width, center + half_width])
                
                rule = HeliosRule(int(c), intervals)
                rule.support = len(X_cluster)
                new_rules.append(rule)
        
        self.rules.extend(new_rules)
        self._prune_and_balance(X)
        logging.info(f"Helios (clustering): Generated {len(new_rules)} rules, total: {len(self.rules)}")

    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Incremental update with new samples.
        
        If no rules exist yet, performs initial training (fit) first.
        Otherwise, follows original Helios incremental update logic.
        
        Args:
            X_new: New features (N, D)
            y_new: New labels (N,)
        """
        logging.info(f"Helios: Incremental update with {len(X_new)} samples, current rules: {len(self.rules)}")
        
        # If no rules exist, perform initial training first
        if len(self.rules) == 0:
            logging.info("Helios: No existing rules, performing initial training...")
            self.fit(X_new, y_new, epochs=30)  # More epochs for initial training
            return
        
        # Initialize HeliosSystem if not done yet
        reinitialized = False
        if self.use_prototype_network:
            input_dim = X_new.shape[1]
            required_classes = int(np.max(y_new)) + 1
            
            # Re-initialize if system handles fewer classes than required
            if self._helios_system is None or (self._num_classes is not None and required_classes > self._num_classes):
                if self._helios_system is not None:
                    logging.info(f"Helios: Expanding num_classes from {self._num_classes} to {required_classes}")
                    # Force re-initialization
                    self._helios_system = None
                
                # Determine new num_classes (keep existing if larger)
                curr_classes = self._num_classes if self._num_classes else 0
                num_classes = max(2, required_classes, curr_classes)
                
                self._init_helios_system(input_dim, num_classes)
                reinitialized = True
                
                # Transfer existing rules to the system
                if self._helios_system is not None and self.rules:
                    self._helios_system.rules = self.rules.copy()
                    logging.info(f"Transferred {len(self.rules)} existing rules to HeliosSystem")
        
        if self.use_prototype_network and self._helios_system is not None:
            if reinitialized:
                # If we re-initialized, we must train the model from scratch on new data
                # because the old model is gone/incompatible.
                logging.info("Helios: System re-initialized, running fit() instead of incremental_update()")
                self._helios_system.fit(X_new, y_new, epochs=30)
                self._sync_rules_from_system()
                return
            else:
                # Use HeliosSystem incremental update
                self._helios_system.incremental_update(X_new, y_new)
                self._sync_rules_from_system()
                return
        
        # Fallback: simple incremental update
        self._incremental_update_clustering(X_new, y_new)
    
    def _incremental_update_clustering(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Fallback incremental update using clustering.
        Follows original Helios logic with optimizations:
        1. Remove rules with low accuracy on new samples (< 70%)
        2. Find residual samples (not correctly classified)
        3. Generate new rules for residual samples
        """
        initial_rule_count = len(self.rules)
        
        # Step 1: Evaluate and remove low-accuracy rules
        rules_to_keep = []
        for rule in self.rules:
            matches = rule.match_batch(X_new)
            if np.any(matches):
                matched_labels = y_new[matches]
                correct = np.sum(matched_labels == rule.class_label)
                total = np.sum(matches)
                accuracy = correct / total if total > 0 else 1.0
                
                # Keep rules with accuracy >= 70%
                if accuracy >= 0.7:
                    rules_to_keep.append(rule)
            else:
                # No matches on new data, keep the rule
                rules_to_keep.append(rule)
        
        removed = initial_rule_count - len(rules_to_keep)
        if removed > 0:
            logging.info(f"Helios (clustering): Removed {removed} low-accuracy rules")
        self.rules = rules_to_keep
        
        # Step 2: Find RESIDUAL samples
        residual_mask = np.ones(len(X_new), dtype=bool)
        for i in range(len(X_new)):
            sample = X_new[i]
            true_label = y_new[i]
            for rule in self.rules:
                if rule.match(sample):
                    if rule.class_label == true_label:
                        residual_mask[i] = False
                    break
        
        X_residual = X_new[residual_mask]
        y_residual = y_new[residual_mask]
        
        logging.info(f"Helios (clustering): {len(X_residual)}/{len(X_new)} residual samples")
        
        # Step 3: Generate rules for residual samples
        if len(X_residual) > 5:
            new_rules = self._generate_clustering_rules(X_residual, y_residual)
            # Merge with new rules having priority (prepend)
            self.rules = new_rules + self.rules
            logging.info(f"Helios (clustering): Generated {len(new_rules)} new rules, total: {len(self.rules)}")
        
        self._prune_and_balance(X_new)
    
    def _generate_clustering_rules(self, X: np.ndarray, y: np.ndarray) -> List[HeliosRule]:
        """Generate rules using clustering for given samples - optimized for tighter bounding boxes."""
        from sklearn.cluster import KMeans
        
        new_rules = []
        classes = np.unique(y)
        
        for c in classes:
            X_c = X[y == c]
            if len(X_c) < 2:
                continue
            
            # More clusters for better coverage - at least 1 per 10 samples
            n_clusters = min(len(X_c) // 10, 50)
            n_clusters = max(3, n_clusters)  # At least 3 clusters
            
            clustering = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit(X_c)
            
            for label in range(n_clusters):
                cluster_mask = (clustering.labels_ == label)
                X_cluster = X_c[cluster_mask]
                
                if len(X_cluster) < 1:
                    continue
                
                mins = np.min(X_cluster, axis=0)
                maxs = np.max(X_cluster, axis=0)
                
                # Tight bounding box with minimal expansion (1% of span)
                intervals = []
                for k in range(len(mins)):
                    span = maxs[k] - mins[k]
                    if span == 0:
                        span = 1e-6
                    expansion = span * 0.01  # 1% expansion for robustness
                    intervals.append([mins[k] - expansion, maxs[k] + expansion])
                
                rule = HeliosRule(int(c), intervals)
                rule.support = len(X_cluster)
                new_rules.append(rule)
        
        return new_rules
    
    def _sync_rules_from_system(self):
        """Sync rules from HeliosSystem to local storage."""
        if self._helios_system is None:
            return
        
        self.rules = self._helios_system.rules
    
    def _prune_and_balance(self, X: np.ndarray):
        """Prune low-support rules and balance across classes - more conservative."""
        # Prune low-support rules - but be conservative
        if self.prune_rule_threshold > 0 and len(X) > 0:
            rules_to_keep = []
            for rule in self.rules:
                matches = rule.match_batch(X)
                rule.support = np.sum(matches)
                # Keep rules with at least some support, or pre-trained rules (support=0 means not tested yet)
                if rule.support >= self.prune_rule_threshold or rule.support == 0:
                    rules_to_keep.append(rule)
            
            removed = len(self.rules) - len(rules_to_keep)
            if removed > 0:
                logging.info(f"Helios: Pruned {removed} low-support rules")
            self.rules = rules_to_keep
        
        # Balance across classes if exceeding max_rules
        if len(self.rules) > self.max_rules:
            class_rules = {}
            for r in self.rules:
                c = r.class_label
                if c not in class_rules:
                    class_rules[c] = []
                class_rules[c].append(r)
            
            # Sort by support (descending)
            for c in class_rules:
                class_rules[c].sort(key=lambda x: x.support, reverse=True)
            
            # Distribute rules evenly across classes
            num_classes = len(class_rules)
            if num_classes == 0:
                return
            
            rules_per_class = self.max_rules // num_classes
            
            pruned_rules = []
            for c in class_rules:
                pruned_rules.extend(class_rules[c][:rules_per_class])
            
            # Fill remaining slots with highest support rules
            remaining = self.max_rules - len(pruned_rules)
            if remaining > 0:
                all_remaining = []
                for c in class_rules:
                    all_remaining.extend(class_rules[c][rules_per_class:])
                all_remaining.sort(key=lambda x: x.support, reverse=True)
                pruned_rules.extend(all_remaining[:remaining])
            
            self.rules = pruned_rules
            logging.info(f"Helios: Balanced to {len(self.rules)} rules")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels using rules.
        
        Following original Helios rule matching logic:
        - Rules are checked in order (first match wins)
        - For samples not matched by any rule, return default class 0
        
        Args:
            X: Input features (N, input_dim)
            
        Returns:
            predictions: Predicted class labels (N,)
            confidences: Prediction confidences (N,)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=np.int64)
        confidences = np.zeros(n_samples, dtype=np.float32)
        
        covered_mask = np.zeros(n_samples, dtype=bool)
        
        conflict_fix = {}
        if self._helios_system is not None and hasattr(self._helios_system, "_conflict_fix"):
            conflict_fix = self._helios_system._conflict_fix or {}

        # Check rules in order (first match wins) with conflict-fix support
        if not conflict_fix:
            for rule in self.rules:
                matches = rule.match_batch(X)
                update_mask = matches & (~covered_mask)
                
                if np.any(update_mask):
                    predictions[update_mask] = rule.class_label
                    confidences[update_mask] = rule.confidence
                    covered_mask[update_mask] = True
                
                if np.all(covered_mask):
                    break
        else:
            for i in range(n_samples):
                if covered_mask[i]:
                    continue
                matched_ids = []
                for ridx, rule in enumerate(self.rules):
                    if rule.match(X[i]):
                        matched_ids.append(rule.id)
                if not matched_ids:
                    continue
                if len(matched_ids) == 1:
                    rule = next(r for r in self.rules if r.id == matched_ids[0])
                    predictions[i] = rule.class_label
                    confidences[i] = rule.confidence
                    covered_mask[i] = True
                else:
                    key = tuple(matched_ids)
                    if key in conflict_fix:
                        predictions[i] = conflict_fix[key]["label"]
                        confidences[i] = 0.5
                        covered_mask[i] = True
                    else:
                        rule = next(r for r in self.rules if r.id == matched_ids[0])
                        predictions[i] = rule.class_label
                        confidences[i] = rule.confidence
                        covered_mask[i] = True
        
        return predictions, confidences

    def to_dict_list(self) -> List[dict]:
        """Serialize rules to list of dicts for JSON/proto compatibility."""
        return [r.to_dict() for r in self.rules]
