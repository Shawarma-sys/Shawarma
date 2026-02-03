"""
LINC Manager - Optimized implementation based on original LINC paper (ICNP 2024)
https://github.com/haolinyan/LINC

Key features from original implementation:
1. SoftTreeClassifier with soft labels from RandomForest
2. Discrete feature handling with equality-based splits
3. K-fold cross-validation for soft label generation
4. Proper incremental update using Mousika-style rule extraction

Optimizations for RNN/sequence models:
1. Faster soft label generation using sklearn's DecisionTreeClassifier
2. Better class balancing in rule pruning
3. Reduced tree depth for faster rule extraction
4. Optimized incremental update to reduce latency
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import logging
import copy
from collections import defaultdict
import time
import random


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


# ============================================================================
# Decision Node for SoftTree
# ============================================================================

class DecisionNode:
    """A node in the decision tree."""
    
    def __init__(self, feature=-1, threshold=None, label=None, 
                 label_dict=None, true_branch=None, false_branch=None):
        self.feature = feature
        self.threshold = threshold
        self.label = label  # If not None, this is a leaf node
        self.label_dict = label_dict  # Soft label distribution
        self.true_branch = true_branch
        self.false_branch = false_branch
    
    def is_leaf(self):
        return self.label is not None


# ============================================================================
# Soft Label Utilities
# ============================================================================

def soft_label_dict(label):
    """
    Convert soft labels (2D array with probabilities) to a dictionary.
    label: shape (num_samples, num_classes)
    """
    assert np.ndim(label) == 2
    label = np.array(label)
    result = {}
    for i in range(label.shape[1]):
        result[i] = np.sum(label[:, i])
    return result


def soft_voting(label_dict):
    """Return the class with highest soft vote."""
    winner_key = list(label_dict.keys())[0]
    for key in label_dict:
        if label_dict[key] > label_dict[winner_key]:
            winner_key = key
        elif label_dict[key] == label_dict[winner_key]:
            winner_key = np.random.choice([key, winner_key])
    return winner_key


def soft_gini(label):
    """
    Calculate Gini impurity for soft labels.
    label: shape (num_samples, num_classes)
    """
    assert np.ndim(label) == 2
    label = np.array(label)
    total = 0
    for i in range(label.shape[1]):
        p = np.sum(label[:, i]) / label.shape[0]
        total += p ** 2
    return 1 - total


def kfold_split_stratified(y, num_fold=2, random_state=42):
    """
    Stratified K-fold split that handles class imbalance.
    """
    # Get unique labels and their counts
    unique_labels = np.unique(y)
    label_indices = {label: np.where(y == label)[0].tolist() for label in unique_labels}
    
    # Shuffle indices for each label
    np.random.seed(random_state)
    for label in label_indices:
        np.random.shuffle(label_indices[label])
    
    # Create folds
    folds = [[] for _ in range(num_fold)]
    for label in unique_labels:
        indices = label_indices[label]
        n = len(indices)
        fold_size = n // num_fold
        for i in range(num_fold):
            if i == num_fold - 1:
                folds[i].extend(indices[i * fold_size:])
            else:
                folds[i].extend(indices[i * fold_size:(i + 1) * fold_size])
    
    # Yield train/test splits
    for i in range(num_fold):
        test_idx = np.array(folds[i])
        train_idx = []
        for j in range(num_fold):
            if j != i:
                train_idx.extend(folds[j])
        yield np.array(train_idx), test_idx


def produce_soft_labels(X, y, round_num=1, fold_num=2, k=1, n_estimators=3, 
                        min_samples_leaf=5, random_state=2023):
    """
    Produce soft labels using RandomForest with k-fold cross-validation.
    This is a key component of the original LINC implementation.
    
    OPTIMIZED: Reduced complexity for faster execution on sequence model embeddings.
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Hard labels, shape (n_samples,)
        round_num: Number of rounds to average
        fold_num: Number of folds for cross-validation
        k: Weight for hard labels vs soft labels
        n_estimators: Number of trees in RandomForest
        min_samples_leaf: Minimum samples per leaf
        random_state: Random seed
    
    Returns:
        soft_label: shape (n_samples, n_classes), probability distribution over classes
    """
    n_samples = len(y)
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # Create a mapping from original class labels to 0-indexed
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_indexed = np.array([class_to_idx[c] for c in y])
    
    soft_label = np.zeros((n_samples, n_classes))
    
    # OPTIMIZATION: For large datasets, use simpler approach
    if n_samples > 5000:
        # Use a single RandomForest fit instead of k-fold for speed
        try:
            clf = RandomForestClassifier(
                n_estimators=min(n_estimators, 5),
                max_depth=10,  # Limit depth for speed
                min_samples_leaf=max(min_samples_leaf, 10),
                criterion='gini',
                random_state=random_state,
                n_jobs=-1
            )
            clf.fit(X, y_indexed)
            soft_label = clf.predict_proba(X)
            
            # Ensure soft_label has correct shape
            if soft_label.shape[1] != n_classes:
                full_soft_label = np.zeros((n_samples, n_classes))
                for i, class_idx in enumerate(clf.classes_):
                    full_soft_label[:, class_idx] = soft_label[:, i]
                soft_label = full_soft_label
                
        except Exception as e:
            logging.warning(f"LINC: Fast soft label generation failed: {e}, using hard labels")
            for i, idx in enumerate(y_indexed):
                soft_label[i, idx] = 1.0
    else:
        # Original k-fold approach for smaller datasets
        for r in range(round_num):
            for train_idx, test_idx in kfold_split_stratified(y, num_fold=fold_num, 
                                                              random_state=random_state + r):
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                
                X_train, y_train = X[train_idx], y_indexed[train_idx]
                X_test = X[test_idx]
                
                # Train RandomForest
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    criterion='gini',
                    random_state=random_state + r,
                    n_jobs=-1
                )
                
                # Handle case where training set might not have all classes
                try:
                    clf.fit(X_train, y_train)
                    pred_prob = clf.predict_proba(X_test)
                    
                    # Map predictions back to full class space
                    for i, class_idx in enumerate(clf.classes_):
                        soft_label[test_idx, class_idx] += pred_prob[:, i]
                except Exception as e:
                    logging.warning(f"RF fold failed: {e}")
                    continue
        
        # Normalize by number of rounds
        if round_num > 0:
            soft_label /= round_num
    
    # Combine with hard labels: soft_label = (soft * k + hard) / (k + 1)
    hard_label = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        hard_label[i, y_indexed[i]] = 1
    
    soft_label = (soft_label * k + hard_label) / (k + 1)
    
    return soft_label, unique_classes


# ============================================================================
# SoftTreeClassifier - Core of LINC
# ============================================================================

class SoftTreeClassifier:
    """
    Soft Label Decision Tree Classifier.
    Uses soft labels for training to improve generalization.
    Supports both discrete ('d') and continuous ('c') features.
    """
    
    def __init__(self, n_features=None, min_sample_leaf=5):
        self.root = None
        self.min_sample_leaf = min_sample_leaf
        self.n_features = n_features  # Number of features to consider at each split
        self.features_attr = None  # 'd' for discrete, 'c' for continuous
        self.num_classes = None
    
    def fit(self, X, y, features_attr=None):
        """
        Fit the tree using soft labels.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Soft labels, shape (n_samples, n_classes)
            features_attr: List of 'd' (discrete) or 'c' (continuous) for each feature
        """
        self.features_attr = features_attr
        X = np.array(X)
        y = np.array(y)
        
        self.num_classes = y.shape[1] if y.ndim == 2 else len(np.unique(y))
        
        if self.n_features is None or self.n_features == 'all':
            self.n_features = X.shape[1]
        elif self.n_features == 'sqrt':
            self.n_features = int(np.sqrt(X.shape[1]))
        elif self.n_features == 'half':
            self.n_features = int(0.5 * X.shape[1])
        
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y):
        """Recursively build the decision tree."""
        y_dict = soft_label_dict(y)
        
        # Check if all samples belong to one class (pure node)
        max_class_ratio = np.max(np.sum(y, axis=0)) / len(y)
        if max_class_ratio > 0.99:
            return DecisionNode(label_dict=y_dict, label=soft_voting(y_dict))
        
        # Find candidate features with more than one unique value
        candidate_features = [i for i in range(X.shape[1]) 
                              if len(np.unique(X[:, i])) > 1]
        
        if not candidate_features:
            return DecisionNode(label_dict=y_dict, label=soft_voting(y_dict))
        
        # Randomly select features to consider
        n_select = min(self.n_features, len(candidate_features))
        selected_features = np.random.choice(candidate_features, n_select, replace=False)
        
        # Find best split
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        current_gini = soft_gini(y)
        
        for feat in selected_features:
            col = X[:, feat]
            unique_vals = np.unique(col)
            attr = self.features_attr[feat] if self.features_attr else 'd'
            
            if attr == 'd' or len(unique_vals) == 1:
                # Discrete: split on each unique value
                thresholds = unique_vals
            else:
                # Continuous: split on midpoints
                # Optimization: limit number of thresholds to check
                if len(unique_vals) > 50:
                    indices = np.linspace(0, len(unique_vals)-2, 50, dtype=int)
                    thresholds = [(unique_vals[i] + unique_vals[i + 1]) / 2 for i in indices]
                else:
                    thresholds = [(unique_vals[i] + unique_vals[i + 1]) / 2 
                                  for i in range(len(unique_vals) - 1)]
            
            for thresh in thresholds:
                true_idx, false_idx = self._split(X, feat, thresh, attr)
                
                if len(true_idx) == 0 or len(false_idx) == 0:
                    continue
                
                p = len(true_idx) / len(X)
                next_gini = p * soft_gini(y[true_idx]) + (1 - p) * soft_gini(y[false_idx])
                gain = current_gini - next_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh
        
        if best_feature is None:
            return DecisionNode(label_dict=y_dict, label=soft_voting(y_dict))
        
        attr = self.features_attr[best_feature] if self.features_attr else 'd'
        true_idx, false_idx = self._split(X, best_feature, best_threshold, attr)
        
        if len(true_idx) == 0 or len(false_idx) == 0:
            return DecisionNode(label_dict=y_dict, label=soft_voting(y_dict))
        
        # Create branches
        if len(true_idx) <= self.min_sample_leaf:
            true_branch = DecisionNode(label_dict=soft_label_dict(y[true_idx]),
                                       label=soft_voting(soft_label_dict(y[true_idx])))
        else:
            true_branch = self._build_tree(X[true_idx], y[true_idx])
        
        if len(false_idx) <= self.min_sample_leaf:
            false_branch = DecisionNode(label_dict=soft_label_dict(y[false_idx]),
                                        label=soft_voting(soft_label_dict(y[false_idx])))
        else:
            false_branch = self._build_tree(X[false_idx], y[false_idx])
        
        return DecisionNode(
            feature=best_feature,
            threshold=best_threshold,
            label_dict=y_dict,
            true_branch=true_branch,
            false_branch=false_branch
        )
    
    def _split(self, X, feature, threshold, attr='d'):
        """Split data based on feature and threshold."""
        if attr == 'd':
            # Discrete: exact equality
            true_idx = np.where(X[:, feature] == threshold)[0]
            false_idx = np.where(X[:, feature] != threshold)[0]
        else:
            # Continuous: >= threshold
            true_idx = np.where(X[:, feature] >= threshold)[0]
            false_idx = np.where(X[:, feature] < threshold)[0]
        return true_idx.tolist(), false_idx.tolist()
    
    def predict(self, X):
        """Predict class labels for samples."""
        X = np.array(X)
        if X.ndim == 1:
            return self._predict_sample(X, self.root)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def _predict_sample(self, x, node):
        """Predict for a single sample."""
        if node.is_leaf():
            return node.label
        
        feat_val = x[node.feature]
        attr = self.features_attr[node.feature] if self.features_attr else 'd'
        
        if attr == 'd':
            if feat_val == node.threshold:
                return self._predict_sample(x, node.true_branch)
            else:
                return self._predict_sample(x, node.false_branch)
        else:
            if feat_val >= node.threshold:
                return self._predict_sample(x, node.true_branch)
            else:
                return self._predict_sample(x, node.false_branch)
    
    def extract_rules(self, class_mapping=None):
        """Extract rules from the tree as a list of LincRule objects."""
        rules = []
        self._extract_rules_recursive(self.root, [], rules, class_mapping)
        return rules
    
    def _extract_rules_recursive(self, node, path, rules, class_mapping=None):
        """Recursively extract rules from tree."""
        if node.is_leaf():
            if path:  # Only add if there are conditions
                class_label = node.label
                if class_mapping is not None:
                    class_label = class_mapping[node.label]
                
                # Calculate confidence from label distribution
                if node.label_dict:
                    total = sum(node.label_dict.values())
                    confidence = node.label_dict.get(node.label, 0) / total if total > 0 else 0.5
                else:
                    confidence = 0.5
                
                rule = LincRule(
                    conditions=path.copy(),
                    class_label=int(class_label),
                    confidence=float(confidence)
                )
                rules.append(rule)
            return
        
        attr = self.features_attr[node.feature] if self.features_attr else 'd'
        
        # True branch
        if attr == 'd':
            path.append((node.feature, node.threshold, 2))  # 2 = equals
        else:
            path.append((node.feature, node.threshold, 1))  # 1 = >=
        self._extract_rules_recursive(node.true_branch, path, rules, class_mapping)
        path.pop()
        
        # False branch
        if attr == 'd':
            path.append((node.feature, node.threshold, 3))  # 3 = not equals
        else:
            path.append((node.feature, node.threshold, 0))  # 0 = <
        self._extract_rules_recursive(node.false_branch, path, rules, class_mapping)
        path.pop()


# ============================================================================
# LINC Rule Definition
# ============================================================================

class LincRule:
    """
    Represents a rule derived from a Decision Tree path.
    """
    
    def __init__(self, conditions, class_label, confidence, sample_indices=None):
        """
        Args:
            conditions: List of (feature_idx, threshold, operator)
                       operator: 0 = equals (for discrete) or < (for continuous)
                                1 = >= (for continuous)
                                2 = not equals (for discrete)
            class_label: Predicted class
            confidence: Rule confidence [0, 1]
            sample_indices: Indices of samples covered by this rule
        """
        self.conditions = conditions
        self.class_label = class_label
        self.confidence = confidence
        self.sample_indices = sample_indices if sample_indices is not None else []
        self.id = id(self)
        self._sample_count = 0
        self._accuracy = 0.0
    
    def match(self, x):
        """Check if a sample matches this rule."""
        for feat_idx, thresh, op in self.conditions:
            val = x[feat_idx]
            if op == 0:  # <
                if not (val < thresh):
                    return False
            elif op == 1:  # >=
                if not (val >= thresh):
                    return False
            elif op == 2:  # ==
                if not (val == thresh):
                    return False
            elif op == 3:  # !=
                if not (val != thresh):
                    return False
        return True
    
    def match_continuous(self, x):
        """Check if sample matches using continuous threshold logic."""
        for feat_idx, thresh, op in self.conditions:
            val = x[feat_idx]
            if op == 0:  # <=
                if val > thresh:
                    return False
            elif op == 1:  # >
                if val <= thresh:
                    return False
        return True
    
    def to_dict(self):
        return {
            'conditions': self.conditions,
            'class': self.class_label,
            'confidence': self.confidence
        }


# ============================================================================
# LINC Manager - Main Class
# ============================================================================

class LincManager:
    """
    LINC Manager - Implementation based on original LINC paper (ICNP 2024)
    
    Two modes available:
    - improved=False: Original LINC with SoftTreeClassifier (LINC Rule-based)
    - improved=True: Optimized LINC with sklearn DecisionTree + better features (LINC Improved)
    
    Key features:
    1. SoftTreeClassifier with soft labels from RandomForest (original mode)
    2. Fast sklearn DecisionTreeClassifier (improved mode)
    3. Better class balancing in rule pruning
    4. Optimized incremental update with conflict detection
    """
    
    def __init__(self, max_rules=1000, min_samples_leaf=3, n_estimators=3, improved=False, seed=2023):
        self.rules = []
        self.max_rules = max_rules
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.allocated_samples = {}
        self.class_mapping = None  # Maps internal indices to original class labels
        self.feature_attr = None  # 'd' for discrete, 'c' for continuous
        self.improved = improved  # Use improved sklearn-based approach
        self._sklearn_tree = None  # sklearn tree for fast prediction (improved mode)
        self._last_fit_time = 0
        self.seed = seed
        setup_seed(self.seed)
    
    def fit(self, X, y):
        """
        Train rules using either SoftTreeClassifier (original) or sklearn DecisionTree (improved).
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Class labels, shape (n_samples,)
        """
        start_time = time.time()
        X = np.array(X)
        y = np.array(y).astype(int)
        
        n_samples = len(X)
        n_classes = len(np.unique(y))
        
        mode_str = "improved" if self.improved else "original"
        logging.info(f"LINC ({mode_str}): Initial fitting with {n_samples} samples, {n_classes} classes")
        
        if n_samples < 10:
            logging.warning("LINC: Too few samples for training")
            return
        
        # Detect feature type (discrete vs continuous)
        self.feature_attr = self._detect_feature_types(X)
        
        # Store class mapping
        unique_classes = np.unique(y)
        self.class_mapping = {i: c for i, c in enumerate(unique_classes)}
        self._reverse_class_mapping = {c: i for i, c in enumerate(unique_classes)}
        
        # Choose fitting method based on mode
        if self.improved:
            self._fit_improved_sklearn(X, y, unique_classes)
        else:
            self._fit_original_soft_tree(X, y, unique_classes)
        
        # Allocate samples to rules
        self._reallocate_samples(X, y)
        
        # Prune if needed
        if len(self.rules) > self.max_rules:
            self._prune_rules(y)
        
        # Add default rule for 100% coverage
        self._add_default_rule(y)
        
        self._last_fit_time = time.time() - start_time
        logging.info(f"LINC ({mode_str}): Extracted {len(self.rules)} rules in {self._last_fit_time:.2f}s")
        self._log_class_distribution()
    
    def _fit_improved_sklearn(self, X, y, unique_classes):
        """
        Improved rule extraction using sklearn DecisionTreeClassifier with class balancing.
        Faster and more effective for sequence model embeddings.
        """
        n_samples = len(X)
        n_classes = len(unique_classes)
        
        # Calculate optimal tree parameters based on data size
        max_depth = min(15, max(5, int(np.log2(n_samples / 10))))
        min_samples = max(self.min_samples_leaf, n_samples // 500)
        
        # Use class weights to handle imbalance
        class_counts = np.bincount(y, minlength=max(y) + 1)
        class_weights = {}
        for c in unique_classes:
            count = class_counts[c] if c < len(class_counts) else 1
            class_weights[c] = n_samples / (n_classes * max(count, 1))
        
        # Train sklearn DecisionTree
        self._sklearn_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples,
            min_samples_split=min_samples * 2,
            class_weight=class_weights,
            random_state=42
        )
        self._sklearn_tree.fit(X, y)
        
        # Extract rules from sklearn tree
        self.rules = self._extract_rules_from_sklearn_tree(X, y)
        
        logging.info(f"LINC (improved): sklearn extraction: depth={max_depth}, min_samples={min_samples}")
    
    def _fit_original_soft_tree(self, X, y, unique_classes):
        """
        Original LINC rule extraction using SoftTreeClassifier with soft labels.
        """
        # Generate soft labels using RandomForest k-fold
        try:
            soft_labels, _ = produce_soft_labels(
                X, y, 
                round_num=1, 
                fold_num=2,
                k=1,
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf
            )
        except Exception as e:
            logging.warning(f"LINC: Soft label generation failed: {e}, using hard labels")
            soft_labels = np.zeros((len(y), len(unique_classes)))
            for i, c in enumerate(y):
                idx = self._reverse_class_mapping.get(c, 0)
                soft_labels[i, idx] = 1.0
        
        # Train SoftTreeClassifier
        tree = SoftTreeClassifier(n_features='all', min_sample_leaf=self.min_samples_leaf)
        tree.fit(X, soft_labels, features_attr=self.feature_attr)
        
        # Extract rules from tree
        self.rules = tree.extract_rules(class_mapping=self.class_mapping)
        
        logging.info(f"LINC (original): SoftTree extraction completed")
    
    def _extract_rules_from_sklearn_tree(self, X, y):
        """Extract rules from sklearn DecisionTreeClassifier."""
        tree = self._sklearn_tree.tree_
        rules = []
        
        def recurse(node_id, path):
            if tree.feature[node_id] == -2:  # Leaf node
                # Get class distribution at this leaf
                class_counts = tree.value[node_id][0]
                total = class_counts.sum()
                if total > 0:
                    predicted_class = np.argmax(class_counts)
                    confidence = class_counts[predicted_class] / total
                    
                    # Only add rule if it has conditions and reasonable confidence
                    if path and confidence > 0.3:
                        # Map back to original class labels
                        original_class = self._sklearn_tree.classes_[predicted_class]
                        rule = LincRule(
                            conditions=path.copy(),
                            class_label=int(original_class),
                            confidence=float(confidence)
                        )
                        rules.append(rule)
                return
            
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            # Left child: feature < threshold (op=0)
            left_path = path + [(feature, threshold, 0)]
            recurse(tree.children_left[node_id], left_path)
            
            # Right child: feature >= threshold (op=1)
            right_path = path + [(feature, threshold, 1)]
            recurse(tree.children_right[node_id], right_path)
        
        recurse(0, [])
        return rules
    
    def _detect_feature_types(self, X):
        """
        Detect whether features are discrete or continuous.
        If a feature is float, treat as continuous.
        Otherwise, if it has few unique values relative to samples, treat as discrete.
        """
        n_samples, n_features = X.shape
        feature_attr = []
        
        for i in range(n_features):
            # Check if float
            if np.issubdtype(X[:, i].dtype, np.floating):
                feature_attr.append('c')
                continue
                
            unique_vals = len(np.unique(X[:, i]))
            # If unique values < 10% of samples or < 20, treat as discrete
            if unique_vals < max(n_samples * 0.1, 20):
                feature_attr.append('d')
            else:
                feature_attr.append('c')
        
        return feature_attr
    
    def incremental_update(self, X_new, y_new):
        """
        Incrementally update rules based on new samples.
        Following original LINC update_streaming.py logic.
        
        OPTIMIZED: 
        1. Limit historical samples to prevent memory explosion
        2. Use fast sklearn mode for large updates
        3. Skip full refit if conflict rate is low
        """
        start_time = time.time()
        X_new = np.array(X_new)
        y_new = np.array(y_new).astype(int)
        
        logging.info(f"LINC: Incremental update with {len(X_new)} samples")
        
        if len(X_new) < 5:
            logging.warning("LINC: Too few samples for update")
            return
        
        # If no rules exist, perform initial training
        if len(self.rules) == 0:
            self.fit(X_new, y_new)
            return
        
        # Detect new classes
        existing_classes = set(r.class_label for r in self.rules)
        new_classes = set(int(c) for c in y_new) - existing_classes
        
        if new_classes:
            logging.info(f"LINC: Detected new classes: {new_classes}")
        
        # Find conflicting rules (rules that misclassify new samples)
        conflict_rules = self._find_conflict_rules(X_new, y_new)
        conflict_rate = len(conflict_rules) / max(len(self.rules), 1)
        
        if conflict_rules:
            logging.info(f"LINC: Found {len(conflict_rules)} conflicting rules, updating...")
        
        # OPTIMIZATION: If conflict rate is very low and no new classes, skip full refit
        if conflict_rate < 0.05 and not new_classes and len(self.rules) > 100:
            logging.info(f"LINC: Low conflict rate ({conflict_rate:.1%}), skipping full refit")
            # Just update allocated samples with new data
            self._update_allocated_samples(X_new, y_new)
            return
        
        # Collect existing samples from allocated_samples (with limit)
        existing_X = []
        existing_y = []
        max_historical = min(10000, len(X_new) * 2)  # Limit historical samples
        
        # Sample from allocated_samples to limit size
        all_historical = []
        for rule_id, samples in self.allocated_samples.items():
            for x, label in samples:
                all_historical.append((x, int(label)))
        
        if len(all_historical) > max_historical:
            # Stratified sampling to maintain class balance
            np.random.shuffle(all_historical)
            all_historical = all_historical[:max_historical]
        
        for x, label in all_historical:
            existing_X.append(x)
            existing_y.append(label)
        
        # Combine with new samples
        if existing_X:
            all_X = np.vstack([existing_X, X_new])
            all_y = np.concatenate([existing_y, y_new])
        else:
            all_X = X_new
            all_y = y_new
        
        # Re-fit with combined data
        self.fit(all_X, all_y)
        
        update_time = time.time() - start_time
        logging.info(f"LINC: Incremental update completed in {update_time:.2f}s")
        
        # Log update results
        self._log_class_distribution()
    
    def _update_allocated_samples(self, X_new, y_new):
        """Update allocated samples without full refit."""
        for i in range(len(X_new)):
            x = X_new[i]
            label = int(y_new[i])
            
            # Find matching rule
            for rule in self.rules:
                if self._rule_matches(rule, x):
                    if rule.id not in self.allocated_samples:
                        self.allocated_samples[rule.id] = []
                    self.allocated_samples[rule.id].append((x, label))
                    break
    
    def _find_conflict_rules(self, X, y):
        """Find rules that incorrectly classify samples."""
        conflict_rules = set()
        
        for i in range(len(X)):
            x = X[i]
            true_label = int(y[i])
            
            for rule in self.rules:
                if self._rule_matches(rule, x):
                    if rule.class_label != true_label:
                        conflict_rules.add(rule.id)
                    break
        
        return conflict_rules
    
    def _rule_matches(self, rule, x):
        """Check if a rule matches a sample."""
        for feat_idx, thresh, op in rule.conditions:
            val = x[feat_idx]
            if op == 0:  # <
                if not (val < thresh):
                    return False
            elif op == 1:  # >=
                if not (val >= thresh):
                    return False
            elif op == 2:  # ==
                if not (val == thresh):
                    return False
            elif op == 3:  # !=
                if not (val != thresh):
                    return False
        return True
    
    def predict(self, X):
        """
        Predict class labels using LINC rules with voting.
        
        In improved mode with sklearn tree available, uses sklearn for fast batch prediction.
        Otherwise uses rule-based prediction with weighted voting.
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Use sklearn tree for fast prediction if available (improved mode)
        if self._sklearn_tree is not None and self.improved:
            predictions = self._sklearn_tree.predict(X)
            # Get prediction probabilities for confidence
            proba = self._sklearn_tree.predict_proba(X)
            confidences = np.max(proba, axis=1)
            return predictions.astype(np.int64), confidences.astype(np.float32)
        
        # Rule-based prediction with weighted voting
        predictions = np.zeros(n_samples, dtype=np.int64)
        confidences = np.zeros(n_samples, dtype=np.float32)
        
        # Build default class from rule distribution
        if self.rules:
            class_counts = defaultdict(int)
            for r in self.rules:
                class_counts[r.class_label] += 1
            default_class = max(class_counts, key=class_counts.get)
        else:
            default_class = 0
        
        for i in range(n_samples):
            x = X[i]
            
            # Find all matching rules
            matching_rules = []
            for rule in self.rules:
                if self._rule_matches(rule, x):
                    matching_rules.append(rule)
            
            if matching_rules:
                # Weighted voting by confidence
                class_votes = defaultdict(float)
                for r in matching_rules:
                    class_votes[r.class_label] += r.confidence
                
                # Select class with highest vote
                best_class = max(class_votes, key=class_votes.get)
                predictions[i] = best_class
                confidences[i] = class_votes[best_class] / len(matching_rules)
            else:
                # No matching rule - use default
                predictions[i] = default_class
                confidences[i] = 0.0
        
        return predictions, confidences
    
    def _reallocate_samples(self, X, y):
        """Allocate samples to matching rules."""
        self.allocated_samples = {}
        
        for i in range(len(X)):
            x = X[i]
            label = int(y[i])
            
            # Find best matching rule for this sample
            best_rule = None
            best_conf = -1
            
            for rule in self.rules:
                if self._rule_matches(rule, x):
                    # Prefer rules that predict the correct class
                    if rule.class_label == label and rule.confidence > best_conf:
                        best_conf = rule.confidence
                        best_rule = rule
            
            # If no correct rule found, find any matching rule
            if best_rule is None:
                for rule in self.rules:
                    if self._rule_matches(rule, x):
                        if rule.confidence > best_conf:
                            best_conf = rule.confidence
                            best_rule = rule
            
            if best_rule:
                if best_rule.id not in self.allocated_samples:
                    self.allocated_samples[best_rule.id] = []
                self.allocated_samples[best_rule.id].append((x, label))
    
    def _prune_rules(self, y):
        """
        Prune rules while maintaining class balance.
        
        OPTIMIZED: Better class balancing to prevent minority class rules from being pruned.
        """
        if len(self.rules) <= self.max_rules:
            return
        
        logging.info(f"LINC: Pruning from {len(self.rules)} to {self.max_rules} rules")
        
        # Calculate rule effectiveness
        for rule in self.rules:
            samples = self.allocated_samples.get(rule.id, [])
            rule._sample_count = len(samples)
            if samples:
                correct = sum(1 for _, label in samples if label == rule.class_label)
                rule._accuracy = correct / len(samples)
            else:
                rule._accuracy = 0.0
        
        # Group rules by class
        class_rules = defaultdict(list)
        for rule in self.rules:
            class_rules[rule.class_label].append(rule)
        
        # Sort rules within each class by effectiveness
        for c in class_rules:
            class_rules[c].sort(
                key=lambda r: (r._accuracy * (r._sample_count + 1), r.confidence),
                reverse=True
            )
        
        # OPTIMIZATION: Ensure minimum rules per class for better minority class coverage
        unique_classes = list(class_rules.keys())
        n_classes = len(unique_classes)
        
        # Calculate class frequencies
        class_counts = {c: sum(1 for label in y if label == c) for c in unique_classes}
        total_samples = sum(class_counts.values())
        
        pruned_rules = []
        
        # OPTIMIZATION: Guarantee more rules for minority classes
        # Minimum rules per class: at least 5% of max_rules or 10, whichever is larger
        min_per_class = max(10, self.max_rules // (n_classes * 2), self.max_rules // 20)
        
        # First pass: guarantee minimum per class (prioritize minority classes)
        # Sort classes by frequency (ascending) to prioritize minority classes
        sorted_classes = sorted(unique_classes, key=lambda c: class_counts.get(c, 0))
        
        for c in sorted_classes:
            n_alloc = min(min_per_class, len(class_rules[c]))
            pruned_rules.extend(class_rules[c][:n_alloc])
        
        # Second pass: fill remaining slots proportionally
        remaining = self.max_rules - len(pruned_rules)
        if remaining > 0 and total_samples > 0:
            # Calculate remaining rules per class
            for c in unique_classes:
                already_alloc = min(min_per_class, len(class_rules[c]))
                remaining_rules = class_rules[c][already_alloc:]
                
                if not remaining_rules:
                    continue
                
                # Use square root of proportion to give more weight to minority classes
                proportion = np.sqrt(class_counts.get(c, 0) / total_samples)
                n_extra = int(remaining * proportion / n_classes)
                n_extra = min(n_extra, len(remaining_rules))
                pruned_rules.extend(remaining_rules[:n_extra])
        
        # Trim if over limit (keep highest quality rules)
        if len(pruned_rules) > self.max_rules:
            # Sort by effectiveness but ensure class diversity
            pruned_rules.sort(
                key=lambda r: (r._accuracy * (r._sample_count + 1), r.confidence),
                reverse=True
            )
            
            # Keep top rules while ensuring minimum per class
            final_rules = []
            class_kept = defaultdict(int)
            min_keep = max(5, self.max_rules // (n_classes * 3))
            
            for rule in pruned_rules:
                if len(final_rules) >= self.max_rules:
                    break
                # Always keep if class hasn't reached minimum
                if class_kept[rule.class_label] < min_keep:
                    final_rules.append(rule)
                    class_kept[rule.class_label] += 1
                elif len(final_rules) < self.max_rules:
                    final_rules.append(rule)
                    class_kept[rule.class_label] += 1
            
            pruned_rules = final_rules
        
        # Update allocated samples
        kept_ids = {r.id for r in pruned_rules}
        self.allocated_samples = {k: v for k, v in self.allocated_samples.items() 
                                   if k in kept_ids}
        
        self.rules = pruned_rules
        
        # Log class distribution after pruning
        class_dist = defaultdict(int)
        for r in self.rules:
            class_dist[r.class_label] += 1
        logging.info(f"LINC: Post-pruning class distribution: {dict(class_dist)}")

    def _add_default_rule(self, y):
        """Add a catch-all default rule for 100% coverage."""
        # Calculate majority class
        unique, counts = np.unique(y, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        
        # Check if default rule already exists
        for rule in self.rules:
            if len(rule.conditions) == 0:
                # Update existing default rule
                rule.class_label = int(majority_class)
                return

        # Create new default rule (empty conditions matches everything)
        default_rule = LincRule(
            conditions=[],
            class_label=int(majority_class),
            confidence=0.5
        )
        # Append to end so it has lowest priority
        self.rules.append(default_rule)
        logging.info(f"LINC: Added default rule for class {majority_class}")

    def _log_class_distribution(self):
        """Log the class distribution of rules."""
        class_dist = defaultdict(int)
        for r in self.rules:
            class_dist[r.class_label] += 1
        logging.info(f"LINC: Rule class distribution: {dict(class_dist)}")
