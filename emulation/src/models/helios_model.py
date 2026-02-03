import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import logging
import copy
import random

# Available distance calculation methods: ['l_2', 'l_n']
# Available calculation modes: ['trans_abs', 'abs_trans']


def set_global_seed(seed):
    """Set the seed for random number generators to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CrossEntropyLabelSmooth(nn.Module):
    """Cross Entropy loss with label smoothing for improved generalization."""
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).mean(0).sum()
        return loss

class Prototype(nn.Module):
    def __init__(self, num_classes, feature_num, prototype_num_classes, temperature, cal_dis, cal_mode):
        super(Prototype, self).__init__()
        self.num_classes = num_classes
        self.feature_num = feature_num
        self.temperature = temperature
        self.cal_dis = cal_dis
        self.cal_mode = cal_mode

        # Initialize prototype and transformation parameters for each class
        self.class_prototype = nn.ParameterList()
        self.class_transform = nn.ParameterList()
        for i in range(self.num_classes):
            # Using ParameterList to properly register parameters
            self.class_prototype.append(nn.Parameter(torch.randn(1, prototype_num_classes[i], self.feature_num)))
            self.class_transform.append(nn.Parameter(torch.randn(1, prototype_num_classes[i], self.feature_num)))

        # Uniformly initialize prototypes between 0 and 1
        # Initialize transforms to 1 (neutral scaling) for stability
        for i in range(self.num_classes):
            nn.init.uniform_(self.class_prototype[i], a=0, b=1)
            nn.init.ones_(self.class_transform[i])

    def forward(self, train_batch):
        # Reshape input tensor to [batch_size, 1, feature_dim]
        # train_batch shape: [batch_size, feature_num]
        train_batch = train_batch.view(train_batch.shape[0], 1, self.feature_num)
        min_dist_class = []

        if self.cal_dis == 'l_2':
            for idx in range(self.num_classes):
                if self.class_prototype[idx].shape[1] == 0:
                    inf_dist = torch.zeros(train_batch.shape[0], device=train_batch.device)
                    min_dist_class.append(inf_dist + 10000)
                    continue
                
                dist_class = (train_batch - self.class_prototype[idx]) ** 2
                dist_class = torch.sum(dist_class, dim=2)
                min_dist, _ = torch.min(dist_class, dim=1)
                min_dist_class.append(min_dist)

        elif self.cal_dis == 'l_n':
            for idx in range(self.num_classes):
                if self.class_prototype[idx].shape[1] == 0:
                    inf_dist = torch.zeros(train_batch.shape[0], device=train_batch.device)
                    min_dist_class.append(inf_dist + 10000)
                    continue

                if self.cal_mode == 'trans_abs':
                    batch_transformed = train_batch * torch.abs(self.class_transform[idx])
                    dist_class = torch.abs(batch_transformed - self.class_prototype[idx])
                elif self.cal_mode == 'abs_trans':
                    dist_class = torch.abs(train_batch - self.class_prototype[idx])
                    dist_class = dist_class * torch.abs(self.class_transform[idx])
                
                # Get max distance across features then min across prototypes
                max_dist, _ = torch.max(dist_class, dim=2)
                min_dist, _ = torch.min(max_dist, dim=1)
                min_dist_class.append(min_dist)

        # Calculate similarity scores
        class_similar_score = []
        for idx in range(self.num_classes):
            score = 1 / (min_dist_class[idx] + 1e-6) * self.temperature
            class_similar_score.append(score.view(-1, 1))

        # Concatenate scores and apply softmax
        min_dist = torch.cat(class_similar_score, dim=1)
        logit = F.softmax(min_dist, dim=1)
        return logit

    def save_parameter(self, path):
        # Save in upstream-compatible format (concatenated list)
        torch.save(list(self.class_prototype) + list(self.class_transform), path)

    def load_parameter(self, path):
        state = torch.load(path, map_location='cpu')
        # Support both formats: [prototypes, transforms] or concatenated list
        if isinstance(state, (list, tuple)) and len(state) == 2 and isinstance(state[0], (list, nn.ParameterList)):
            prototypes = state[0]
            transforms = state[1]
        else:
            half = len(state) // 2
            prototypes = state[:half]
            transforms = state[half:]

        for i in range(min(self.num_classes, len(prototypes))):
            if self.class_prototype[i].shape[1] != prototypes[i].shape[1]:
                self.class_prototype[i] = nn.Parameter(
                    torch.FloatTensor(1, prototypes[i].shape[1], self.feature_num)
                )
            self.class_prototype[i].data = prototypes[i].data

        for i in range(min(self.num_classes, len(transforms))):
            if self.class_transform[i].shape[1] != transforms[i].shape[1]:
                self.class_transform[i] = nn.Parameter(
                    torch.FloatTensor(1, transforms[i].shape[1], self.feature_num)
                )
            self.class_transform[i].data = transforms[i].data


def init_model(model, init_type, dbscan_eps, min_samples, X_train, y_train, feature_min, feature_max):
    if init_type == 'NONE':
        return model
    
    device = model.class_prototype[0].device
    
    if init_type == 'DBSCAN':
        unique_labels = np.unique(y_train)
        centers = {}

        for label in unique_labels:
            class_mask = (y_train == label)
            X_class = X_train[class_mask]

            denom = feature_max - feature_min
            denom[denom == 0] = 1.0
            X_class_scaled = (X_class - feature_min) / denom

            dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X_class_scaled)

            centers[label] = []
            unique_clusters = set(cluster_labels)
            for cluster in unique_clusters:
                if cluster == -1:
                    continue
                cluster_mask = (cluster_labels == cluster)
                cluster_points = X_class_scaled[cluster_mask]
                if cluster_points.shape[0] == 0:
                    continue
                cluster_center = np.mean(cluster_points, axis=0)
                centers[label].append(cluster_center)

        for i in range(model.num_classes):
            if i not in centers or len(centers[i]) == 0:
                model.class_prototype[i] = nn.Parameter(torch.randn(1, 0, model.feature_num, device=device))
                model.class_transform[i] = nn.Parameter(torch.ones(1, 0, model.feature_num, device=device))
                continue

            num_clusters = len(centers[i])
            model.class_prototype[i] = nn.Parameter(torch.zeros(1, num_clusters, model.feature_num, device=device))
            model.class_transform[i] = nn.Parameter(torch.ones(1, num_clusters, model.feature_num, device=device))

            for idx, center in enumerate(centers[i]):
                with torch.no_grad():
                    model.class_prototype[i][0][idx].copy_(torch.from_numpy(center).to(device))
    
    elif init_type == 'KMEANS' or init_type == 'KMeaans': 
        unique_labels = np.unique(y_train)
        centers = {}
        
        for label in unique_labels:
            class_mask = (y_train == label)
            X_class = X_train[class_mask]
            
            denom = feature_max - feature_min
            denom[denom == 0] = 1.0
            X_class_scaled = (X_class - feature_min) / denom
            
            # Count unique samples to avoid ConvergenceWarning
            unique_samples = np.unique(X_class_scaled, axis=0)
            n_unique = len(unique_samples)
            n_clusters = min(100, X_class_scaled.shape[0], n_unique)
            if n_clusters < 1: n_clusters = 1
            
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
            cluster_labels = kmeans.fit_predict(X_class_scaled)
            
            centers[label] = []
            unique_clusters = set(cluster_labels)
            for cluster in unique_clusters:
                cluster_mask = (cluster_labels == cluster)
                cluster_points = X_class_scaled[cluster_mask]
                cluster_center = np.mean(cluster_points, axis=0)
                centers[label].append(cluster_center)
        
        
        for i in range(model.num_classes):
            if i not in centers:
                model.class_prototype[i] = nn.Parameter(torch.randn(1, 0, model.feature_num, device=device))
                model.class_transform[i] = nn.Parameter(torch.ones(1, 0, model.feature_num, device=device))
                continue
            
            num_clusters = len(centers[i])
            model.class_prototype[i] = nn.Parameter(torch.zeros(1, num_clusters, model.feature_num, device=device))
            model.class_transform[i] = nn.Parameter(torch.zeros(1, num_clusters, model.feature_num, device=device) + 1.0)
            
            for idx, center in enumerate(centers[i]):
                with torch.no_grad():
                    model.class_prototype[i][0][idx].copy_(torch.from_numpy(center).to(device))
                    
    return model

def model_prune(model, num_classes, X_train, y_train, feature_min, feature_max, max_proto_num, prune_T):
    batch_size = 512
    device = model.class_prototype[0].device
    
    current_max_proto = 0
    for i in range(num_classes):
        if i < len(model.class_prototype):
            current_max_proto = max(current_max_proto, model.class_prototype[i].shape[1])
    
    support_num = np.zeros((num_classes, max(max_proto_num, current_max_proto)), dtype=np.int64)
    
    idx = 0
    while idx < X_train.shape[0]:
        end = min(idx + batch_size, X_train.shape[0])
        batch_x = X_train[idx:end]
        batch_y = y_train[idx:end]
        
        denom = feature_max - feature_min
        denom[denom == 0] = 1.0
        batch_x = (batch_x - feature_min) / denom
        
        batch_x_t = torch.from_numpy(batch_x).float().to(device)
        predictions = torch.zeros(batch_x_t.shape[0], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model(batch_x_t)
            _, preds = torch.max(logits, dim=1)
            predictions = preds
            
            batch_x_view = batch_x_t.view(batch_x_t.shape[0], 1, model.feature_num)
            
            for class_id in range(num_classes):
                if model.class_prototype[class_id].shape[1] == 0:
                    continue
                
                if model.cal_dis == 'l_n':
                    if model.cal_mode == 'abs_trans':
                        dist = torch.abs(batch_x_view - model.class_prototype[class_id])
                        dist = dist * torch.abs(model.class_transform[class_id])
                    else: 
                        dist = torch.abs(batch_x_view * torch.abs(model.class_transform[class_id]) - model.class_prototype[class_id])
                    
                    max_d, _ = torch.max(dist, dim=2)
                    _, cls_indices = torch.min(max_d, dim=1)
                else: 
                    dist = (batch_x_view - model.class_prototype[class_id]) ** 2
                    dist = torch.sum(dist, dim=2)
                    _, cls_indices = torch.min(dist, dim=1)
                
                mask = (predictions == class_id) & (torch.from_numpy(batch_y).to(device) == class_id)
                if mask.any():
                    valid_indices = cls_indices[mask].cpu().numpy()
                    for pid in valid_indices:
                        if pid < support_num.shape[1]:
                            support_num[class_id][pid] += 1
                            
        idx += batch_size

    for class_id in range(num_classes):
        supports = support_num[class_id]
        current_size = model.class_prototype[class_id].shape[1]
        valid_supports = supports[:current_size]
        
        to_keep_indices = [i for i, s in enumerate(valid_supports) if s > prune_T]
        
        if len(to_keep_indices) < current_size:
            if len(to_keep_indices) > 0:
                new_proto = model.class_prototype[class_id][:, to_keep_indices, :]
                new_trans = model.class_transform[class_id][:, to_keep_indices, :]
                model.class_prototype[class_id] = nn.Parameter(new_proto)
                model.class_transform[class_id] = nn.Parameter(new_trans)
            else:
                model.class_prototype[class_id] = nn.Parameter(
                    torch.randn(1, 0, model.feature_num, device=device)
                )
                model.class_transform[class_id] = nn.Parameter(
                    torch.ones(1, 0, model.feature_num, device=device)
                )
            
    return model

class HeliosRule:
    """
    Helios matching rule (Hypercube/Bounding Box).
    """
    def __init__(self, class_label, intervals, confidence=1.0, prototype_idx=-1):
        self.class_label = class_label
        self.intervals = intervals  # List of [min, max] for each feature
        self.confidence = confidence
        self.prototype_idx = prototype_idx
        self.support = 0
        self.id = id(self)

    def match(self, x):
        for i, (min_val, max_val) in enumerate(self.intervals):
            if x[i] < min_val or x[i] > max_val:
                return False
        return True
    
    @property
    def conditions(self):
        """
        Convert intervals to LINC-compatible conditions list.
        Returns: list of (feature_idx, threshold, operator)
        Operator 0: <=, Operator 1: >
        
        Note: We use a larger epsilon (1e-6) to handle floating point precision issues,
        especially for point intervals where min == max.
        """
        conds = []
        epsilon = 1e-6  # Larger epsilon for floating point robustness
        for i, (min_val, max_val) in enumerate(self.intervals):
            # For the lower bound, we want feature > (min - epsilon)
            # This is equivalent to feature >= min with some tolerance
            conds.append((i, min_val - epsilon, 1)) 
            # For the upper bound, we want feature <= (max + epsilon)
            # This provides tolerance for floating point comparison
            conds.append((i, max_val + epsilon, 0))
        return conds

    def match_batch(self, X):
        """Check if samples match this rule (bounding box check)."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        num_rule_features = len(self.intervals)
        num_data_features = X.shape[1]
        
        # Handle dimension mismatch
        if num_rule_features != num_data_features:
            # Use minimum of both dimensions
            check_dim = min(num_rule_features, num_data_features)
            X_subset = X[:, :check_dim]
            mins = np.array([self.intervals[i][0] for i in range(check_dim)])
            maxs = np.array([self.intervals[i][1] for i in range(check_dim)])
        else:
            X_subset = X
            mins = np.array([interval[0] for interval in self.intervals])
            maxs = np.array([interval[1] for interval in self.intervals])
        
        lower_check = np.all(X_subset >= mins, axis=1)
        upper_check = np.all(X_subset <= maxs, axis=1)
        return lower_check & upper_check
    
    def to_dict(self):
        return {
            'class_label': int(self.class_label),
            'intervals': [[float(v) for v in i] for i in self.intervals],
            'confidence': float(self.confidence),
            'prototype_idx': int(self.prototype_idx),
            'support': int(self.support)
        }

class HeliosSystem:
    def __init__(self, input_dim, num_classes, embedding_dim=64, hidden_dim=128, num_prototypes_per_class=10, 
                 max_rules=1000, radio=1.0, boost_num=6, prune_threshold=3, device='cuda',
                 label_smoothing=0.1, seed=2024, strict_conflict=True):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_proto_num = num_prototypes_per_class
        self.max_rules = max_rules
        self.radio = radio
        self.boost_num = boost_num
        self.prune_T = prune_threshold
        self.device = device
        self.rules = []
        
        self.temperature = 0.1
        self.cal_dis = 'l_n'
        self.cal_mode = 'abs_trans'
        self.init_type = 'DBSCAN'
        self.dbscan_eps = 0.1
        self.min_samples = 2
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.strict_conflict = strict_conflict
        set_global_seed(self.seed)
        
        # Persistent prototype model - preserved across incremental updates
        self.model = None
        # Track accumulated prototypes across boosting iterations
        self._accumulated_prototypes = {c: [] for c in range(num_classes)}
        self._accumulated_transforms = {c: [] for c in range(num_classes)}
        self._conflict_fix = {}

    def fit(self, X, y, epochs=20, batch_size=256, lr=0.001):
        feature_min = np.min(X, axis=0).astype(float)
        feature_max = np.max(X, axis=0).astype(float)
        feature_max[feature_max == feature_min] += 1e-6
        
        self.feature_min = feature_min
        self.feature_max = feature_max
        
        self.rules = []
        
        logging.info(f"HeliosSystem.fit: Training on {len(X)} samples, {len(np.unique(y))} classes")
        
        new_rules = self._run_boosting(X, y, epochs, batch_size, lr)
        self.rules.extend(new_rules)

        # Prune rules by support on training data (upstream Prune_rules)
        self._prune_rules_by_support(X, y, min_support=self.prune_T)
        
        logging.info(f"HeliosSystem.fit: Generated {len(self.rules)} rules")

    def incremental_update(self, X_new, y_new):
        """
        Incremental update following original Helios Incremental_rules logic:
        1. Remove rules that CONFLICT with new samples (match wrong class)
        2. Find RESIDUAL samples (not correctly matched by any rule)
        3. Generate new rules for residual samples via boosting
        
        Key insight from original Helios: Rules are removed if they match ANY sample
        from a DIFFERENT class. This is more aggressive but ensures rule purity.
        """
        logging.info(f"HeliosSystem: Incremental update with {len(X_new)} samples, current rules: {len(self.rules)}")
        
        # Update feature bounds to include new data
        if hasattr(self, 'feature_min') and hasattr(self, 'feature_max'):
            self.feature_min = np.minimum(self.feature_min, np.min(X_new, axis=0).astype(float))
            self.feature_max = np.maximum(self.feature_max, np.max(X_new, axis=0).astype(float))
            self.feature_max[self.feature_max == self.feature_min] += 1e-6
        else:
            self.feature_min = np.min(X_new, axis=0).astype(float)
            self.feature_max = np.max(X_new, axis=0).astype(float)
            self.feature_max[self.feature_max == self.feature_min] += 1e-6
        
        # If no rules exist, perform initial training
        if len(self.rules) == 0:
            logging.info("HeliosSystem: No existing rules, performing initial training...")
            self.fit(X_new, y_new, epochs=30)
            return
        
        # --- Optimized Step 1 & 2: Vectorized Matching ---
        
        # 1. Compile rules to matrix
        n_samples = len(X_new)
        n_rules = len(self.rules)
        n_features = X_new.shape[1]
        
        mins = np.full((n_rules, n_features), -np.inf)
        maxs = np.full((n_rules, n_features), np.inf)
        rule_class_labels = np.array([r.class_label for r in self.rules])
        
        for i, r in enumerate(self.rules):
            r_int = r.intervals
            dim = min(len(r_int), n_features)
            if dim > 0:
                # Optimized extraction
                intervals_arr = np.array(r_int[:dim])
                mins[i, :dim] = intervals_arr[:, 0]
                maxs[i, :dim] = intervals_arr[:, 1]
        
        # 2. Compute Match Matrix: (Samples, Rules)
        # Use chunking to avoid OOM for large N*M
        chunk_size = 500
        all_matches = np.zeros((n_samples, n_rules), dtype=bool)
        
        for i in range(0, n_rules, chunk_size):
            end = min(i + chunk_size, n_rules)
            mins_chunk = mins[i:end]  # (Chunk, Feat)
            maxs_chunk = maxs[i:end]
            
            # Broadcast against all samples: (Samples, 1, Feat) vs (1, Chunk, Feat)
            # This splits Rules into chunks, but broadcast input X against it.
            # If Samples is very large, we should loop samples too?
            # Assuming Samples ~4000 (update_samples), Rules ~10000. 
            # 4000 * 500 * 80 * 2 (bool) is small.
            
            # X_new: (Samples, Feat) -> (Samples, 1, Feat)
            X_exp = X_new[:, np.newaxis, :]
            
            # Check conditions
            matches_chunk = (X_exp >= mins_chunk) & (X_exp <= maxs_chunk)
            # Fall check across features
            all_matches[:, i:end] = np.all(matches_chunk, axis=2)
            
        # 3. Step 1: Filter rules based on conflicts/accuracy
        # Strict conflict removal (upstream Incremental_rules):
        # remove any rule that matches a sample of a different class.
        rules_to_keep = []
        conflicted_count = 0

        # Vectorized counts
        correct_counts = np.sum(all_matches & (y_new[:, np.newaxis] == rule_class_labels), axis=0)
        wrong_counts = np.sum(all_matches & (y_new[:, np.newaxis] != rule_class_labels), axis=0)

        for i, rule in enumerate(self.rules):
            correct = correct_counts[i]
            wrong = wrong_counts[i]
            total = correct + wrong

            if self.strict_conflict:
                if wrong == 0:
                    rules_to_keep.append(rule)
                else:
                    conflicted_count += 1
            else:
                accuracy = correct / total if total > 0 else 1.0
                if total == 0 or accuracy >= 0.7:
                    rules_to_keep.append(rule)
                else:
                    conflicted_count += 1

        if conflicted_count > 0:
            if self.strict_conflict:
                logging.info(f"HeliosSystem: Removed {conflicted_count} conflicting rules (strict removal)")
            else:
                logging.info(f"HeliosSystem: Removed {conflicted_count} low-accuracy rules (accuracy < 70%)")
        self.rules = rules_to_keep
        
        # Need to rebuild rule list means 'all_matches' indices are now invalid for removed rules.
        # But we need 'all_matches' for residual detection?
        # Step 2: Find RESIDUAL samples
        # Residual = not correctly matched by ANY *surviving* rule.
        # "First match wins" logic applies.
        
        # We need to re-evaluate or filter 'all_matches' for kept rules.
        # kept_indices maps new rules to old indices
        kept_mask = np.zeros(n_rules, dtype=bool)
        kept_indices = []
        current_idx = 0
        
        # We can iterate rules and check if kept.
        # Optimized:
        rule_kept_map = np.zeros(n_rules, dtype=bool)
        total_counts = correct_counts + wrong_counts
        if self.strict_conflict:
            rule_kept_map = (wrong_counts == 0)
        else:
            accs = np.divide(correct_counts, total_counts, out=np.ones_like(correct_counts, dtype=float), where=total_counts!=0)
            rule_kept_map = (total_counts == 0) | (accs >= 0.7)
        
        # Surviving matches matrix
        surviving_matches = all_matches[:, rule_kept_map]
        surviving_labels = rule_class_labels[rule_kept_map]

        # Build conflict-fix mapping from overlapping rules (upstream solve_conflict)
        conflict_fix = self._build_conflict_fix(surviving_matches, y_new)
        self._conflict_fix = conflict_fix

        # Residual detection with overlap handling
        residual_mask = self._residual_mask_with_conflict(surviving_matches, surviving_labels, y_new, conflict_fix)
            
        X_resid = X_new[residual_mask]
        y_resid = y_new[residual_mask]
        
        residual_ratio = len(X_resid) / len(X_new) if len(X_new) > 0 else 0
        logging.info(f"HeliosSystem: {len(X_resid)}/{len(X_new)} residual samples ({residual_ratio:.1%})")
        
        # Step 3: Generate new rules for residual samples
        if len(X_resid) > 5:
            # OPTIMIZATION: Use _run_boosting (train from scratch) instead of _incremental_boosting
            # Reverting to training new models per boosting iteration for better accuracy on residuals.
            # While _incremental_boosting (fine-tuning) is theoretically faster, it seems to degrade accuracy
            # likely due to the global model not adapting well to the specific hard residual samples.
            new_rules = self._run_boosting(X_resid, y_resid, epochs=20, batch_size=256, lr=0.001)
            self.rules = self._merge_rules(new_rules, self.rules)
            logging.info(f"HeliosSystem: Added {len(new_rules)} new rules, total: {len(self.rules)}")
        else:
            logging.info("HeliosSystem: Few residual samples, skipping rule generation")

        # Prune rules by support on latest data
        self._prune_rules_by_support(X_new, y_new, min_support=self.prune_T)
        
        # Balance if exceeding max
        if len(self.rules) > self.max_rules:
            self._balance_prune_rules()
    
    def _merge_rules(self, new_rules, old_rules):
        """
        Merge rule lists following original Helios merge_rules logic.
        New rules take priority, duplicates are removed.
        """
        result = []
        seen_rules = set()
        
        # Add new rules first (they have priority)
        for rule in new_rules:
            # Create hashable representation of intervals
            rule_tuple = tuple(tuple(interval) for interval in rule.intervals)
            rule_key = (rule.class_label, rule_tuple)
            if rule_key not in seen_rules:
                seen_rules.add(rule_key)
                result.append(rule)
        
        # Add old rules if not duplicates
        for rule in old_rules:
            rule_tuple = tuple(tuple(interval) for interval in rule.intervals)
            rule_key = (rule.class_label, rule_tuple)
            if rule_key not in seen_rules:
                seen_rules.add(rule_key)
                result.append(rule)
        
        return result

    def _build_conflict_fix(self, match_matrix, y_true):
        """Build mapping from overlapping rule sets to majority class label.

        This mirrors upstream solve_conflict: for samples matched by multiple rules,
        record the label and later use majority vote to resolve conflicts.
        """
        conflict_fix = {}
        for i in range(match_matrix.shape[0]):
            matched = np.where(match_matrix[i])[0]
            if matched.size <= 1:
                continue
            key = tuple(matched.tolist())
            if key not in conflict_fix:
                conflict_fix[key] = []
            conflict_fix[key].append(int(y_true[i]))

        # Reduce to majority label
        for key, labels in conflict_fix.items():
            counts = {}
            for lbl in labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            majority = max(counts, key=counts.get)
            conflict_fix[key] = majority

        return conflict_fix

    def _residual_mask_with_conflict(self, match_matrix, rule_labels, y_true, conflict_fix):
        """Residual mask with overlap resolution.

        A sample is NOT residual if:
        - it matches exactly one rule and label is correct; or
        - it matches multiple rules and conflict_fix maps the rule-set to the correct label.
        Otherwise it remains residual.
        """
        n_samples = match_matrix.shape[0]
        residual_mask = np.ones(n_samples, dtype=bool)

        for i in range(n_samples):
            matched = np.where(match_matrix[i])[0]
            if matched.size == 0:
                continue
            if matched.size == 1:
                pred = rule_labels[matched[0]]
                if pred == y_true[i]:
                    residual_mask[i] = False
            else:
                key = tuple(matched.tolist())
                if key in conflict_fix and conflict_fix[key] == y_true[i]:
                    residual_mask[i] = False

        return residual_mask
    
    def _prune_rules_by_support(self, X, y, min_support=None):
        """Remove rules with low support on given data.
        
        Support is counted as total matches (not just correct matches),
        following original Helios Prune_rules logic.
        """
        if min_support is None:
            min_support = self.prune_T
        if min_support <= 0:
            return
        
        rules_to_keep = []
        for rule in self.rules:
            matches = rule.match_batch(X)
            # Count total matches (original Helios counts sample_num, not correct_num)
            rule.support = np.sum(matches)
            
            if rule.support >= min_support:
                rules_to_keep.append(rule)
        
        removed = len(self.rules) - len(rules_to_keep)
        if removed > 0:
            logging.info(f"HeliosSystem: Pruned {removed} low-support rules (support < {min_support}).")
        self.rules = rules_to_keep
    
    def _balance_prune_rules(self):
        """Prune rules while maintaining class balance."""
        class_rules = {}
        for r in self.rules:
            c = r.class_label
            if c not in class_rules:
                class_rules[c] = []
            class_rules[c].append(r)
        
        # Sort by support (descending)
        for c in class_rules:
            class_rules[c].sort(key=lambda x: x.support, reverse=True)
        
        num_classes = len(class_rules)
        if num_classes == 0:
            return
        
        rules_per_class = self.max_rules // num_classes
        
        pruned_rules = []
        for c in class_rules:
            pruned_rules.extend(class_rules[c][:rules_per_class])
        
        # Fill remaining slots
        remaining = self.max_rules - len(pruned_rules)
        if remaining > 0:
            all_remaining = []
            for c in class_rules:
                all_remaining.extend(class_rules[c][rules_per_class:])
            all_remaining.sort(key=lambda x: x.support, reverse=True)
            pruned_rules.extend(all_remaining[:remaining])
        
        self.rules = pruned_rules
        logging.info(f"HeliosSystem: Balanced to {len(self.rules)} rules.")

    def _update_coverage_vectorized(self, X, y, rules):
        """
        Vectorized calculation of uncovered mask and accuracy.
        A sample is 'covered' (removed from residuals) ONLY if:
        1. It is matched by at least one rule.
        2. The FIRST matching rule (priority order) predicts the CORRECT label.
        
        Returns:
            uncovered_mask: Boolean array, True if sample should remain in residuals.
            accuracy: Fraction of samples correctly predicted.
            coverage: Fraction of samples matched by any rule (regardless of correctness).
        """
        if not rules or len(X) == 0:
            return np.ones(len(X), dtype=bool), 0.0, 0.0
            
        n_samples = len(X)
        n_rules = len(rules)
        n_features = X.shape[1]
        
        # Precompute rule bounds (Rules, Features)
        mins = np.full((n_rules, n_features), -np.inf)
        maxs = np.full((n_rules, n_features), np.inf)
        rule_labels = np.array([r.class_label for r in rules])
        
        for i, r in enumerate(rules):
            r_int = r.intervals
            dim = min(len(r_int), n_features)
            # Use NumPy array creation only if dim > 0 to avoid shape errors
            if dim > 0:
                intervals_arr = np.array(r_int[:dim])
                # Check shape to ensure correct assignment
                if intervals_arr.ndim == 2 and intervals_arr.shape[1] == 2:
                    mins[i, :dim] = intervals_arr[:, 0]
                    maxs[i, :dim] = intervals_arr[:, 1]
        
        # Chunked processing to find FIRST match
        # We need first match index for each sample.
        first_match_indices = np.full(n_samples, -1, dtype=int)
        
        # Optim: if samples are too many, chunk samples too? 
        # But X_exp (Samples, 1, Feat) size dominates.
        
        chunk_size = 500
        # Iterate rules in chunks
        for i in range(0, n_rules, chunk_size):
            end = min(i + chunk_size, n_rules)
            
            # Identify samples not yet matched
            # Optimization: Mask logic
            unmatched_indices = np.where(first_match_indices == -1)[0]
            if len(unmatched_indices) == 0:
                break
                
            # Use only unmatched samples for comparison to save memory/compute
            X_sub = X[unmatched_indices, np.newaxis, :]
            
            mins_chunk = mins[i:end]
            maxs_chunk = maxs[i:end]
            
            # (SubSamples, Chunk_Rules, Feat)
            matches_feat = (X_sub >= mins_chunk) & (X_sub <= maxs_chunk)
            # (SubSamples, Chunk_Rules)
            matches_rule = np.all(matches_feat, axis=2)
            
            # Samples that have a match in this chunk
            has_match_in_chunk = np.any(matches_rule, axis=1)
            
            if np.any(has_match_in_chunk):
                # Relative index in sub-batch
                matched_sub_indices = np.where(has_match_in_chunk)[0]
                
                # Relative rule index
                chunk_rel_idx = np.argmax(matches_rule[matched_sub_indices], axis=1)
                
                # Map back to original indices
                original_indices = unmatched_indices[matched_sub_indices]
                
                first_match_indices[original_indices] = chunk_rel_idx + i
                
        # Determine Status
        # Uncovered if: No match OR (Match but Wrong Label)
        
        matched_mask = (first_match_indices != -1)
        correct_mask = np.zeros(n_samples, dtype=bool)
        
        if np.any(matched_mask):
            pred_labels = rule_labels[first_match_indices[matched_mask]]
            true_labels = y[matched_mask]
            correct_preds = (pred_labels == true_labels)
            
            # Set correct_mask
            matched_indices = np.where(matched_mask)[0]
            correct_mask[matched_indices] = correct_preds

        # Uncovered = Not (Matched AND Correct) -> Keep boosting for these
        uncovered_mask = ~correct_mask
        
        accuracy = np.mean(correct_mask)
        coverage = np.mean(matched_mask)
        
        return uncovered_mask, accuracy, coverage

    def _incremental_boosting(self, X, y, epochs=15, batch_size=256, lr=0.001):
        """
        Incremental boosting that preserves and extends existing prototypes.
        Unlike _run_boosting, this method:
        1. Reuses existing prototype model if available
        2. Adds new prototypes for uncovered classes/regions
        3. Fine-tunes existing prototypes with new data
        """
        generated_rules = []
        
        # Use stored feature bounds (already updated in incremental_update)
        feature_min = self.feature_min
        feature_max = self.feature_max
        
        current_X = X
        current_y = y
        
        logging.info(f"HeliosSystem: Incremental boosting with {len(X)} samples")
        
        for boost_iter in range(self.boost_num):
            if len(current_X) == 0:
                logging.info(f"HeliosSystem: Incremental boost iter {boost_iter} - no samples left")
                break
            
            # Check which classes need new prototypes
            unique_classes = np.unique(current_y)
            
            if self.model is None:
                # First time: create new model
                self.model = Prototype(self.num_classes, self.input_dim, 
                                      [self.max_proto_num] * self.num_classes, 
                                      self.temperature, self.cal_dis, self.cal_mode).to(self.device)
                self.model = init_model(self.model, self.init_type, self.dbscan_eps, self.min_samples, 
                                       current_X, current_y, feature_min, feature_max)
            else:
                # Incremental: add new prototypes for classes with residual samples
                self._add_prototypes_for_residuals(current_X, current_y, feature_min, feature_max)
            
            # Fine-tune the model with current residual samples
            optimizer = torch.optim.Adam(list(self.model.parameters()), lr=lr)
            if self.label_smoothing and self.label_smoothing > 0:
                criterion = CrossEntropyLabelSmooth(self.num_classes, epsilon=self.label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss()
            
            best_acc = 0.0
            best_state = None
            
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                indices = np.random.permutation(len(current_X))
                for i in range(0, len(current_X), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    batch_x = current_X[batch_idx]
                    batch_y = current_y[batch_idx]
                    
                    batch_x = (batch_x - feature_min) / (feature_max - feature_min)
                    
                    bx = torch.from_numpy(batch_x).float().to(self.device)
                    by = torch.from_numpy(batch_y).long().to(self.device)
                    
                    optimizer.zero_grad()
                    logits = self.model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Evaluate every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        X_norm = (current_X - feature_min) / (feature_max - feature_min)
                        X_t = torch.from_numpy(X_norm).float().to(self.device)
                        logits = self.model(X_t)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        acc = np.mean(preds == current_y)
                        
                        if acc > best_acc:
                            best_acc = acc
                            best_state = copy.deepcopy(self.model.state_dict())
                    self.model.train()
            
            # Restore best model
            if best_state is not None:
                self.model.load_state_dict(best_state)
            
            # Prune low-support prototypes
            self.model = model_prune(self.model, self.num_classes, current_X, current_y, 
                                    feature_min, feature_max, self.max_proto_num, self.prune_T)
            
            # Convert current prototypes to rules
            iter_rules = self._convert_prototypes_to_rules(current_X, current_y)
            generated_rules.extend(iter_rules)
            
            # Find uncovered samples for next iteration (Vectorized)
            uncovered_mask, rule_acc, _ = self._update_coverage_vectorized(current_X, current_y, iter_rules)
            
            # Find uncovered samples for next iteration
            # uncovered_mask = np.ones(len(current_X), dtype=bool)
            # correct_count = 0
            # for i, x in enumerate(current_X):
            #     for rule in iter_rules:
            #         if rule.match(x):
            #             if rule.class_label == current_y[i]:
            #                 correct_count += 1
            #                 uncovered_mask[i] = False
            #             break
            
            # rule_acc = correct_count / len(current_X) if len(current_X) > 0 else 0
            current_X = current_X[uncovered_mask]
            current_y = current_y[uncovered_mask]
            
            logging.info(f"HeliosSystem: Incremental boost iter {boost_iter}, proto_acc={best_acc:.4f}, rule_acc={rule_acc:.4f}, {len(iter_rules)} rules, {len(current_X)} residual")
            
        return generated_rules
    
    def _add_prototypes_for_residuals(self, X, y, feature_min, feature_max):
        """Add new prototypes for classes with residual samples."""
        from sklearn.cluster import KMeans
        
        unique_classes = np.unique(y)
        
        for c in unique_classes:
            c = int(c)
            if c >= self.num_classes:
                continue
                
            c_mask = (y == c)
            X_c = X[c_mask]
            
            if len(X_c) < 2:
                continue
            
            # Normalize
            denom = feature_max - feature_min
            denom[denom == 0] = 1.0
            X_c_norm = (X_c - feature_min) / denom
            
            # Count unique samples to avoid ConvergenceWarning
            unique_samples = np.unique(X_c_norm, axis=0)
            n_unique = len(unique_samples)
            
            # Determine how many new prototypes to add - be more aggressive
            current_proto_count = self.model.class_prototype[c].shape[1]
            max_new = max(1, self.max_proto_num * 2 - current_proto_count)  # Allow 2x max_proto_num
            # More clusters for better coverage: at least 1 per 20 samples
            n_new = min(max_new, max(len(X_c) // 20, 5), 20, n_unique)  # At least 5, at most 20 new prototypes, but not more than unique samples
            
            if n_new < 1:
                continue
            
            # Cluster residual samples to find new prototype centers
            kmeans = KMeans(n_clusters=n_new, n_init='auto', random_state=42)
            kmeans.fit(X_c_norm)
            new_centers = kmeans.cluster_centers_
            
            # Append new prototypes to existing ones
            with torch.no_grad():
                old_proto = self.model.class_prototype[c].data
                old_trans = self.model.class_transform[c].data
                
                new_proto_tensor = torch.from_numpy(new_centers).float().to(self.device).unsqueeze(0)
                new_trans_tensor = torch.ones_like(new_proto_tensor)
                
                combined_proto = torch.cat([old_proto, new_proto_tensor], dim=1)
                combined_trans = torch.cat([old_trans, new_trans_tensor], dim=1)
                
                self.model.class_prototype[c] = nn.Parameter(combined_proto)
                self.model.class_transform[c] = nn.Parameter(combined_trans)
            
            logging.debug(f"HeliosSystem: Added {n_new} prototypes for class {c} (total: {combined_proto.shape[1]})")
            
    def _run_boosting(self, X, y, epochs=20, batch_size=256, lr=0.001):
        generated_rules = []
        
        feature_min = self.feature_min if hasattr(self, 'feature_min') else np.min(X, axis=0)
        feature_max = self.feature_max if hasattr(self, 'feature_max') else np.max(X, axis=0)
        if hasattr(self, 'feature_min'):
            feature_min = np.minimum(feature_min, np.min(X, axis=0))
            feature_max = np.maximum(feature_max, np.max(X, axis=0))
        
        self.feature_min = feature_min
        self.feature_max = feature_max 
        self.feature_max[self.feature_max == self.feature_min] += 1e-6
        
        current_X = X
        current_y = y
        
        logging.info(f"HeliosSystem: Starting boosting with {len(X)} samples, {self.boost_num} iterations")
        
        for boost_iter in range(self.boost_num):
            if len(current_X) == 0:
                logging.info(f"HeliosSystem: Boost iter {boost_iter} - no samples left, stopping")
                break

            # Skip if only one class (upstream cases_continue)
            if len(np.unique(current_y)) <= 1:
                logging.info(f"HeliosSystem: Boost iter {boost_iter} - single class, skipping")
                break
            
            logging.info(f"HeliosSystem: Boost iter {boost_iter} - training on {len(current_X)} samples")
                
            self.model = Prototype(self.num_classes, self.input_dim, 
                                  [self.max_proto_num] * self.num_classes, 
                                  self.temperature, self.cal_dis, self.cal_mode).to(self.device)
            
            self.model = init_model(self.model, self.init_type, self.dbscan_eps, self.min_samples, 
                                   current_X, current_y, self.feature_min, self.feature_max)

            # Skip if no prototypes initialized
            total_proto = sum(p.shape[1] for p in self.model.class_prototype)
            if total_proto == 0:
                logging.info(f"HeliosSystem: Boost iter {boost_iter} - no prototypes, skipping")
                continue
            
            optimizer = torch.optim.Adam(list(self.model.parameters()), lr=lr)
            if self.label_smoothing and self.label_smoothing > 0:
                criterion = CrossEntropyLabelSmooth(self.num_classes, epsilon=self.label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss()
            
            # Training with evaluation
            best_acc = 0.0
            best_state = None
            patience_counter = 0
            patience = 5  # Early stopping patience
            
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                indices = np.random.permutation(len(current_X))
                for i in range(0, len(current_X), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    batch_x = current_X[batch_idx]
                    batch_y = current_y[batch_idx]
                    
                    batch_x = (batch_x - self.feature_min) / (self.feature_max - self.feature_min)
                    
                    bx = torch.from_numpy(batch_x).float().to(self.device)
                    by = torch.from_numpy(batch_y).long().to(self.device)
                    
                    optimizer.zero_grad()
                    logits = self.model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Evaluate on training data every 5 epochs
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    self.model.eval()
                    with torch.no_grad():
                        X_norm = (current_X - self.feature_min) / (self.feature_max - self.feature_min)
                        X_t = torch.from_numpy(X_norm).float().to(self.device)
                        logits = self.model(X_t)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        acc = np.mean(preds == current_y)
                        
                        if acc > best_acc:
                            best_acc = acc
                            best_state = copy.deepcopy(self.model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        avg_loss = total_loss / max(1, len(current_X) // batch_size)
                        logging.debug(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.4f}, best_acc={best_acc:.4f}")
                    
                    self.model.train()
                    
                    # Early stopping
                    if patience_counter >= patience:
                        logging.debug(f"  Early stopping at epoch {epoch+1}")
                        break
            
            # Restore best model
            if best_state is not None:
                self.model.load_state_dict(best_state)
            
            logging.info(f"HeliosSystem: Boost iter {boost_iter} - best training accuracy: {best_acc:.4f}")
            
            self.model = model_prune(self.model, self.num_classes, current_X, current_y, 
                                    self.feature_min, self.feature_max, self.max_proto_num, self.prune_T)
            
            iter_rules = self._convert_prototypes_to_rules(current_X, current_y)
            generated_rules.extend(iter_rules)
            
            # Find uncovered samples for next iteration (Vectorized)
            uncovered_mask, rule_acc, _ = self._update_coverage_vectorized(current_X, current_y, iter_rules)
            
            # Find uncovered samples for next iteration
            # uncovered_mask = np.ones(len(current_X), dtype=bool)
            # correct_count = 0
            # for i, x in enumerate(current_X):
            #     for rule in iter_rules:
            #         if rule.match(x):
            #             if rule.class_label == current_y[i]:
            #                 correct_count += 1
            #                 uncovered_mask[i] = False
            #             break
            
            # rule_acc = correct_count / len(current_X) if len(current_X) > 0 else 0
            logging.info(f"HeliosSystem: Boost iter {boost_iter} - generated {len(iter_rules)} rules, rule accuracy: {rule_acc:.4f}, uncovered: {np.sum(uncovered_mask)}")
            
            current_X = current_X[uncovered_mask]
            current_y = current_y[uncovered_mask]
        
        # Final evaluation on all data
        _, accuracy, coverage = self._update_coverage_vectorized(X, y, generated_rules)
        # total_correct = 0
        # total_covered = 0
        # for i in range(len(X)):
        #     for rule in generated_rules:
        #         if rule.match(X[i]):
        #             total_covered += 1
        #             if rule.class_label == y[i]:
        #                 total_correct += 1
        #             break
        
        # coverage = total_covered / len(X) if len(X) > 0 else 0
        # accuracy = total_correct / len(X) if len(X) > 0 else 0
        logging.info(f"HeliosSystem: Final - {len(generated_rules)} rules, coverage: {coverage:.4f}, accuracy: {accuracy:.4f}")
            
        return generated_rules

    def _convert_prototypes_to_rules(self, X, y):
        """
        Convert prototypes to hypercube matching rules.
        
        Following original Helios logic (Convert_to_rules in incremental_boost.py):
        1. For each prototype, collect samples assigned to it with correct predictions
        2. Calculate distance threshold as mean * threshold_radio (not mean + std!)
        3. Filter samples exceeding threshold (sorted by distance, pop from top)
        4. Create bounding box from remaining samples (min/max per feature)
        
        Key optimization: Use tighter bounding boxes with small expansion for robustness.
        """
        new_rules = []
        
        denom = self.feature_max - self.feature_min
        denom[denom == 0] = 1.0
        X_norm = (X - self.feature_min) / denom
        
        with torch.no_grad():
            self.model.eval()
            
            # Get model predictions first
            X_t = torch.from_numpy(X_norm).float().to(self.device)
            logits = self.model(X_t)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            for c in range(self.num_classes):
                if self.model.class_prototype[c].shape[1] == 0:
                    continue
                
                # Filter: only samples where prediction == label == c
                c_mask = (y == c) & (predictions == c)
                if not np.any(c_mask): 
                    continue
                
                X_c = X_norm[c_mask]
                X_raw_c = X[c_mask]
                
                X_c_t = torch.from_numpy(X_c).float().to(self.device)
                X_c_view = X_c_t.view(X_c_t.shape[0], 1, self.input_dim)
                proto = self.model.class_prototype[c]
                trans = self.model.class_transform[c]
                
                # Calculate distances following original Helios Eq.(1)
                if self.cal_dis == 'l_n':
                    if self.cal_mode == 'abs_trans':
                        dist = torch.abs(X_c_view - proto) * torch.abs(trans)
                    else:
                        dist = torch.abs(X_c_view * torch.abs(trans) - proto)
                    max_d, _ = torch.max(dist, dim=2)
                    dist_vals, min_indices = torch.min(max_d, dim=1)
                else:
                    dist = (X_c_view - proto) ** 2
                    dist_vals = torch.sum(dist, dim=2)
                    dist_vals, min_indices = torch.min(dist_vals, dim=1)
                
                min_indices = min_indices.cpu().numpy()
                dist_vals = dist_vals.cpu().numpy()
                
                # Build distance list per prototype (matching original Calculate_distance_list)
                for pid in range(proto.shape[1]):
                    assigned_mask = (min_indices == pid)
                    if not np.any(assigned_mask):
                        continue
                    
                    pid_dists = dist_vals[assigned_mask]
                    pid_samples = X_raw_c[assigned_mask]
                    
                    if len(pid_dists) == 0:
                        continue
                    
                    # Sort by distance (descending) - matching original logic
                    sorted_indices = np.argsort(pid_dists)[::-1]
                    pid_dists = pid_dists[sorted_indices]
                    pid_samples = pid_samples[sorted_indices]
                    
                    # Calculate threshold: mean * radio (original Helios Eq.(6))
                    # Use a slightly larger radio for better coverage
                    threshold = np.mean(pid_dists) * self.radio
                    
                    # Pop samples above threshold (matching original logic)
                    # Keep removing from top (highest distance) while above threshold
                    keep_start = 0
                    while keep_start < len(pid_dists) - 1 and pid_dists[keep_start] > threshold:
                        keep_start += 1

                    # Take the valid samples (lowest distances)
                    final_samples = pid_samples[keep_start:]
                    
                    if len(final_samples) == 0:
                        continue

                    # Create Bounding Box Rule (min/max per feature)
                    # Add small expansion (1% of range) for robustness against noise
                    mins = np.min(final_samples, axis=0)
                    maxs = np.max(final_samples, axis=0)
                    
                    intervals = []
                    for k in range(self.input_dim):
                        span = maxs[k] - mins[k]
                        if span == 0:
                            span = 1e-6
                        # Small expansion for robustness (1% of span)
                        expansion = span * 0.01
                        intervals.append([float(mins[k] - expansion), float(maxs[k] + expansion)])
                        
                    rule = HeliosRule(c, intervals, confidence=1.0, prototype_idx=pid)
                    rule.support = len(final_samples)
                    new_rules.append(rule)
                    
        return new_rules

    def to_dict_list(self):
        return [{
            'class_label': int(r.class_label),
            'intervals': [[float(v) for v in i] for i in r.intervals],
            'confidence': float(r.confidence)
        } for r in self.rules]

    def save_model(self, path):
        if self.model:
            self.model.save_parameter(path)
