import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List

@torch.jit.script
def compute_last_grad(prob: torch.Tensor, label: torch.Tensor, fc_weight: torch.Tensor, num_learned_class: int):
    """
    Returns a scalar tensor containing the squared-grad sum for the last-layer gradient.
    Avoid converting to a Python float here to prevent implicit device synchronization inside
    the JIT function; let the caller decide when to move data to CPU or convert to float.
    """
    # oh_label = F.one_hot(label.long(), num_learned_class)
    # grad = torch.matmul((prob - oh_label), fc_weight)
    # Optimized: avoid full one_hot expansion if possible, or use indexing
    # But for matmul, (prob - one_hot) * weight is standard.
    # Let's keep it but ensure types are correct to avoid casts.
    grad = torch.matmul((prob - F.one_hot(label.long(), num_learned_class).to(prob.dtype)), fc_weight)
    return (grad ** 2).sum()

@torch.jit.script
def compute_grads_sq_sum(grads: List[torch.Tensor]):
    """
    Returns a scalar tensor equal to the sum of squared gradients. Do not call
    .item() here to avoid device synchronization inside the JIT function.
    """
    if len(grads) == 0:
        return torch.tensor(0.0)
    # Optimization: Avoid cat() which allocates new memory. Sum squares individually.
    # But JIT loop might be slower than cat + sum for many small tensors.
    # For typical layers, cat is okay. But let's try to avoid flatten overhead if possible.
    # Actually, stack is better if shapes are same, but they might not be (bias vs weight).
    # Let's stick to a simple loop accumulation which JIT can optimize well.
    sum_sq = torch.tensor(0.0, device=grads[0].device)
    for g in grads:
        sum_sq += torch.sum(g ** 2)
    return sum_sq

def compute_freeze_idx_np(cumulative_backward_flops, cumulative_fisher, total_flops, batch_freeze_score, num_model, total_fisher, epoch_f1, idle_f1, g_t, g_t_max):
    """
    Pure numpy implementation of the freeze index calculation.
    Optimized: Pre-compute constants, avoid redundant array allocations.
    """
    # Pre-compute constants to avoid repeated division
    inv_total_flops = 1.0 / (total_flops + 1e-10)
    inv_total_fisher = 1.0 / (total_fisher + 1e-10)
    
    # Compute freeze_score in-place style (avoid intermediate array)
    denom_flops = total_flops - cumulative_backward_flops + 1e-10
    freeze_score = total_flops / denom_flops * (total_fisher - cumulative_fisher + 1e-10) * inv_total_fisher
    
    max_score = freeze_score.max()  # Use .max() method instead of np.max for small arrays
    
    # Compute fisher_saved and fisher_loss using pre-computed inverses
    fisher_saved = cumulative_backward_flops * inv_total_flops * max_score
    fisher_loss = cumulative_fisher * inv_total_fisher * batch_freeze_score
    
    # w_save = A(z_t) / A_idle
    w_save = epoch_f1 / idle_f1
    # w_loss = 1 − A_idle + (2*A_idle − 1) * cbrt(1 − A(z_t)/A_idle)
    # Optimization: Pre-compute cbrt argument
    cbrt_arg = 1.0 - w_save
    w_loss = 1.0 - idle_f1 + (2.0 * idle_f1 - 1.0) * np.cbrt(cbrt_arg)
    
    # Compute modified_score directly without intermediate allocation
    modified_score = w_save * g_t_max * fisher_saved - w_loss * g_t * fisher_loss
    
    # Find optimal freeze index
    max_modified = modified_score.max()
    if max_modified > 0:
        # Optimization: Use argmax for single max, then check for ties only if needed
        max_idx = int(modified_score.argmax())
        # Quick check if there might be ties (only check neighbors)
        if max_idx < num_model - 1 and np.isclose(modified_score[max_idx + 1], max_modified, rtol=1e-6, atol=1e-12):
            # Full tie-breaking only when necessary
            max_idxs = np.flatnonzero(np.isclose(modified_score, max_modified, rtol=1e-6, atol=1e-12))
            optimal_freeze = int(max_idxs[-1] + 1)
        else:
            optimal_freeze = max_idx + 1
    else:
        optimal_freeze = 0
    
    return optimal_freeze

class LayerFreezer:
    """
    Adaptive layer freezer for torchlens-style layer list models
    """
    def __init__(self, model, ema_ratio=0.1, T=10, idle_f1=0.95, prewarm=True):
        self.model = model  # torchlens layer list
        self.blockwise_backward_flops = []
        self.total_model_flops = 0
        self.ema_ratio = ema_ratio
        self.fisher = np.zeros(max(0, len(self.model) - 2), dtype=np.float32)
        self.cumulative_fisher = []
        self.freeze_idx = []
        self.last_grad_mean = 0.0
        self.fc_weight = None
        self.last_param_layer_idx = None
        self.num_learned_class = 0
        self.last_grad = 0.0
        self.idle_f1 = idle_f1
        self.g_t_max_so_far = 0.0
        # Horizon for g(t)
        self.T = max(1, int(T))
        self.init_model_params()
        # Pre-warm helper functions so their one-time setup cost does not
        # show up in the first fisher or freeze timing.
        if prewarm:
            try:
                self._warmup_helpers(prewarm=True)
            except Exception:
                # If warmup fails for any reason, don't break initialization.
                pass

    def init_model_params(self):
        """
        Initialize model parameters and calculate total FLOPs.
        This method calculates the total FLOPs of the model and identifies the last layer with parameters.
        """
        self.blockwise_backward_flops = [getattr(layer, 'backward_flops', 0) or 0 for layer in self.model[1:-1]]
        forward_flops = [getattr(layer, 'flops', 0) or 0 for layer in self.model[1:-1]]
        self.total_model_flops = sum(forward_flops) + sum(self.blockwise_backward_flops)
        self.cumulative_backward_flops = np.cumsum(np.array(self.blockwise_backward_flops, dtype=np.float32))
        # Find the last layer with parameters
        for idx in range(len(self.model) - 1, -1, -1):
            params = getattr(self.model[idx], 'parent_params', [])
            if params:
                self.fc_weight = params[0]
                self.last_param_layer_idx = idx
                break
        if self.fc_weight is None:
            raise RuntimeError('No layer with parent_params found')
        self.num_learned_class = self.fc_weight.shape[0]

    def calculate_fisher(self, fisher_ema_ratio=0.5, *, prewarm: bool = False):
        # Optimization: Pre-fetch device once and reuse
        device = None
        for layer in self.model:
            params = getattr(layer, 'parent_params', [])
            if params:
                try:
                    device = params[0].device
                    break
                except Exception:
                    pass
        if device is None:
            device = torch.device('cpu')

        # Optimization: Pre-allocate list with known size
        num_layers = len(self.model) - 2
        per_layer_tensors = []
        
        # Use enumerate with start to skip first layer
        for i in range(1, len(self.model) - 1):
            layer = self.model[i]
            params = getattr(layer, 'parent_params', [])
            
            # Optimization: Inline gradient sum computation to avoid function call overhead
            sum_sq = torch.tensor(0.0, device=device)
            for p in params:
                if p.requires_grad and p.grad is not None:
                    g = p.grad
                    if not torch.isnan(g).any():
                        sum_sq = sum_sq + torch.sum(g * g)  # Avoid ** operator
            
            per_layer_tensors.append(sum_sq)

        if len(per_layer_tensors) == 0:
            layer_fisher = np.zeros(num_layers, dtype=np.float32)
        else:
            # Optimization: Stack and transfer in one operation
            stacked = torch.stack(per_layer_tensors)
            layer_fisher = stacked.detach().cpu().numpy().astype(np.float32, copy=False)

        if prewarm:
            return layer_fisher

        # Update EMA-style fisher values
        if not hasattr(self, 'fisher') or np.all(self.fisher == 0):
            self.fisher = layer_fisher.copy()
        else:
            # Optimization: In-place update
            self.fisher += fisher_ema_ratio * (layer_fisher - self.fisher)
        
        self.total_fisher = float(self.fisher.sum())
        self.cumulative_fisher = np.cumsum(self.fisher)
        return self.fisher

    def get_freeze_idx(self, prob, label, epoch_f1, t, F1_d, prewarm=False):
        # t_time = time.time()
        # Use JIT-compiled function for the gradient computation
        # compute_last_grad now returns a scalar tensor; convert explicitly and on-demand.
        try:
            last_grad_t = compute_last_grad(prob, label, self.fc_weight, self.num_learned_class)
            last_grad_val = float(last_grad_t.detach().cpu().item())
        except Exception:
            last_grad_val = 0.0

        if prewarm:
            # Do not update state during prewarm
            batch_freeze_score = 1.0
        else:
            self.last_grad = last_grad_val
            if self.last_grad_mean == 0.0:
                self.last_grad_mean = self.last_grad
            batch_freeze_score = self.last_grad / (self.last_grad_mean + 1e-10)
            # print(f"last_grad: {self.last_grad}, batch freeze score: {batch_freeze_score}")
            self.last_grad_mean += self.ema_ratio * (self.last_grad - self.last_grad_mean)

        # Prepare numpy buffers with safe fallbacks for warmup or first-run scenarios
        if not isinstance(self.cumulative_fisher, np.ndarray) or self.cumulative_fisher.size == 0:
            cumulative_fisher = np.zeros_like(self.cumulative_backward_flops, dtype=np.float32)
        else:
            cumulative_fisher = self.cumulative_fisher

        total_fisher = getattr(self, "total_fisher", float(cumulative_fisher.sum()))
        if isinstance(total_fisher, (list, tuple)):
            total_fisher = float(np.sum(total_fisher))
        if total_fisher <= 0:
            total_fisher = 1.0

        # Compute g(t) per request (F1_d, t provided externally):
        # g(t) = 1/T * [sum_{k=0}^t (T-k)/T * F1_d + sum_{k=t+1}^T (T-k)/T * F1(t)]
        def _compute_g_t_np(t_val, T_val, F1_d_val, F1_t_val) -> float:
            T_val = max(1, int(T_val))
            t_val = int(max(0, min(t_val, T_val)))
            k = np.arange(0, T_val + 1, dtype=np.float32)
            weights = (T_val - k) / float(T_val)
            mask1 = k <= t_val
            mask2 = k > t_val
            s = weights[mask1].sum(dtype=np.float32) * float(F1_d_val) + weights[mask2].sum(dtype=np.float32) * float(F1_t_val)
            return float(s / float(T_val))

        g_t = _compute_g_t_np(t, self.T, float(F1_d), float(epoch_f1))
        self.g_t_max_so_far = max(self.g_t_max_so_far, g_t)

        # Use the numpy implementation for the core computation
        optimal_freeze = compute_freeze_idx_np(
            self.cumulative_backward_flops,
            cumulative_fisher,
            self.total_model_flops,
            batch_freeze_score,
            len(self.model) - 2,
            total_fisher,
            epoch_f1,
            self.idle_f1,
            g_t,
            self.g_t_max_so_far,
        )
        if prewarm:
            return list(range(int(optimal_freeze) + 1))
        self.freeze_idx = list(range(int(optimal_freeze) + 1))
        if len(self.freeze_idx) == len(self.model) - 1:
            return self.freeze_idx, True
        else:
            return self.freeze_idx, False
    def visualize_policy(self, save_path):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            return

        plt.switch_backend('Agg')

        # Setup Grid
        L = len(self.model) - 2 # Action space 0 to L
        T_max = float(self.T)
        A_start = 0.50
        A_end = 1.0
        
        t_range = np.linspace(0, T_max, 50)
        a_range = np.linspace(A_start, A_end, 50)
        T_grid, A_grid = np.meshgrid(t_range, a_range)
        
        Z_best_n = np.zeros_like(T_grid)
        
        # Ensure we have data
        if not isinstance(self.cumulative_backward_flops, np.ndarray):
            return
            
        total_flops = self.total_model_flops
        total_fisher = getattr(self, "total_fisher", 1.0)
        num_model = len(self.model) - 2 
        
        # Iterate
        for i in range(T_grid.shape[0]):
            for j in range(T_grid.shape[1]):
                t_val = T_grid[i, j]
                a_val = A_grid[i, j]
                
                # Reusing _compute_g_t_np logic inline:
                T_val = max(1, int(self.T))
                cur_t = int(max(0, min(t_val, T_val)))
                
                k = np.arange(0, T_val + 1, dtype=np.float32)
                weights = (T_val - k) / float(T_val)
                mask1 = k <= cur_t
                mask2 = k > cur_t
                
                F1_d_val = self.idle_f1 
                s = weights[mask1].sum() * F1_d_val + weights[mask2].sum() * a_val
                g_t = float(s / float(T_val))
                
                best_n = compute_freeze_idx_np(
                    self.cumulative_backward_flops,
                    self.cumulative_fisher,
                    total_flops,
                    1.0, # batch_freeze_score
                    num_model,
                    total_fisher,
                    a_val, # epoch_f1
                    self.idle_f1,
                    g_t,
                    g_t # g_t_max approximation for static map
                )
                Z_best_n[i, j] = best_n

        # Visualization
        fig = plt.figure(figsize=(20, 9))
        
        # --- Simulate a "Real" Training Trajectory (for visualization only) ---
        traj_t = np.linspace(0, T_max, 50)
        # Logarithmic growth of accuracy
        traj_a = A_start + (self.idle_f1 - A_start + 0.02) * np.log(1 + 8 * traj_t/T_max) / np.log(9)
        traj_a = np.clip(traj_a, A_start, A_end)
        
        # Find the n* for each point on this trajectory
        traj_n = []
        for k in range(len(traj_t)):
            # Find nearest index in grid to lookup Z (simplified)
            t_idx = np.abs(t_range - traj_t[k]).argmin()
            a_idx = np.abs(a_range - traj_a[k]).argmin()
            traj_n.append(Z_best_n[a_idx, t_idx])
        traj_n = np.array(traj_n)

        # --- Subplot 1: 3D Surface Policy Map ---
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax1.plot_surface(T_grid, A_grid, Z_best_n, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)
        
        # Plot Trajectory
        ax1.plot(traj_t, traj_a, traj_n + 0.2, color='red', linewidth=3, 
                label='Simulated Training Run', linestyle='--')
        
        ax1.set_xlabel('Time ($T$)', fontsize=14, labelpad=10)
        ax1.set_ylabel('Accuracy ($A(z_t)$)', fontsize=14, labelpad=10)
        ax1.set_zlabel('Optimal Frozen Layers ($n$)', fontsize=14, labelpad=10)
        ax1.set_title('Policy Surface: n* vs ($T$, $A(z_t)$)', fontsize=16)
        ax1.set_zlim(0, L)
        ax1.view_init(elev=35, azim=-120)
        ax1.legend(loc='upper left')

        # --- Subplot 2: 3D Bar Policy (Discrete Blocks) ---
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        # Downsample for bar plot
        step = 2 
        _t = T_grid[::step, ::step].flatten()
        _a = A_grid[::step, ::step].flatten()
        _z = np.zeros_like(_t)
        _n = Z_best_n[::step, ::step].flatten()

        # Bar dimensions
        dt = (T_max / 50) * step * 0.8
        da = ((A_end - A_start) / 50) * step * 0.8

        # Color mapping based on n
        norm = plt.Normalize(0, L)
        colors = cm.viridis(norm(_n))

        ax2.bar3d(_t, _a, _z, dt, da, _n, color=colors, shade=True)

        # Plot Trajectory on top
        ax2.plot(traj_t, traj_a, traj_n + 0.5, color='red', linewidth=4, 
                label='Simulated Training Run', zorder=10)

        ax2.set_xlabel('Time ($T$)', fontsize=14, labelpad=10)
        ax2.set_ylabel('Accuracy ($A(z_t)$)', fontsize=14, labelpad=10)
        ax2.set_zlabel('Frozen Layers ($n$)', fontsize=14, labelpad=10)
        ax2.set_title('Discrete Policy Map', fontsize=16)
        ax2.set_zlim(0, L)
        ax2.view_init(elev=35, azim=-120)
        ax2.legend(loc='upper left')
        
        # Shared Colorbar
        cbar = fig.colorbar(surf, ax=[ax1, ax2], shrink=0.6, aspect=20, pad=0.05)
        cbar.set_label('Number of Frozen Layers', fontsize=14)
        cbar.set_ticks(np.arange(0, L+1, 2))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy visualization saved to {save_path}")
        plt.close()

    def freeze_layers(self):
        if self.freeze_idx:
            for i in self.freeze_idx:
                self.freeze_layer(i)

    def freeze_layer(self, layer_index):
        for p in getattr(self.model[layer_index], 'parent_params', []):
            p.requires_grad = False

    def unfreeze_layers(self):
        for layer in self.model:
            for p in getattr(layer, 'parent_params', []):
                p.requires_grad = True
        self.freeze_idx = []

    def _warmup_helpers(self, prewarm=True):
        """
        Call the helper functions once with small dummy inputs to trigger
        any one-time setup (JIT scripts, CUDA kernels, numpy caches) ahead
        of measurement so the first real batch has less overhead.
        """
        # Warm up torch.jit.script functions with trivial tensors
        # Determine device for warmup: prefer fc_weight device or provided device
        warmup_device = None
        try:
            warmup_device = self.fc_weight.device
        except Exception:
            warmup_device = torch.device('cpu')

        try:
            dummy_grad = [torch.zeros(1, device=warmup_device)]
            # compute_grads_sq_sum expects a list of tensors
            compute_grads_sq_sum(dummy_grad)
        except Exception:
            pass

        try:
            # compute_last_grad requires (prob, label, fc_weight, num_learned_class)
            if self.fc_weight is not None:
                prob = torch.zeros((1, max(1, int(self.num_learned_class))), device=warmup_device)
                label = torch.zeros(1, dtype=torch.long, device=warmup_device)
                dummy_fc = self.fc_weight.detach().to(warmup_device)
                # Ensure num_learned_class at least 1
                compute_last_grad(prob, label, dummy_fc, max(1, int(self.num_learned_class)))
        except Exception:
            pass

        try:
            # Warm up freeze-index helper: call with small numpy arrays to
            # populate any numpy caches and ensure consistent code paths.
            cumulative_backward_flops = np.array(self.blockwise_backward_flops[1:], dtype=np.float32)
            cumulative_fisher = np.zeros_like(cumulative_backward_flops)
            total_flops = float(self.total_model_flops) if hasattr(self, 'total_model_flops') else 1.0
            batch_freeze_score = 1.0
            num_model = len(self.model)
            total_fisher = float(max(1.0, np.sum(self.fisher)))
            epoch_f1 = 0.9
            idle_f1 = 0.9
            # Compute a simple g(t) for warmup with t=0 and F1_d defaulting to idle_f1
            T_val = max(1, int(self.T))
            k = np.arange(0, T_val + 1, dtype=np.float32)
            weights = (T_val - k) / float(T_val)
            s = weights[k <= 0].sum(dtype=np.float32) * float(idle_f1) + weights[k > 0].sum(dtype=np.float32) * float(epoch_f1)
            g_t = float(s / float(T_val))
            # Run once during init to avoid numpy allocations on the first batch
            compute_freeze_idx_np(cumulative_backward_flops, cumulative_fisher, total_flops, batch_freeze_score, num_model, total_fisher, epoch_f1, idle_f1, g_t, g_t)
        except Exception:
            pass

        self._prewarm_fisher_buffers(prewarm=prewarm)

    def _prewarm_fisher_buffers(self, prewarm=True):
        """Populate parameter grads with zeros and run a dry Fisher pass."""
        backup_grads = []
        try:
            for layer in self.model:
                params = getattr(layer, 'parent_params', [])
                layer_backup = []
                for param in params:
                    layer_backup.append(param.grad)
                    if param.requires_grad:
                        try:
                            param.grad = torch.zeros_like(param)
                        except Exception:
                            param.grad = None
                backup_grads.append((params, layer_backup))

            self.calculate_fisher(prewarm=prewarm)
        except Exception:
            pass
        finally:
            for params, grads in backup_grads:
                for param, grad in zip(params, grads):
                    param.grad = grad