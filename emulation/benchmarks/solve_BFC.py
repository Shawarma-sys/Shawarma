import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 0. Server Configuration
# ==========================================
plt.switch_backend('Agg')

# ==========================================
# 1. Parameter Setup
# ==========================================
L = 5              # Total layers (Action space: 0 to 5)
T_max = 10         # Max Training Time
A_start = 0.50      # Min Accuracy to plot
A_end = 1.0         # Max Accuracy to plot

# Define the Grid (State Space)
# We treat Time and Accuracy as independent state variables to map the whole policy
t_range = np.linspace(0, T_max, 50)
a_range = np.linspace(A_start, A_end, 50)
T_grid, A_grid = np.meshgrid(t_range, a_range)

# Layer Attributes (Simulation)
# BF: Cost savings (Assumed decreasing with depth)
layer_costs = np.array([0] + [100 - i*8 for i in range(L)]) 
cum_saved_cost = np.cumsum(layer_costs) 

# Info: Information Loss (Assumed increasing with depth)
layer_infos = np.array([0] + [i*1.5 for i in range(L)])
cum_lost_info = np.cumsum(layer_infos) 

# Target Accuracy for formula
A_idle = 0.95 

# ==========================================
# 2. Compute Optimal Policy n*(t, A)
# ==========================================
# Matrix to store the optimal layer choice (Z axis)
Z_best_n = np.zeros_like(T_grid)

# Helper function to get weights
def get_weights(acc, target):
    if acc >= target: acc = target - 1e-5
    ws = acc / target
    exponent = 1 - (acc / target)
    wl = (1 - target) + (2 * target - 1) * (3 ** exponent)
    return ws, wl

# Iterate through every state (t, A)
for i in range(T_grid.shape[0]):
    for j in range(T_grid.shape[1]):
        t_val = T_grid[i, j]
        a_val = A_grid[i, j]
        
        # 1. Calculate Time Urgency g(t)
        # g(t) decays as t -> T_max
        g_t = 1.0 - (0.8 * t_val / T_max) 
        if g_t < 0.1: g_t = 0.1 # Floor value
        
        # 2. Calculate Accuracy Weights
        w_save, w_loss = get_weights(a_val, A_idle)
        
        # 3. Solve for best n
        best_n = 0
        max_bfc = -np.inf
        IC_max = 0.15 # Constant
        
        # Grid search for optimal n at this specific state
        for n in range(L + 1):
            gain = w_save * g_t * IC_max * cum_saved_cost[n]
            loss = w_loss * g_t * cum_lost_info[n]
            bfc = gain - loss
            
            if bfc > max_bfc:
                max_bfc = bfc
                best_n = n
        
        Z_best_n[i, j] = best_n

# ==========================================
# 3. Simulate a "Real" Training Trajectory
# ==========================================
# Let's verify: In a real run, accuracy increases with time.
# We plot this path on the surface to show what typically happens.
traj_t = np.linspace(0, T_max, 50)
# Logarithmic growth of accuracy
traj_a = A_start + (A_idle - A_start + 0.02) * np.log(1 + 8 * traj_t/T_max) / np.log(9)
traj_a = np.clip(traj_a, A_start, A_end)

# Find the n* for each point on this trajectory
traj_n = []
for k in range(len(traj_t)):
    # Find nearest index in grid to lookup Z (simplified)
    t_idx = np.abs(t_range - traj_t[k]).argmin()
    a_idx = np.abs(a_range - traj_a[k]).argmin()
    traj_n.append(Z_best_n[a_idx, t_idx])
traj_n = np.array(traj_n)

# ==========================================
# 4. Visualization
# ==========================================
fig = plt.figure(figsize=(20, 9))

# --- Subplot 1: 3D Surface Policy Map ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# Plot Surface (Z = Layer Count)
surf = ax1.plot_surface(T_grid, A_grid, Z_best_n, cmap=cm.viridis, 
                       linewidth=0, antialiased=True, alpha=0.9)

# Plot Trajectory
ax1.plot(traj_t, traj_a, traj_n + 0.2, color='red', linewidth=3, 
         label='Simulated Training Run', linestyle='--')

ax1.set_xlabel('Time ($t$)', fontsize=14, labelpad=10)
ax1.set_ylabel('Accuracy ($A(z_t)$)', fontsize=14, labelpad=10)
ax1.set_zlabel('Optimal Frozen Layers ($n$)', fontsize=14, labelpad=10)
ax1.set_title('Policy Surface: n* vs (t, A)', fontsize=16)
ax1.set_zlim(0, L)
ax1.view_init(elev=35, azim=-120) # View from "high accuracy, late time" corner

# --- Subplot 2: 3D Bar Policy (Discrete Blocks) ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Downsample for bar plot (too many bars is slow/ugly)
step = 2 # Skip every 2nd point
_t = T_grid[::step, ::step].flatten()
_a = A_grid[::step, ::step].flatten()
_z = np.zeros_like(_t)
_n = Z_best_n[::step, ::step].flatten()

# Bar dimensions
dt = (T_max / 50) * step * 0.8
da = ((A_end - A_start) / 50) * step * 0.8

# Color mapping based on n (layer count)
norm = plt.Normalize(0, L)
colors = cm.viridis(norm(_n))

ax2.bar3d(_t, _a, _z, dt, da, _n, color=colors, shade=True)

# Plot Trajectory on top
ax2.plot(traj_t, traj_a, traj_n + 0.5, color='red', linewidth=4, 
         label='Simulated Training Run', zorder=10)

ax2.set_xlabel('Time ($t$)', fontsize=14, labelpad=10)
ax2.set_ylabel('Accuracy ($A(z_t)$)', fontsize=14, labelpad=10)
ax2.set_zlabel('Frozen Layers ($n$)', fontsize=14, labelpad=10)
ax2.set_title('Discrete Policy Map', fontsize=16)
ax2.set_zlim(0, L)
ax2.view_init(elev=35, azim=-120)

# Shared Colorbar
cbar = fig.colorbar(surf, ax=[ax1, ax2], shrink=0.6, aspect=20, pad=0.05)
cbar.set_label('Number of Frozen Layers', fontsize=14)
# Make colorbar discrete ticks
cbar.set_ticks(np.arange(0, L+1, 2))

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

filename = 'policy_map_t_acc.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully as '{filename}'")