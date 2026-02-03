#!/usr/bin/env python3
# *************************************************************************
#
# Comparison experiment for LINC vs Original Retrain vs Adaptive Retrain
# 
# LINC: Rule-based incremental model update (ICNP 2024)
# Original Retrain: Full model retraining without layer freezing
# Adaptive Retrain: Adaptive layer freezing based retraining
#
# All three methods use the same control plane framework for fair comparison.
#
# Supports: MLP1, TextCNN1, RNN1
#
# *************************************************************************

import os
import sys
import argparse
import subprocess
import time
import signal
import datetime
import json
import numpy as np

Shawarma_home = os.getenv("Shawarma_HOME")
if Shawarma_home is None:
    Shawarma_home = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    os.environ["Shawarma_HOME"] = Shawarma_home
    print(f"Shawarma_HOME not set, using: {Shawarma_home}")

# Sequence model configurations
SEQUENCE_MODELS = {"TextCNN1", "TextCNN2", "RNN1"}

COMMON_SEQUENCE_DEFAULTS = {
    "sequence_length": 9,
    "sequence_feature_dim": 2,
    "len_vocab": 1501,
    "ipd_vocab": 2561,
    "len_embedding_bits": 10,
    "ipd_embedding_bits": 8,
}

TEXTCNN_DEFAULTS = {
    "textcnn_nk": 4,
    "textcnn_ebdin": 4,
}

RNN_DEFAULTS = {
    "rnn_in": 12,
    "rnn_hidden": 16,
    "rnn_dropout": 0.0,
}

# LINC-specific configurations
LINC_DEFAULTS = {
    "rule_threshold": 0.3,        # Confidence threshold for rule activation
    "max_rules": 1000,            # Maximum number of rules to maintain
    "update_samples": 4000,       # Number of samples for incremental update
}

# Helios-specific configurations
# Following original Helios paper parameters with optimizations
HELIOS_DEFAULTS = {
    "max_rules": 3000,          # More rules for better coverage
    "radio": 1.5,               # threshold_radio from original (mean * radio) - larger for better coverage
    "boost_num": 6,             # Number of boosting iterations (more iterations for better rules)
    "prune_rule": 3,            # Rule pruning threshold (lower to keep more rules)
}


def append_sequence_model_args(cmd, model_class, num_classes=None):
    """Append sequence model specific arguments to command."""
    if model_class not in SEQUENCE_MODELS:
        return
    
    # Add num_classes first (required for sequence models)
    if num_classes is not None:
        cmd.extend(["--num_classes", str(num_classes)])
    
    for name, value in COMMON_SEQUENCE_DEFAULTS.items():
        cmd.extend([f"--{name}", str(value)])
    
    if model_class in ["TextCNN1", "TextCNN2"]:
        for name, value in TEXTCNN_DEFAULTS.items():
            cmd.extend([f"--{name}", str(value)])
    elif model_class == "RNN1":
        for name, value in RNN_DEFAULTS.items():
            cmd.extend([f"--{name}", str(value)])


def run_command(cmd, description, env=None):
    print(f"Starting: {description}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Finished: {description}\n")


def kill_process_on_port(port):
    print(f"Cleaning up port {port}...")
    try:
        cmd = ["lsof", "-t", f"-i:{port}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                print(f"Killing process {pid} on port {port}")
                os.kill(int(pid), signal.SIGKILL)
            time.sleep(2)
    except Exception as e:
        print(f"Port cleanup failed: {e}")


def run_single_experiment(args, retrain_mode, results_file, clean_env):
    """
    Run a single experiment with specified retraining mode.
    
    retrain_mode: 'linc', 'original', 'adaptive', or 'static'
    
    All modes use the same control plane framework for fair comparison.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {retrain_mode} ({args.model_class})")
    print(f"{'='*60}\n")
    
    kill_process_on_port(50051)
    
    # Build control plane command
    cmd_cp = [
        sys.executable,
        f"{Shawarma_home}/emulation/benchmarks/control_plane_retrain_torchlens.py",
        "--device", args.cp_device,
        "--application_type", "intrusion_detection",
        "--job_name", f"cp_{retrain_mode}_{args.dataset_name}_{args.model_class}_retrain",
        "--labeler_type", "DNN_classifier",
        "--labeler_dnn_class", args.labeler_class,
        "--labeler_dnn_path", f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/best_{args.labeler_class}_{args.dataset_name}.pth",
        "-r", "0.005",
        "--total_epochs", "60",
        "--model_class", args.model_class,
        "-m", f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/best_{args.model_class}_{args.dataset_name}.pth",
        "-o", results_file,
        "--logdir", f"{Shawarma_home}/emulation/logs/control_plane",
        "--test_size", "100",
        "--memory_size", "5000",
        "--label_diff_threshold", "0.05",
        "--memory_sample_ratio", "0.1",
        "--retrain_batch_size", "512",
        "--use_ground_truth_for_drift",
    ]
    
    # Add seed and deterministic flags if provided
    if args.seed is not None:
        cmd_cp.extend(["--seed", str(args.seed)])
    if args.deterministic:
        cmd_cp.append("--deterministic")
    
    # Add sequence model specific arguments (TextCNN1, RNN1, etc.)
    append_sequence_model_args(cmd_cp, args.model_class, num_classes=args.num_classes)
    
    # Add mode-specific flags
    if retrain_mode == 'adaptive':
        cmd_cp.append("--adaptive_freeze")
    elif retrain_mode == 'helios':
        cmd_cp.append("--helios_mode")
        cmd_cp.extend(["--helios_max_rules", str(HELIOS_DEFAULTS['max_rules'])])
        cmd_cp.extend(["--helios_radio", str(HELIOS_DEFAULTS['radio'])])
        cmd_cp.extend(["--helios_boost_num", str(HELIOS_DEFAULTS['boost_num'])])
        cmd_cp.extend(["--helios_prune_rule", str(HELIOS_DEFAULTS['prune_rule'])])
        
        # Add pre-trained Helios rules path if available
        helios_rules_path = f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/helios_rules_{args.model_class}_{args.dataset_name}.json"
        if os.path.exists(helios_rules_path):
            cmd_cp.extend(["--helios_rules_path", helios_rules_path])
            print(f"Using pre-trained Helios rules from: {helios_rules_path}")
        else:
            print(f"Note: Pre-trained Helios rules not found at {helios_rules_path}")
            print("Helios will start with empty rules and use DNN predictions until first drift")
    elif retrain_mode == 'linc':
        # LINC mode (original): use rule-based incremental update with SoftTreeClassifier
        cmd_cp.append("--linc_mode")
        cmd_cp.extend(["--linc_max_rules", str(LINC_DEFAULTS['max_rules'])])
        cmd_cp.extend(["--linc_rule_threshold", str(LINC_DEFAULTS['rule_threshold'])])
        cmd_cp.extend(["--linc_update_samples", str(LINC_DEFAULTS['update_samples'])])
        
        # Add pre-trained LINC rules path
        linc_rules_path = f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/linc_rules_{args.model_class}_{args.dataset_name}.json"
        if os.path.exists(linc_rules_path):
            cmd_cp.extend(["--linc_rules_path", linc_rules_path])
            print(f"Using pre-trained LINC rules from: {linc_rules_path}")
        else:
            print(f"Warning: Pre-trained LINC rules not found at {linc_rules_path}")
            print("LINC will start with empty rules and use DNN predictions until first drift")
    # For 'original' and 'static', no additional flags needed
    # 'static' will be derived from results (no retraining happens when no drift)
    
    # Start control plane
    cp_process = subprocess.Popen(cmd_cp, env=clean_env, start_new_session=True)
    time.sleep(10)
    
    if cp_process.poll() is not None:
        print(f"Error: Control Plane failed to start (exit code {cp_process.returncode})")
        return None
    
    # Build data plane command
    dp_env = clean_env.copy()
    dp_env["SHAWARMA_DP_MODEL_CLASS"] = args.model_class
    dp_env["SHAWARMA_DP_BASE_MODEL_PATH"] = f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/best_{args.model_class}_{args.dataset_name}.pth"
    dp_env["SHAWARMA_DP_NUM_CLASSES"] = str(args.num_classes)
    
    # For LINC modes, pass rules path to data plane via environment variable
    if retrain_mode == 'linc':
        # LINC: use improved rules
        linc_rules_path = f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/linc_rules_{args.model_class}_{args.dataset_name}.json"
        if os.path.exists(linc_rules_path):
            dp_env["SHAWARMA_DP_LINC_RULES_PATH"] = linc_rules_path
            print(f"Data plane will load pre-trained LINC rules from: {linc_rules_path}")
        else:
            print(f"Warning: Pre-trained LINC rules not found at {linc_rules_path}")
            print("LINC will start with empty rules and use DNN predictions until first drift")
    
    cmd_dp = [
        sys.executable,
        f"{Shawarma_home}/emulation/scripts/experiments/data_plane_torchlens_experiment.py",
        "--feature_file", args.feature_file,
        "--label_file", args.label_file,
        "--device", args.dp_device,
    ]
    
    try:
        run_command(cmd_dp, f"Data Plane ({retrain_mode})", env=dp_env)
    except Exception as e:
        print(f"Experiment failed: {e}")
    finally:
        if cp_process.poll() is None:
            print("Waiting for Control Plane to shutdown...")
            try:
                cp_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("Control Plane did not shutdown gracefully. Killing...")
                try:
                    os.killpg(os.getpgid(cp_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        kill_process_on_port(50051)
    
    return results_file


def load_results(results_file):
    """Load experiment results from JSONL or JSON file."""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        content = f.read().strip()
        if content.startswith('{'):
            # JSON format
            return json.loads(content)
        else:
            # JSONL format
            for line in content.split('\n'):
                if line.strip():
                    return json.loads(line)
    return None


def plot_comparison(results_dict, output_dir, experiment_name):
    """Plot comparison of accuracy and F1 scores for all methods."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # --- Style Configuration ---
    # Attempt to use a cleaner style if available
    available_styles = plt.style.available
    if 'seaborn-v0_8-paper' in available_styles:
        plt.style.use('seaborn-v0_8-paper')
    elif 'ggplot' in available_styles:
        plt.style.use('ggplot')
    else:
        plt.style.use('default')

    # Academic Paper Settings (TrueType fonts, Times New Roman)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Custom RC params for professional/academic look
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif'],
        'font.size': 28,
        'axes.labelsize': 32,
        'axes.titlesize': 32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 28,
        'lines.linewidth': 3.0,
        'figure.autolayout': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    # Improved Color Palette (High contrast, professional)
    colors = {
        'linc': '#008B8B',           # DarkCyan/Teal (Distinct baseline)
        'helios': '#9467BD',         # Purple (Distinct baseline)
        'original': '#1F77B4',       # Blue (Distinct baseline)
        'adaptive': '#D62728',       # Red (Ours, prominent)
        'static': '#999999',         # Gray (Less conspicuous)
    }
    
    # Define z-orders to ensure our method is on top
    z_orders = {
        'linc': 3,
        'helios': 3,
        'original': 3,
        'adaptive': 5,           # Topmost
        'static': 2              # Bottom
    }
    
    markers = {
        'linc': 'D',           # Diamond
        'helios': '^',         # Triangle Up
        'original': 'o',       # Circle
        'adaptive': 's',       # Square
        'static': 'X',         # X
    }
    
    labels = {
        'linc': 'LINC',
        'helios': 'Helios',
        'original': 'Caravan',
        'adaptive': 'Shawarma',
        'static': 'Static',
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine max length
    max_len = 0
    for mode, data in results_dict.items():
        if data:
            if 'my_f1s' in data:
                max_len = max(max_len, len(data.get('my_f1s', [])))
            if 'static_f1s' in data:
                max_len = max(max_len, len(data.get('static_f1s', [])))
    
    if max_len == 0:
        print("No data to plot")
        return
    
    windows = list(range(1, max_len + 1))
    
    # Helper to add phase backgrounds
    def add_phase_backgrounds(ax, length):
        # Shade every other 20-window phase
        for i in range(0, length, 40):
            start = i
            end = min(i + 20, length)
            # Use a very light gray for background
            ax.axvspan(start, end, color='#E0E0E0', alpha=0.3, lw=0, zorder=0)
            
        # Add subtle vertical lines at phase boundaries
        for i in range(20, length, 20):
             ax.axvline(x=i, linestyle="--", color='gray', alpha=0.5, linewidth=1, zorder=1)

    # --- Plot F1 Scores ---
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    # Add backgrounds first so they are behind data
    add_phase_backgrounds(ax, max_len)
    
    for mode in ['linc', 'helios', 'original', 'adaptive', 'static']:
        data = results_dict.get(mode)
        if data:
            if mode == 'static':
                f1s = data.get('static_f1s', [])
            else:
                f1s = data.get('my_f1s', [])
            
            if f1s:
                plt.plot(windows[:len(f1s)], f1s, 
                        linewidth=2.5, markersize=7, marker=markers[mode],
                        label=labels[mode], color=colors[mode], markevery=5, alpha=0.9, zorder=z_orders[mode])
    
    plt.xlabel('Window Index')
    plt.ylabel('F1 Score (Macro)')
    plt.title('F1 Score Comparison: All Methods')
    plt.xlim([0, max_len + 2])
    plt.ylim([0.0, 1.05])
    
    # Legend: use best location, slightly transparent background
    plt.legend(loc='best', frameon=True, framealpha=0.5, fancybox=True, shadow=True, ncol=2, fontsize=24)
    
    f1_output = f"{output_dir}/{experiment_name}_f1_comparison.png"
    plt.savefig(f1_output, dpi=300, bbox_inches='tight')
    plt.savefig(f1_output.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"F1 comparison saved to {f1_output}")
    plt.close()
    
    # --- Plot Accuracy ---
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    add_phase_backgrounds(ax, max_len)
    
    for mode in ['linc', 'helios', 'original', 'adaptive', 'static']:
        data = results_dict.get(mode)
        if data:
            if mode == 'static':
                acc = data.get('static_accuracy', [])
            else:
                acc = data.get('my_accuracy', [])
            
            if acc:
                plt.plot(windows[:len(acc)], acc,
                        linewidth=2.5, markersize=7, marker=markers[mode],
                        label=labels[mode], color=colors[mode], markevery=5, alpha=0.9, zorder=z_orders[mode])
    
    plt.xlabel('Window Index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: All Methods')
    plt.xlim([0, max_len + 2])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='best', frameon=True, framealpha=0.5, fancybox=True, shadow=True, ncol=2, fontsize=24)
    
    acc_output = f"{output_dir}/{experiment_name}_accuracy_comparison.png"
    plt.savefig(acc_output, dpi=300, bbox_inches='tight')
    plt.savefig(acc_output.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Accuracy comparison saved to {acc_output}")
    plt.close()

    # --- Plot Combined F1 and Accuracy (Stacked Subplots) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Add backgrounds to both axes
    add_phase_backgrounds(ax1, max_len)
    add_phase_backgrounds(ax2, max_len)
    
    # Plot on separate axes
    for mode in ['linc', 'helios', 'original', 'adaptive', 'static']:
        data = results_dict.get(mode)
        if data:
            if mode == 'static':
                f1s = data.get('static_f1s', [])
                acc = data.get('static_accuracy', [])
            else:
                f1s = data.get('my_f1s', [])
                acc = data.get('my_accuracy', [])
            
            # Plot F1 on ax1 (Top)
            if f1s:
                ax1.plot(windows[:len(f1s)], f1s, 
                        linewidth=2.5, markersize=7, marker=markers[mode],
                        color=colors[mode], markevery=5, alpha=0.9, zorder=z_orders[mode],
                        label=labels[mode])
            
            # Plot Accuracy on ax2 (Bottom)
            if acc:
                ax2.plot(windows[:len(acc)], acc,
                        linewidth=2.5, markersize=7, marker=markers[mode],
                        color=colors[mode], markevery=5, alpha=0.9, zorder=z_orders[mode],
                        label=labels[mode])
    
    ax1.set_ylabel('F1 Score')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Window Index')
    
    ax1.set_ylim(top=1.05)
    ax2.set_ylim(top=1.05)
    ax1.set_xlim([0, max_len + 2])
    ax2.set_xlim([0, max_len + 2])
    
    ax1.set_title('Performance Comparison: All Methods')
    
    # Legend - put on bottom plot (Accuracy)
    ax2.legend(loc='best', frameon=True, framealpha=0.4, fancybox=True, shadow=True, ncol=1, fontsize=28)
    
    # Adjust layout to remove gap
    plt.subplots_adjust(hspace=0.05)
    
    combined_output = f"{output_dir}/{experiment_name}_combined_stacked.png"
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    plt.savefig(combined_output.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Combined stacked comparison saved to {combined_output}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    # Collect stats for bar chart
    stats = {}
    for mode in ['linc', 'helios', 'original', 'adaptive', 'static']:
        data = results_dict.get(mode)
        if data:
            if mode == 'static':
                f1s = data.get('static_f1s', [])
                acc = data.get('static_accuracy', [])
            else:
                f1s = data.get('my_f1s', [])
                acc = data.get('my_accuracy', [])
            
            if f1s:
                # Use data from window 40 onwards (index 40+) for more accurate comparison 
                # as windows 1-40 were used for pre-training/rule extraction
                f1s_subset = f1s[40:] if len(f1s) > 40 else f1s
                acc_subset = acc[40:] if len(acc) > 40 else acc
                
                stats[mode] = {
                    'f1_mean': np.mean(f1s_subset),
                    'f1_std': np.std(f1s_subset),
                    'acc_mean': np.mean(acc_subset) if acc_subset else 0,
                    'acc_std': np.std(acc_subset) if acc_subset else 0,
                }
                print(f"\n{labels[mode]} (from window 41 onwards):")
                print(f"  Avg F1: {stats[mode]['f1_mean']:.4f} ± {stats[mode]['f1_std']:.4f}")
                print(f"  Avg Accuracy: {stats[mode]['acc_mean']:.4f} ± {stats[mode]['acc_std']:.4f}")

    # --- Plot Combined Bar Chart for Summary Statistics ---
    if stats:
        # Create a single figure for both metrics
        fig, ax = plt.subplots(figsize=(14, 6))
        from matplotlib.patches import Patch
        
        methods = []
        method_labels_display = []
        f1_means = []
        f1_stds = []
        acc_means = []
        acc_stds = []
        bar_colors = []
        
        # Order matters here - swap helios and original (Caravan) to avoid legend overlap
        for mode in ['static', 'helios', 'original', 'linc', 'adaptive']:
            if mode in stats:
                methods.append(mode)
                # Wrap labels for x-axis to save horizontal space
                method_labels_display.append(labels[mode].replace(' (', '\n('))
                f1_means.append(stats[mode]['f1_mean'])
                f1_stds.append(stats[mode]['f1_std'])
                acc_means.append(stats[mode]['acc_mean'])
                acc_stds.append(stats[mode]['acc_std'])
                bar_colors.append(colors[mode])
        
        x = np.arange(len(methods))
        width = 0.35  # Width of each bar
        
        # Plot F1 bars (Left)
        bars1 = ax.bar(x - width/2, f1_means, width, yerr=f1_stds, 
                       capsize=3, color=bar_colors, edgecolor='black', linewidth=1.2,
                       error_kw={'elinewidth': 1})
        
        # Plot Accuracy bars (Right) with hatching
        bars2 = ax.bar(x + width/2, acc_means, width, yerr=acc_stds, 
                       capsize=3, color=bar_colors, edgecolor='black', linewidth=1.2,
                       hatch='///', error_kw={'elinewidth': 1})

        ax.set_ylabel('Score')
        ax.set_title('Average Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels_display, rotation=0, ha='center')
        ax.set_ylim([0, 1.15]) 
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add labels to bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.02,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=28, fontweight='bold', rotation=0)

        autolabel(bars1)
        autolabel(bars2)
        
        # Custom legend for F1 vs Accuracy
        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='black', label='F1 Score'),
            Patch(facecolor='lightgray', edgecolor='black', hatch='///', label='Accuracy')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=20, ncol=2)
        
        plt.tight_layout()
        bar_output = f"{output_dir}/{experiment_name}_summary_bar.png"
        plt.savefig(bar_output, dpi=300, bbox_inches='tight')
        plt.savefig(bar_output.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"\nSummary bar chart saved to {bar_output}")
        plt.close()
    
    # Prepare data for distribution plots
    ordered_modes = ['static', 'original', 'helios', 'linc', 'adaptive']
    existing_modes = [m for m in ordered_modes if m in results_dict and results_dict[m]]
    
    plot_data_f1 = []
    plot_data_acc = []
    
    for mode in existing_modes:
        data = results_dict[mode]
        if mode == 'static':
            f1s = data.get('static_f1s', [])
            acc = data.get('static_accuracy', [])
        else:
            f1s = data.get('my_f1s', [])
            acc = data.get('my_accuracy', [])
        
        # Use subset > 40
        f1s_subset = f1s[40:] if len(f1s) > 40 else f1s
        acc_subset = acc[40:] if len(acc) > 40 else acc
        
        plot_data_f1.append(f1s_subset)
        plot_data_acc.append(acc_subset)

    # --- Plot Kernel Density Estimation (KDE) - Artistic "Mountains" ---
    # This often looks very professional and "smooth"
    try:
        from scipy.stats import gaussian_kde
        if stats:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            x_grid = np.linspace(0, 1.0, 500)
            
            def plot_kde(ax, modes, data_list, colors, title, xlabel):
                for mode, data in zip(modes, data_list):
                    if len(data) > 1 and np.std(data) > 0:
                        density = gaussian_kde(data, bw_method='scott')
                        y = density(x_grid)
                        
                        # Clip manually to avoid visual artifacts beyond [0, 1]
                        # Since F1/Acc are strictly [0,1], KDE tails beyond 1 are mathematical artifacts
                        ax.plot(x_grid, y, color=colors[mode], label=labels[mode], linewidth=2)
                        ax.fill_between(x_grid, y, alpha=0.2, color=colors[mode])
                    elif len(data) > 0:
                        # Fallback for constant data (spike)
                        ax.axvline(x=data[0], color=colors[mode], label=labels[mode], 
                                 linewidth=2, linestyle='--')

                ax.set_title(title, fontsize=32)
                ax.set_xlabel(xlabel, fontsize=28)
                ax.set_ylabel('Density', fontsize=28)
                ax.set_xlim([0.4, 1.01])
                ax.legend(loc='best', frameon=True, framealpha=0.5, fontsize=24)
                ax.tick_params(axis='both', labelsize=24)
                ax.grid(True, alpha=0.3)

            plot_kde(axes[0], existing_modes, plot_data_f1, colors, 
                    'F1 Score Density (Distribution Shape)', 'F1 Score')
            
            plot_kde(axes[1], existing_modes, plot_data_acc, colors, 
                    'Accuracy Density (Distribution Shape)', 'Accuracy')
            
            # Only left plot shows y-axis label
            axes[1].set_ylabel('')
            
            plt.tight_layout()
            kde_output = f"{output_dir}/{experiment_name}_summary_kde.png"
            plt.savefig(kde_output, dpi=300, bbox_inches='tight')
            plt.savefig(kde_output.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"\nSummary KDE plot saved to {kde_output}")
            plt.close()
            
    except ImportError:
        print("Skipping KDE plot (scipy not installed)")
    except Exception as e:
        print(f"Skipping KDE plot due to error: {e}")
    
    # Save summary to JSON
    summary_output = f"{output_dir}/{experiment_name}_summary.json"
    with open(summary_output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Summary statistics saved to {summary_output}")


def main():
    parser = argparse.ArgumentParser(description="Compare LINC vs Original vs Adaptive retraining")
    parser.add_argument("--name", type=str, default="linc_comparison", help="Experiment name")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., cic-ids-2018) - overrides feature_file derivation")
    parser.add_argument("--feature_file", type=str, default="cic-ids-2018-redistributed_X.npy", help="Feature file")
    parser.add_argument("--label_file", type=str, default="cic-ids-2018-redistributed_y_binary.npy", help="Label file")
    parser.add_argument("--cp_device", type=str, default="cuda:0", help="Control plane device")
    parser.add_argument("--dp_device", type=str, default="cuda:1", help="Data plane device")
    parser.add_argument("--model_class", type=str, default="MLP1", 
                        choices=["MLP1", "TextCNN1", "RNN1"],
                        help="Model class to use (MLP1, TextCNN1, RNN1)")
    parser.add_argument("--labeler_class", type=str, default=None,
                        help="Labeler class (default: {model_class}Teacher)")
    parser.add_argument("--skip_attacks", type=str, help="Comma-separated list of attack labels to skip/exclude")
    parser.add_argument("--num_classes", type=int, default=None, 
                        help="Number of classes (auto-detected from label file if not provided)")
    parser.add_argument("--skip_linc", action="store_true", help="Skip LINC experiment")
    parser.add_argument("--skip_helios", action="store_true", help="Skip Helios experiment")
    parser.add_argument("--skip_original", action="store_true", help="Skip original retrain experiment")
    parser.add_argument("--skip_adaptive", action="store_true", help="Skip adaptive retrain experiment")
    parser.add_argument("--plot_only", action="store_true", help="Only plot from existing results")
    parser.add_argument("--linc_results", type=str, help="Path to existing LINC results")
    parser.add_argument("--helios_results", type=str, help="Path to existing Helios results")
    parser.add_argument("--original_results", type=str, help="Path to existing original results")
    parser.add_argument("--adaptive_results", type=str, help="Path to existing adaptive results")
    parser.add_argument("--timestamp", type=str, help="Timestamp for plotting existing results (e.g., 20260121-213531)")
    
    # LINC-specific arguments
    parser.add_argument("--linc_rule_threshold", type=float, default=0.3, help="LINC rule confidence threshold")
    parser.add_argument("--linc_max_rules", type=int, default=1000, help="LINC maximum number of rules")
    parser.add_argument("--linc_update_samples", type=int, default=4000, help="LINC samples for incremental update")

    # Helios-specific arguments
    parser.add_argument("--helios_max_rules", type=int, default=3000, help="Helios max rules")
    parser.add_argument("--helios_radio", type=float, default=1.5, help="Helios radio parameter")
    parser.add_argument("--helios_boost_num", type=int, default=6, help="Helios boosting iterations")
    parser.add_argument("--helios_prune_rule", type=int, default=3, help="Helios prune rule threshold")
    
    # Reproducibility arguments
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA behavior (may reduce performance)")

    args = parser.parse_args()
    
    if args.dataset:
        dataset_name = args.dataset
    else:
        # Extract dataset name from feature_file
        dataset_basename = os.path.basename(args.feature_file)
        dataset_name = dataset_basename.split('.')[0].replace('_X', '').replace('_redist', '').replace('-redistributed', '')
    
    # Normalize common dataset names to match checkpoint files
    if '2018' in dataset_name:
        dataset_name = 'cic-ids-2018'
    elif 'iscxvpn' in dataset_name.lower():
        dataset_name = 'iscxvpn'
    elif 'ustc-tfc2016' in dataset_name.lower():
        dataset_name = 'ustc-tfc2016'
    
    args.dataset_name = dataset_name
    print(f"Extracted dataset name: {args.dataset_name}")

    # Determine number of classes
    if args.num_classes is not None:
        # User explicitly specified num_classes
        print(f"Using user-specified number of classes: {args.num_classes}")
    else:
        # Try to auto-detect from label file
        label_path = os.path.join(f"{Shawarma_home}/emulation/datasets", args.label_file)
        if os.path.exists(label_path):
            try:
                labels = np.load(label_path)
                args.num_classes = len(np.unique(labels))
                print(f"Auto-detected {args.num_classes} classes from label file: {args.label_file}")
            except Exception as e:
                print(f"Warning: Could not read label file for class detection: {e}")
                # Fallback to dataset map
                dataset_class_map = {
                    'cic-ids-2018': 7,
                    'iscxvpn': 7,
                    'ustc-tfc2016': 12,
                }
                args.num_classes = dataset_class_map.get(args.dataset_name, 7)
                print(f"Falling back to default: {args.num_classes} classes for {args.dataset_name}")
        else:
            # Fallback to dataset map
            dataset_class_map = {
                'cic-ids-2018': 7,
                'iscxvpn': 7,
                'ustc-tfc2016': 12,
            }
            args.num_classes = dataset_class_map.get(args.dataset_name, 7)
            print(f"Label file not found, using default: {args.num_classes} classes for {args.dataset_name}")
    
    # Set default labeler class based on model class
    if args.labeler_class is None:
        labeler_class_map = {
            "MLP1": "MLP1Teacher",
            "TextCNN1": "TextCNNTeacher",
            "RNN1": "RNNTeacher",
        }
        args.labeler_class = labeler_class_map.get(args.model_class, f"{args.model_class}Teacher")
    
    # Determine model checkpoint directory based on model class
    model_dir_map = {
        "MLP1": "mlp",
        "TextCNN1": "cnn",
        "RNN1": "rnn",
    }
    args.model_dir = model_dir_map.get(args.model_class, "mlp")
    
    # Update LINC defaults with command line arguments
    LINC_DEFAULTS['rule_threshold'] = args.linc_rule_threshold
    LINC_DEFAULTS['max_rules'] = args.linc_max_rules
    LINC_DEFAULTS['update_samples'] = args.linc_update_samples
    
    # Update Helios defaults
    HELIOS_DEFAULTS['max_rules'] = args.helios_max_rules
    HELIOS_DEFAULTS['radio'] = args.helios_radio
    HELIOS_DEFAULTS['boost_num'] = args.helios_boost_num
    HELIOS_DEFAULTS['prune_rule'] = args.helios_prune_rule

    # Paths
    results_dir = f"{Shawarma_home}/emulation/scripts/results"
    figures_dir = f"{Shawarma_home}/emulation/scripts/figures"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Clean environment
    clean_env = os.environ.copy()
    for key in list(clean_env.keys()):
        if key.startswith("SHAWARMA_CP_") or key.startswith("SHAWARMA_DP_"):
            del clean_env[key]
    
    results_dict = {}
    
    if args.plot_only:
        # If timestamp is provided, try to auto-populate result paths if not explicitly provided
        if args.timestamp:
            # Use the provided timestamp for output figures as well to match result files
            timestamp = args.timestamp
            
            base_pattern = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}"
            print(f"Auto-resolving result files with timestamp: {args.timestamp}")
            
            if not args.linc_results:
                args.linc_results = f"{base_pattern}_linc_{args.timestamp}.jsonl"
            if not args.helios_results:
                args.helios_results = f"{base_pattern}_helios_{args.timestamp}.jsonl"
            if not args.original_results:
                args.original_results = f"{base_pattern}_original_{args.timestamp}.jsonl"
            if not args.adaptive_results:
                args.adaptive_results = f"{base_pattern}_adaptive_{args.timestamp}.jsonl"

        # Load existing results
        if args.linc_results:
            results_dict['linc'] = load_results(args.linc_results)
        if args.helios_results:
            results_dict['helios'] = load_results(args.helios_results)
        if args.original_results:
            results_dict['original'] = load_results(args.original_results)
        if args.adaptive_results:
            results_dict['adaptive'] = load_results(args.adaptive_results)
        # Static results always come from original experiment for consistency
        if results_dict.get('original') and 'static_f1s' in results_dict['original']:
            results_dict['static'] = results_dict['original']
        else:
            # Fallback if original was skipped
            for mode in ['adaptive', 'linc', 'helios']:
                if results_dict.get(mode) and 'static_f1s' in results_dict[mode]:
                    results_dict['static'] = results_dict[mode]
                    print(f"Warning: Using static data from '{mode}' (original was skipped)")
                    break
    else:
        # Run experiments
        
        # Run LINC experiment
        if not args.skip_linc:
            linc_results_file = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_linc_{timestamp}.jsonl"
            run_single_experiment(args, 'linc', linc_results_file, clean_env)
            results_dict['linc'] = load_results(linc_results_file)
            time.sleep(5)

        # Run Helios experiment
        if not args.skip_helios:
            helios_results_file = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_helios_{timestamp}.jsonl"
            run_single_experiment(args, 'helios', helios_results_file, clean_env)
            results_dict['helios'] = load_results(helios_results_file)
            time.sleep(5)
        
        # Run original retrain experiment
        if not args.skip_original:
            original_results_file = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_original_{timestamp}.jsonl"
            run_single_experiment(args, 'original', original_results_file, clean_env)
            results_dict['original'] = load_results(original_results_file)
            time.sleep(5)
        
        # Run adaptive retrain experiment
        if not args.skip_adaptive:
            adaptive_results_file = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_adaptive_{timestamp}.jsonl"
            run_single_experiment(args, 'adaptive', adaptive_results_file, clean_env)
            results_dict['adaptive'] = load_results(adaptive_results_file)
        
        # Static results always come from original experiment for consistency
        if results_dict.get('original') and 'static_f1s' in results_dict['original']:
            results_dict['static'] = results_dict['original']
        else:
            # Fallback if original was skipped
            for mode in ['adaptive', 'linc', 'helios']:
                if results_dict.get(mode) and 'static_f1s' in results_dict[mode]:
                    results_dict['static'] = results_dict[mode]
                    print(f"Warning: Using static data from '{mode}' (original was skipped)")
                    break
    
    # Plot comparison
    if any(results_dict.values()):
        plot_comparison(results_dict, figures_dir, f"{args.name}_{args.dataset_name}_{args.model_class}_{timestamp}")
    else:
        print("No results to plot!")
    
    print(f"\nExperiment completed. Results in {figures_dir}")


if __name__ == "__main__":
    main()
