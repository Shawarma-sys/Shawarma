#!/usr/bin/env python3
# *************************************************************************
#
# Comparison experiment for original_retrain vs adaptive_retrain vs static
# Supports: MLP1, TextCNN1, RNN1
#
# *************************************************************************

import os
import sys
import argparse
import glob
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
    
    retrain_mode: 'original', 'adaptive', or 'static'
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {retrain_mode} retraining ({args.model_class})")
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

    if args.seed is not None:
        cmd_cp.extend(["--seed", str(args.seed)])
    if args.deterministic:
        cmd_cp.append("--deterministic")
    
    # Add sequence model specific arguments (TextCNN1, RNN1, etc.)
    append_sequence_model_args(cmd_cp, args.model_class, num_classes=args.num_classes)
    
    # Add adaptive_freeze flag only for adaptive mode
    if retrain_mode == 'adaptive':
        cmd_cp.append("--adaptive_freeze")
    # For 'static' mode, we don't start control plane retraining at all
    # but we still need the server running for data plane communication
    
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
    
    cmd_dp = [
        sys.executable,
        f"{Shawarma_home}/emulation/scripts/experiments/data_plane_torchlens_experiment.py",
        "--feature_file", args.feature_file,
        "--label_file", args.label_file,
        "--device", args.dp_device,
    ]

    if args.seed is not None:
        cmd_dp.extend(["--seed", str(args.seed)])
    if args.deterministic:
        cmd_dp.append("--deterministic")
    
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
    """Load experiment results from JSONL file."""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        for line in f:
            return json.loads(line)
    return None


def plot_comparison(results_dict, output_dir, experiment_name):
    """Plot comparison of accuracy and F1 scores (Academic Style)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    
    # --- Style Configuration ---
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
    
    # Improved Color Palette
    colors = {
        'original': '#1F77B4',       # Blue
        'adaptive': '#D62728',       # Red
        'static': '#999999',         # Gray
    }
    
    # Define z-orders
    z_orders = {
        'original': 3,
        'adaptive': 5,           # Topmost
        'static': 2              # Bottom
    }
    
    markers = {
        'original': 'o',
        'adaptive': 's',
        'static': 'X',
    }
    
    labels = {
        'original': 'Full Retrain',
        'adaptive': 'Adaptive Retrain',
        'static': 'Static',
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine max length
    max_len = 0
    for mode, data in results_dict.items():
        if data:
            if mode == 'static':
                max_len = max(max_len, len(data.get('static_f1s', [])))
            else:
                max_len = max(max_len, len(data.get('my_f1s', [])))
    
    if max_len == 0:
        print("No data to plot")
        return
    
    windows = list(range(1, max_len + 1))
    
    # Helper to add phase backgrounds
    def add_phase_backgrounds(ax, length):
        for i in range(0, length, 40):
            start = i
            end = min(i + 20, length)
            ax.axvspan(start, end, color='#E0E0E0', alpha=0.3, lw=0, zorder=0)
        for i in range(20, length, 20):
             ax.axvline(x=i, linestyle="--", color='gray', alpha=0.5, linewidth=1, zorder=1)

    # --- Plot F1 Scores ---
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    add_phase_backgrounds(ax, max_len)
    
    for mode in ['original', 'adaptive', 'static']:
        data = results_dict.get(mode)
        if data:
            if mode == 'static':
                # Use static_f1s from whatever experiment provided the static result
                f1s = data.get('static_f1s', [])
            else:
                f1s = data.get('my_f1s', [])
            
            if f1s:
                plt.plot(windows[:len(f1s)], f1s, 
                        linewidth=2.5, markersize=7, marker=markers[mode],
                        label=labels[mode], color=colors[mode], markevery=5, alpha=0.9, zorder=z_orders[mode])
    
    plt.xlabel('Window Index')
    plt.ylabel('F1 Score (Macro)')
    plt.title('F1 Score Comparison: Original vs Adaptive vs Static')
    plt.xlim([0, max_len + 2])
    plt.ylim(top=1.05)
    plt.legend(loc='best', frameon=True, framealpha=0.5, fancybox=True, shadow=True)
    
    f1_output = f"{output_dir}/{experiment_name}_f1_comparison.png"
    plt.savefig(f1_output, dpi=300, bbox_inches='tight')
    plt.savefig(f1_output.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"F1 comparison saved to {f1_output}")
    plt.close()
    
    # --- Plot Accuracy ---
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    add_phase_backgrounds(ax, max_len)
    
    for mode in ['original', 'adaptive', 'static']:
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
    plt.title('Accuracy Comparison: Original vs Adaptive vs Static')
    plt.xlim([0, max_len + 2])
    plt.ylim(top=1.05)
    plt.legend(loc='best', frameon=True, framealpha=0.5, fancybox=True, shadow=True)
    
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
    for mode in ['original', 'adaptive', 'static']:
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
    
    ax1.set_title('Performance Comparison')
    
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
    for mode in ['original', 'adaptive', 'static']:
        data = results_dict.get(mode)
        if data:
            if mode == 'static':
                f1s = data.get('static_f1s', [])
                acc = data.get('static_accuracy', [])
            else:
                f1s = data.get('my_f1s', [])
                acc = data.get('my_accuracy', [])
            
            if f1s:
                # Use data from window 40 onwards
                start_idx = 40 if len(f1s) > 40 else 0
                f1s_subset = f1s[start_idx:]
                acc_subset = acc[start_idx:] if acc else []
                
                stats[mode] = {
                    'f1_mean': np.mean(f1s_subset),
                    'f1_std': np.std(f1s_subset),
                    'acc_mean': np.mean(acc_subset) if acc_subset else 0,
                    'acc_std': np.std(acc_subset) if acc_subset else 0,
                }
                print(f"\n{labels[mode]} (from window {start_idx+1} onwards):")
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
        
        for mode in ['static', 'original', 'adaptive']:
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
        ax.legend(handles=legend_elements, loc='upper left', fontsize=28, ncol=1)
        
        plt.tight_layout()
        bar_output = f"{output_dir}/{experiment_name}_summary_bar.png"
        plt.savefig(bar_output, dpi=300, bbox_inches='tight')
        plt.savefig(bar_output.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"\nSummary bar chart saved to {bar_output}")
        plt.close()

    # --- Plot Kernel Density Estimation (KDE) ---
    try:
        from scipy.stats import gaussian_kde
        if stats:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            x_grid = np.linspace(0, 1.0, 500)
            
            # Helper for KDE
            def plot_kde(ax, modes, key_f1, key_acc, type_key, title, xlabel):
                for mode in modes:
                    if mode not in results_dict or not results_dict[mode]:
                        continue
                        
                    data_src = results_dict.get(mode)
                    if type_key == 'f1':
                        if mode == 'static':
                            data = data_src.get('static_f1s', [])
                        else:
                            data = data_src.get('my_f1s', [])
                    else:
                        if mode == 'static':
                            data = data_src.get('static_accuracy', [])
                        else:
                            data = data_src.get('my_accuracy', [])

                    # Use subset
                    start = 40 if len(data) > 40 else 0
                    data = data[start:]
                    
                    if len(data) > 1 and np.std(data) > 0:
                        density = gaussian_kde(data, bw_method='scott')
                        y = density(x_grid)
                        ax.plot(x_grid, y, color=colors[mode], label=labels[mode], linewidth=2)
                        ax.fill_between(x_grid, y, alpha=0.2, color=colors[mode])
                    elif len(data) > 0:
                        ax.axvline(x=data[0], color=colors[mode], label=labels[mode], linewidth=2, linestyle='--')

                ax.set_title(title, fontsize=32)
                ax.set_xlabel(xlabel, fontsize=28)
                ax.set_ylabel('Density', fontsize=28)
                ax.set_xlim([0.4, 1.01])
                ax.legend(loc='best', frameon=True, framealpha=0.5, fontsize=24)
                ax.tick_params(axis='both', labelsize=24)
                ax.grid(True, alpha=0.3)

            modes_to_plot = ['static', 'original', 'adaptive']
            plot_kde(axes[0], modes_to_plot, 'f1', 'acc', 'f1', 'F1 Score Density', 'F1 Score')
            plot_kde(axes[1], modes_to_plot, 'f1', 'acc', 'acc', 'Accuracy Density', 'Accuracy')
            
            # Only left plot shows y-axis label
            axes[1].set_ylabel('')
            
            plt.tight_layout()
            kde_output = f"{output_dir}/{experiment_name}_summary_kde.png"
            plt.savefig(kde_output, dpi=300, bbox_inches='tight')
            plt.savefig(kde_output.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"\nSummary KDE plot saved to {kde_output}")
            plt.close()
            
    except ImportError:
        pass
    except Exception as e:
        print(f"Skipping KDE plot: {e}")

    # Save summary to JSON
    summary_output = f"{output_dir}/{experiment_name}_summary.json"
    with open(summary_output, 'w') as f:
        # Convert numpy types to native types
        stats_native = {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in stats.items()}
        json.dump(stats_native, f, indent=2)
    print(f"Summary statistics saved to {summary_output}")


def main():
    parser = argparse.ArgumentParser(description="Compare Original vs Adaptive vs Static retraining")
    parser.add_argument("--name", type=str, default="retrain_comparison", help="Experiment name")
    parser.add_argument("--feature_file", type=str, default="experiment_2018_0.6_X.npy", help="Feature file")
    parser.add_argument("--label_file", type=str, default="experiment_2018_0.6_y.npy", help="Label file")
    parser.add_argument("--cp_device", type=str, default="cuda:0", help="Control plane device")
    parser.add_argument("--dp_device", type=str, default="cuda:1", help="Data plane device")
    parser.add_argument("--model_class", type=str, default="MLP1", 
                        choices=["MLP1", "TextCNN1", "TextCNN2", "RNN1"],
                        help="Model class to use (MLP1, TextCNN1, TextCNN2, RNN1)")
    parser.add_argument("--labeler_class", type=str, default=None,
                        help="Labeler class (default: {model_class}Teacher)")
    parser.add_argument("--skip_attacks", type=str, help="Comma-separated list of attack labels to skip/exclude")
    parser.add_argument("--num_classes", type=int, default=None, 
                        help="Number of classes (auto-detected from label file if not provided)")
    parser.add_argument("--skip_original", action="store_true", help="Skip original retrain experiment")
    parser.add_argument("--skip_adaptive", action="store_true", help="Skip adaptive retrain experiment")
    parser.add_argument("--plot_only", action="store_true", help="Only plot from existing results")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., cic-ids-2018) - overrides feature_file derivation")
    parser.add_argument("--original_results", type=str, help="Path to existing original results")
    parser.add_argument("--adaptive_results", type=str, help="Path to existing adaptive results")
    parser.add_argument("--timestamp", type=str, help="Timestamp for plotting existing results")
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
            "TextCNN2": "TextCNNTeacher",
            "RNN1": "RNNTeacher",
        }
        args.labeler_class = labeler_class_map.get(args.model_class, f"{args.model_class}Teacher")
    
    # Determine model checkpoint directory based on model class
    model_dir_map = {
        "MLP1": "mlp",
        "TextCNN2": "cnn",
        "TextCNN1": "cnn",
        "RNN1": "rnn",
    }
    args.model_dir = model_dir_map.get(args.model_class, "mlp")
    
    # Paths
    results_dir = f"{Shawarma_home}/emulation/scripts/results"
    figures_dir = f"{Shawarma_home}/emulation/scripts/figures"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if args.plot_only and args.timestamp and not (args.original_results and args.adaptive_results):
        print(f"Attempting to auto-resolve result files for timestamp: {args.timestamp}")
        
        # We need to construct the likely filename
        base_pattern = f"{args.name}_{args.dataset_name}_{args.model_class}"
        
        if not args.original_results:
            candidate = os.path.join(results_dir, f"{base_pattern}_original_{args.timestamp}.jsonl")
            if os.path.exists(candidate):
                args.original_results = candidate
                print(f"Found original results: {candidate}")
            else:
                # Try globbing just in case name is different
                candidates = glob.glob(os.path.join(results_dir, f"*{args.model_class}*original*{args.timestamp}*.jsonl"))
                # Filter by dataset if possible
                candidates = [c for c in candidates if args.dataset_name in c]
                if candidates:
                    args.original_results = candidates[0]
                    print(f"Found original results (glob): {candidates[0]}")
        
        if not args.adaptive_results:
            candidate = os.path.join(results_dir, f"{base_pattern}_adaptive_{args.timestamp}.jsonl")
            if os.path.exists(candidate):
                args.adaptive_results = candidate
                print(f"Found adaptive results: {candidate}")
            else:
                 # Try globbing
                candidates = glob.glob(os.path.join(results_dir, f"*{args.model_class}*adaptive*{args.timestamp}*.jsonl"))
                candidates = [c for c in candidates if args.dataset_name in c]
                if candidates:
                    args.adaptive_results = candidates[0]
                    print(f"Found adaptive results (glob): {candidates[0]}")

    timestamp = args.timestamp if args.timestamp else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Clean environment
    clean_env = os.environ.copy()
    for key in list(clean_env.keys()):
        if key.startswith("SHAWARMA_CP_") or key.startswith("SHAWARMA_DP_"):
            del clean_env[key]
    
    results_dict = {}
    
    if args.plot_only:
        # Load existing results
        if args.original_results:
            results_dict['original'] = load_results(args.original_results)
        if args.adaptive_results:
            results_dict['adaptive'] = load_results(args.adaptive_results)
        # Static results come from either experiment
        if results_dict.get('original'):
            results_dict['static'] = results_dict['original']
        elif results_dict.get('adaptive'):
            results_dict['static'] = results_dict['adaptive']
    else:
        # Run experiments
        if not args.skip_original:
            original_results_file = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_original_{timestamp}.jsonl"
            run_single_experiment(args, 'original', original_results_file, clean_env)
            results_dict['original'] = load_results(original_results_file)
            time.sleep(5)
        
        if not args.skip_adaptive:
            adaptive_results_file = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_adaptive_{timestamp}.jsonl"
            run_single_experiment(args, 'adaptive', adaptive_results_file, clean_env)
            results_dict['adaptive'] = load_results(adaptive_results_file)
        
        # Static results come from either experiment (they're the same)
        if results_dict.get('original'):
            results_dict['static'] = results_dict['original']
        elif results_dict.get('adaptive'):
            results_dict['static'] = results_dict['adaptive']
    
    # Plot comparison
    if any(results_dict.values()):
        plot_comparison(results_dict, figures_dir, f"{args.name}_{args.dataset_name}_{args.model_class}_{timestamp}")
    else:
        print("No results to plot!")
    
    print(f"\nExperiment completed. Results in {figures_dir}")


if __name__ == "__main__":
    main()
