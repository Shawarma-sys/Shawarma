#!/usr/bin/env python3
# *************************************************************************
#
# Ablation experiment: Memory (0.1) vs No Memory (0.0)
# Across both Full Retrain and Adaptive Retrain modes
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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


def run_single_experiment(args, exp_label, memory_ratio, retrain_mode, results_file, clean_env):
    """
    Run a single experiment with specified parameters.
    retrain_mode: 'adaptive' or 'original'
    memory_ratio: float (e.g. 0.0 or 0.1)
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_label}")
    print(f"Mode: {retrain_mode}, Memory Ratio: {memory_ratio}")
    print(f"{'='*60}\n")
    
    kill_process_on_port(50051)
    
    # Build control plane command
    cmd_cp = [
        sys.executable,
        f"{Shawarma_home}/emulation/benchmarks/control_plane_retrain_torchlens.py",
        "--device", args.cp_device,
        "--application_type", "intrusion_detection",
        "--job_name", f"cp_{exp_label}_{args.dataset_name}_{args.model_class}",
        "--labeler_type", "DNN_classifier",
        "--labeler_dnn_class", args.labeler_class,
        "--labeler_dnn_path", f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/best_{args.labeler_class}_{args.dataset_name}_mixed.pth",
        "-r", "0.005",
        "--total_epochs", "60",
        "--model_class", args.model_class,
        "-m", f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/best_{args.model_class}_{args.dataset_name}_mixed.pth",
        "-o", results_file,
        "--logdir", f"{Shawarma_home}/emulation/logs/control_plane",
        "--test_size", "100",
        "--memory_size", "5000",
        "--label_diff_threshold", "0.05",
        "--memory_sample_ratio", str(memory_ratio),
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
    
    # Start control plane
    cp_process = subprocess.Popen(cmd_cp, env=clean_env, start_new_session=True)
    time.sleep(10)
    
    if cp_process.poll() is not None:
        print(f"Error: Control Plane failed to start (exit code {cp_process.returncode})")
        return None
    
    # Build data plane command
    dp_env = clean_env.copy()
    dp_env["SHAWARMA_DP_MODEL_CLASS"] = args.model_class
    dp_env["SHAWARMA_DP_BASE_MODEL_PATH"] = f"{Shawarma_home}/emulation/models/checkpoint/{args.model_dir}/best_{args.model_class}_{args.dataset_name}_mixed.pth"
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
        run_command(cmd_dp, f"Data Plane ({exp_label})", env=dp_env)
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
    """Plot comparison of Memory vs No Memory across Retraining Modes."""
    
    # --- Style Configuration ---
    available_styles = plt.style.available
    if 'seaborn-v0_8-paper' in available_styles:
        plt.style.use('seaborn-v0_8-paper')
    elif 'ggplot' in available_styles:
        plt.style.use('ggplot')
    else:
        plt.style.use('default')

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

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
    
    # Define keys to plot
    plot_keys = ['adaptive_mem', 'adaptive_no_mem', 'original_mem', 'original_no_mem', 'static']
    
    # Colors
    colors = {
        'adaptive_mem': '#D62728',     # Red
        'adaptive_no_mem': '#FF7F7F',  # Light Red
        'original_mem': '#1F77B4',     # Blue
        'original_no_mem': '#7FBAE5',  # Light Blue
        'static': '#999999',           # Gray
    }
    
    z_orders = {
        'adaptive_mem': 5,
        'adaptive_no_mem': 4,
        'original_mem': 3,
        'original_no_mem': 3,
        'static': 2
    }
    
    markers = {
        'adaptive_mem': 's',
        'adaptive_no_mem': 's',
        'original_mem': 'o',
        'original_no_mem': 'o',
        'static': 'X',
    }
    
    linestyles = {
        'adaptive_mem': '-',
        'adaptive_no_mem': '--',
        'original_mem': '-',
        'original_no_mem': '--',
        'static': '-',
    }
    
    labels = {
        'adaptive_mem': 'Adaptive (w/)',
        'adaptive_no_mem': 'Adaptive (w/o)',
        'original_mem': 'Full (w/)',
        'original_no_mem': 'Full (w/o)',
        'static': 'Static',
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine max length
    max_len = 0
    for key in plot_keys:
        data = results_dict.get(key)
        if data:
            if key == 'static':
                max_len = max(max_len, len(data.get('static_f1s', [])))
            else:
                max_len = max(max_len, len(data.get('my_f1s', [])))
    
    if max_len == 0:
        print("No data to plot")
        return
    
    windows = list(range(1, max_len + 1))
    
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
    
    for key in plot_keys:
        data = results_dict.get(key)
        if data:
            if key == 'static':
                f1s = data.get('static_f1s', [])
            else:
                f1s = data.get('my_f1s', [])
            
            if f1s:
                plt.plot(windows[:len(f1s)], f1s, 
                        linewidth=2.5, markersize=7, marker=markers.get(key, 'o'),
                        linestyle=linestyles.get(key, '-'),
                        label=labels.get(key, key), color=colors.get(key, 'black'), 
                        markevery=5, alpha=0.9, zorder=z_orders.get(key, 3))
    
    plt.xlabel('Window Index')
    plt.ylabel('F1 Score (Macro)')
    plt.title('Impact of Memory on Adaptive vs Full Retraining (F1)')
    plt.xlim([0, max_len + 2])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True, shadow=True, ncol=2, fontsize=24)
    
    f1_output = f"{output_dir}/{experiment_name}_f1_comparison.png"
    plt.savefig(f1_output, dpi=300, bbox_inches='tight')
    plt.savefig(f1_output.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"F1 comparison saved to {f1_output}")
    plt.close()
    
    # --- Plot Accuracy ---
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    add_phase_backgrounds(ax, max_len)
    
    for key in plot_keys:
        data = results_dict.get(key)
        if data:
            if key == 'static':
                acc = data.get('static_accuracy', [])
            else:
                acc = data.get('my_accuracy', [])
            
            if acc:
                plt.plot(windows[:len(acc)], acc,
                        linewidth=2.5, markersize=7, marker=markers.get(key, 'o'),
                        linestyle=linestyles.get(key, '-'),
                        label=labels.get(key, key), color=colors.get(key, 'black'),
                        markevery=5, alpha=0.9, zorder=z_orders.get(key, 3))
    
    plt.xlabel('Window Index')
    plt.ylabel('Accuracy')
    plt.title('Impact of Memory on Adaptive vs Full Retraining (Accuracy)')
    plt.xlim([0, max_len + 2])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True, shadow=True, ncol=2, fontsize=24)
    
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
    for key in plot_keys:
        data = results_dict.get(key)
        if data:
            if key == 'static':
                f1s = data.get('static_f1s', [])
                acc = data.get('static_accuracy', [])
            else:
                f1s = data.get('my_f1s', [])
                acc = data.get('my_accuracy', [])
            
            # Plot F1 on ax1 (Top)
            if f1s:
                ax1.plot(windows[:len(f1s)], f1s, 
                        linewidth=2.5, markersize=7, marker=markers[key],
                        color=colors[key], markevery=5, alpha=0.9, zorder=z_orders[key],
                        linestyle=linestyles[key], label=labels[key])
            
            # Plot Accuracy on ax2 (Bottom)
            if acc:
                ax2.plot(windows[:len(acc)], acc,
                        linewidth=2.5, markersize=7, marker=markers[key],
                        color=colors[key], markevery=5, alpha=0.9, zorder=z_orders[key],
                        linestyle=linestyles[key], label=labels[key])
    
    ax1.set_ylabel('F1 Score')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Window Index')
    
    ax1.set_ylim(top=1.05)
    ax2.set_ylim(top=1.05)
    ax1.set_xlim([0, max_len + 2])
    ax2.set_xlim([0, max_len + 2])
    
    ax1.set_title('Impact of Memory on Adaptive vs Full Retraining')
    
    # Legend - put on bottom plot (Accuracy)
    ax2.legend(loc='best', frameon=True, framealpha=0.4, fancybox=True, shadow=True, ncol=1, fontsize=28)
    
    # Adjust layout to remove gap
    plt.subplots_adjust(hspace=0.05)
    
    combined_output = f"{output_dir}/{experiment_name}_combined_stacked.png"
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    plt.savefig(combined_output.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Combined stacked comparison saved to {combined_output}")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    stats = {}
    for key in plot_keys:
        data = results_dict.get(key)
        if data:
            if key == 'static':
                f1s = data.get('static_f1s', [])
                acc = data.get('static_accuracy', [])
            else:
                f1s = data.get('my_f1s', [])
                acc = data.get('my_accuracy', [])
            
            if f1s:
                start_idx = 40 if len(f1s) > 40 else 0
                f1s_subset = f1s[start_idx:]
                acc_subset = acc[start_idx:] if acc else []
                
                stats[key] = {
                    'f1_mean': np.mean(f1s_subset),
                    'f1_std': np.std(f1s_subset),
                    'acc_mean': np.mean(acc_subset) if acc_subset else 0,
                    'acc_std': np.std(acc_subset) if acc_subset else 0,
                }
                print(f"\n{labels.get(key, key)} (from window {start_idx+1} onwards):")
                print(f"  Avg F1: {stats[key]['f1_mean']:.4f} ± {stats[key]['f1_std']:.4f}")
                print(f"  Avg Accuracy: {stats[key]['acc_mean']:.4f} ± {stats[key]['acc_std']:.4f}")

    # Bar chart
    if stats:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        methods = []
        method_labels_display = []
        f1_means = []
        f1_stds = []
        acc_means = []
        acc_stds = []
        bar_colors = []
        
        # Display order
        display_keys = ['static', 'original_no_mem', 'original_mem', 'adaptive_no_mem', 'adaptive_mem']
        
        for key in display_keys:
            if key in stats:
                methods.append(key)
                method_labels_display.append(labels.get(key, key).replace(' (', '\n('))
                f1_means.append(stats[key]['f1_mean'])
                f1_stds.append(stats[key]['f1_std'])
                acc_means.append(stats[key]['acc_mean'])
                acc_stds.append(stats[key]['acc_std'])
                bar_colors.append(colors.get(key, 'gray'))
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, f1_means, width, yerr=f1_stds, 
                       capsize=3, color=bar_colors, edgecolor='black', linewidth=1.2,
                       error_kw={'elinewidth': 1})
        
        bars2 = ax.bar(x + width/2, acc_means, width, yerr=acc_stds, 
                       capsize=3, color=bar_colors, edgecolor='black', linewidth=1.2,
                       hatch='///', error_kw={'elinewidth': 1})

        ax.set_ylabel('Score')
        ax.set_title('Average Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels_display, rotation=0, ha='center')
        ax.set_ylim([0, 1.15]) 
        ax.grid(True, axis='y', alpha=0.3)
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.02,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=28, fontweight='bold', rotation=0)

        autolabel(bars1)
        autolabel(bars2)
        
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

    # Save summary
    summary_output = f"{output_dir}/{experiment_name}_summary.json"
    with open(summary_output, 'w') as f:
        stats_native = {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in stats.items()}
        json.dump(stats_native, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Ablation: Memory vs No Memory across Retrain Modes")
    parser.add_argument("--name", type=str, default="ablation_memory_full", help="Experiment name")
    parser.add_argument("--feature_file", type=str, default="experiment_2018_0.6_X.npy", help="Feature file")
    parser.add_argument("--label_file", type=str, default="experiment_2018_0.6_y.npy", help="Label file")
    parser.add_argument("--cp_device", type=str, default="cuda:0", help="Control plane device")
    parser.add_argument("--dp_device", type=str, default="cuda:1", help="Data plane device")
    parser.add_argument("--model_class", type=str, default="MLP1", 
                        choices=["MLP1", "TextCNN1", "TextCNN2", "RNN1"],
                        help="Model class")
    parser.add_argument("--labeler_class", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    
    # Flags to skip specific experiments
    parser.add_argument("--skip_adaptive_mem", action="store_true", help="Skip Adaptive + Memory")
    parser.add_argument("--skip_adaptive_no_mem", action="store_true", help="Skip Adaptive + No Memory")
    parser.add_argument("--skip_original_mem", action="store_true", help="Skip Full Retrain + Memory")
    parser.add_argument("--skip_original_no_mem", action="store_true", help="Skip Full Retrain + No Memory")
    
    parser.add_argument("--plot_only", action="store_true", help="Only plot")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    
    # Paths for existing results (if plot_only)
    parser.add_argument("--adaptive_mem_results", type=str)
    parser.add_argument("--adaptive_no_mem_results", type=str)
    parser.add_argument("--original_mem_results", type=str)
    parser.add_argument("--original_no_mem_results", type=str)
    
    parser.add_argument("--timestamp", type=str, help="Timestamp for plotting")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_basename = os.path.basename(args.feature_file)
        dataset_name = dataset_basename.split('.')[0].replace('_X', '').replace('_redist', '').replace('-redistributed', '')
    
    if '2018' in dataset_name:
        dataset_name = 'cic-ids-2018'
    elif 'iscxvpn' in dataset_name.lower():
        dataset_name = 'iscxvpn'
    elif 'ustc-tfc2016' in dataset_name.lower():
        dataset_name = 'ustc-tfc2016'
    
    args.dataset_name = dataset_name
    print(f"Extracted dataset name: {args.dataset_name}")

    if args.num_classes is None:
        label_path = os.path.join(f"{Shawarma_home}/emulation/datasets", args.label_file)
        if os.path.exists(label_path):
            try:
                labels = np.load(label_path)
                args.num_classes = len(np.unique(labels))
                print(f"Auto-detected {args.num_classes} classes")
            except:
                pass
        
        if args.num_classes is None:
            dataset_class_map = {
                'cic-ids-2018': 7,
                'iscxvpn': 7,
                'ustc-tfc2016': 12,
            }
            args.num_classes = dataset_class_map.get(args.dataset_name, 7)
            print(f"Using default: {args.num_classes} classes")
    
    if args.labeler_class is None:
        labeler_class_map = {
            "MLP1": "MLP1Teacher",
            "TextCNN1": "TextCNNTeacher",
            "TextCNN2": "TextCNNTeacher",
            "RNN1": "RNNTeacher",
        }
        args.labeler_class = labeler_class_map.get(args.model_class, f"{args.model_class}Teacher")
    
    model_dir_map = {
        "MLP1": "mlp",
        "TextCNN2": "cnn",
        "TextCNN1": "cnn",
        "RNN1": "rnn",
    }
    args.model_dir = model_dir_map.get(args.model_class, "mlp")
    
    results_dir = f"{Shawarma_home}/emulation/scripts/results"
    figures_dir = f"{Shawarma_home}/emulation/scripts/figures"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    timestamp = args.timestamp if args.timestamp else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    clean_env = os.environ.copy()
    for key in list(clean_env.keys()):
        if key.startswith("SHAWARMA_CP_") or key.startswith("SHAWARMA_DP_"):
            del clean_env[key]
    
    results_dict = {}
    
    if args.plot_only:
        if args.adaptive_mem_results:
            results_dict['adaptive_mem'] = load_results(args.adaptive_mem_results)
        if args.adaptive_no_mem_results:
            results_dict['adaptive_no_mem'] = load_results(args.adaptive_no_mem_results)
        if args.original_mem_results:
            results_dict['original_mem'] = load_results(args.original_mem_results)
        if args.original_no_mem_results:
            results_dict['original_no_mem'] = load_results(args.original_no_mem_results)
        
        # Pick static from whatever is available
        for key in ['adaptive_mem', 'adaptive_no_mem', 'original_mem', 'original_no_mem']:
            if results_dict.get(key):
                results_dict['static'] = results_dict[key]
                break
             
    else:
        # 1. Adaptive + Memory
        if not args.skip_adaptive_mem:
            res = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_adaptive_mem_{timestamp}.jsonl"
            run_single_experiment(args, 'adaptive_mem', 0.1, 'adaptive', res, clean_env)
            results_dict['adaptive_mem'] = load_results(res)
            time.sleep(5)
        
        # 2. Adaptive + No Memory
        if not args.skip_adaptive_no_mem:
            res = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_adaptive_no_mem_{timestamp}.jsonl"
            run_single_experiment(args, 'adaptive_no_mem', 0.0, 'adaptive', res, clean_env)
            results_dict['adaptive_no_mem'] = load_results(res)
            time.sleep(5)
            
        # 3. Original + Memory
        if not args.skip_original_mem:
            res = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_original_mem_{timestamp}.jsonl"
            run_single_experiment(args, 'original_mem', 0.1, 'original', res, clean_env)
            results_dict['original_mem'] = load_results(res)
            time.sleep(5)

        # 4. Original + No Memory
        if not args.skip_original_no_mem:
            res = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_original_no_mem_{timestamp}.jsonl"
            run_single_experiment(args, 'original_no_mem', 0.0, 'original', res, clean_env)
            results_dict['original_no_mem'] = load_results(res)
        
        # Static from available
        for key in ['adaptive_mem', 'adaptive_no_mem', 'original_mem', 'original_no_mem']:
            if results_dict.get(key):
                results_dict['static'] = results_dict[key]
                break

    if any(results_dict.values()):
        plot_comparison(results_dict, figures_dir, f"{args.name}_{args.dataset_name}_{args.model_class}_{timestamp}")
    else:
        print("No results to plot!")
    
    print(f"\nExperiment completed. Results in {figures_dir}")


if __name__ == "__main__":
    main()
