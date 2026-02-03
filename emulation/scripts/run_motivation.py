#!/usr/bin/env python3
# *************************************************************************
#
# Motivation experiment: Full Retrain (Original)
# Based on run_retrain_comparison.py
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

Shawarma_home = os.getenv("Shawarma_HOME")
if Shawarma_home is None:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    os.environ["Shawarma_HOME"] = path
    Shawarma_home = path
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
        # sys.exit(result.returncode) # Don't exit immediately so we can cleanup
        raise RuntimeError(f"{description} failed")
    print(f"Finished: {description}\n")


def kill_process_on_port(port):
    print(f"Cleaning up port {port}...")
    try:
        cmd = ["lsof", "-t", f"-i:{port}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    print(f"Killing process {pid} on port {port}")
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            time.sleep(2)
    except Exception as e:
        print(f"Port cleanup failed: {e}")


def run_single_experiment(args, results_file, clean_env, deployment_delay=0.0):
    """
    Run the motivation experiment (Original / Full Retrain).
    """
    retrain_mode = 'original' # Fixed to original for motivation experiment
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {retrain_mode} retraining ({args.model_class}), Delay: {deployment_delay*1000}ms")
    print(f"{'='*60}\n")
    
    kill_process_on_port(50051)
    
    # Build control plane command
    cmd_cp = [
        sys.executable,
        f"{Shawarma_home}/emulation/benchmarks/control_plane_retrain_torchlens.py",
        "--device", args.cp_device,
        "--application_type", "intrusion_detection",
        "--job_name", f"cp_{retrain_mode}_{args.dataset_name}_{args.model_class}_retrain_delay_{deployment_delay}",
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
        "--memory_sample_ratio", "0.0",
        "--retrain_batch_size", "512",
        "--use_ground_truth_for_drift",
    ]
    
    if deployment_delay > 0:
        cmd_cp.extend(["--deployment_delay", str(deployment_delay)])

    if args.seed is not None:
        cmd_cp.extend(["--seed", str(args.seed)])
    if args.deterministic:
        cmd_cp.append("--deterministic")
    
    # Add sequence model specific arguments (TextCNN1, RNN1, etc.)
    append_sequence_model_args(cmd_cp, args.model_class, num_classes=args.num_classes)
    
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


def plot_results(results_dict, output_dir, experiment_name):
    """Plot accuracy and F1 scores for multiple experiments with different delays."""
    if not results_dict:
        print("No results to plot")
        return

    # --- Style Configuration ---
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('default')

    # Academic Paper Settings
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

    # Find max length across all results
    max_len = 0
    for delay, res in results_dict.items():
        if res and 'my_f1s' in res:
            max_len = max(max_len, len(res['my_f1s']))
    
    if max_len == 0:
        print("Empty results data")
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
    
    os.makedirs(output_dir, exist_ok=True)

    # Use a color map for different delays
    markers = ['o', 's', '^', 'D', 'v', 'X']
    
    sorted_delays = sorted(results_dict.keys())
    num_delays = len(sorted_delays)
    
    # Generate blue gradient: Darkest (small delay) -> Lightest (large delay)
    if num_delays > 1:
        color_vals = np.linspace(0.9, 0.4, num_delays)
    else:
        color_vals = [0.9]

    # Helper for adding offline model
    def add_offline_model_curve(metric_key):
        # Just pick from the first available result that has the static data
        for _d in sorted_delays:
            if results_dict[_d] and metric_key in results_dict[_d]:
                static_data = results_dict[_d][metric_key]
                plt.plot(windows[:len(static_data)], static_data, linewidth=2.5,
                         linestyle='--', color='gray', label='Offline Model',
                         zorder=5) # High zorder to be visible
                return

    # Plot F1
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    add_phase_backgrounds(ax, max_len)

    # Add Offline Model Curve
    add_offline_model_curve('static_f1s')
    
    for idx, delay in enumerate(sorted_delays):
        res = results_dict[delay]
        if not res: continue
        
        f1s = res.get('my_f1s', [])
        
        label = f"Delay {delay}s" if float(delay) > 0 else "Full Retrain"
        
        # Determine color
        color = plt.cm.Blues(color_vals[idx])
        marker = markers[idx % len(markers)]
        
        plt.plot(windows[:len(f1s)], f1s, linewidth=2.5, marker=marker, 
                 label=label, color=color, markevery=5, zorder=3 + idx)

    plt.xlabel('Window Index')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison - Different Deployment Delays')
    plt.xlim([0, max_len + 2])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='best', frameon=True, framealpha=0.5, fancybox=True, shadow=True, fontsize=24, ncol=2)
    plt.savefig(f"{output_dir}/{experiment_name}_f1.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/{experiment_name}_f1.pdf", bbox_inches='tight')
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    add_phase_backgrounds(ax, max_len)

    # Add Offline Model Curve
    add_offline_model_curve('static_accuracy')
    
    for idx, delay in enumerate(sorted_delays):
        res = results_dict[delay]
        if not res: continue
        
        acc = res.get('my_accuracy', [])
        
        if delay == 0:
             label = 'Full Retrain (0ms)'
        else:
             label = f'Full Retrain ({int(delay*1000)}ms)'
        
        # Determine color
        color = plt.cm.Blues(color_vals[idx])
        marker = markers[idx % len(markers)]
        
        plt.plot(windows[:len(acc)], acc, linewidth=2.5, marker=marker, 
                 label=label, color=color, markevery=5, zorder=3 + idx)

    plt.xlabel('Window Index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison - Different Deployment Delays')
    plt.xlim([0, max_len + 2])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='best', frameon=True, framealpha=0.5, fancybox=True, shadow=True, fontsize=24, ncol=2)
    plt.savefig(f"{output_dir}/{experiment_name}_acc.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/{experiment_name}_acc.pdf", bbox_inches='tight')
    plt.close()

    # --- Plot Combined F1 and Accuracy (Stacked Subplots) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Add backgrounds to both axes
    add_phase_backgrounds(ax1, max_len)
    add_phase_backgrounds(ax2, max_len)

    # Helper for adding offline model (Stacked)
    def add_offline_model_curve_stacked(ax, metric_key):
        # Find any result to get static data
        for res in results_dict.values():
            if res and metric_key in res:
                data = res[metric_key]
                if data:
                     ax.plot(windows[:len(data)], data, 
                             linewidth=3.0, linestyle='--', color='gray', label='Static', zorder=1)
                return

    # Add Offline Model Curves
    add_offline_model_curve_stacked(ax1, 'static_f1s')
    add_offline_model_curve_stacked(ax2, 'static_accuracy')

    # Plot Delays
    for idx, delay in enumerate(sorted_delays):
        res = results_dict[delay]
        if not res: continue
            
        f1s = res.get('my_f1s', [])
        acc = res.get('my_accuracy', [])
        
        if float(delay) == 0:
             label = 'Full Retrain (0ms)'
        else:
             label = f'Full Retrain ({int(float(delay)*1000)}ms)'
        
        # Determine color
        color = plt.cm.Blues(color_vals[idx])
        marker = markers[idx % len(markers)]
        
        # Plot F1 on ax1
        if f1s:
            ax1.plot(windows[:len(f1s)], f1s, linewidth=2.5, marker=marker, 
                    label=label, color=color, markevery=5, zorder=3 + idx)
            
        # Plot Accuracy on ax2
        if acc:
            ax2.plot(windows[:len(acc)], acc, linewidth=2.5, marker=marker, 
                    label=label, color=color, markevery=5, zorder=3 + idx)

    ax1.set_ylabel('F1 Score')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Window Index')
    
    ax1.set_ylim(top=1.05)
    ax2.set_ylim(top=1.05)
    ax1.set_xlim([0, max_len + 2])
    ax2.set_xlim([0, max_len + 2])
    
    ax1.set_title('Impact of Deployment Delay')
    
    # Legend - put on bottom plot (Accuracy)
    ax2.legend(loc='best', frameon=True, framealpha=0.4, fancybox=True, shadow=True, ncol=1, fontsize=28)
    
    # Adjust layout to remove gap
    plt.subplots_adjust(hspace=0.05)
    
    combined_output = f"{output_dir}/{experiment_name}_combined_stacked.png"
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    plt.savefig(combined_output.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Combined stacked comparison saved to {combined_output}")
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Motivation Experiment (Full Retrain)")
    parser.add_argument("--name", type=str, default="motivation_retrain", help="Experiment name")
    parser.add_argument("--feature_file", type=str, default="experiment_2018_0.6_X.npy", help="Feature file")
    parser.add_argument("--label_file", type=str, default="experiment_2018_0.6_y.npy", help="Label file")
    parser.add_argument("--cp_device", type=str, default="cuda:0", help="Control plane device")
    parser.add_argument("--dp_device", type=str, default="cuda:1", help="Data plane device")
    parser.add_argument("--model_class", type=str, default="MLP1", 
                        choices=["MLP1", "TextCNN1", "TextCNN2", "RNN1"],
                        help="Model class to use (MLP1, TextCNN1, TextCNN2, RNN1)")
    parser.add_argument("--labeler_class", type=str, default=None,
                        help="Labeler class (default: {model_class}Teacher)")
    parser.add_argument("--num_classes", type=int, default=None, 
                        help="Number of classes (auto-detected from label file if not provided)")
    parser.add_argument("--plot_only", action="store_true", help="Only plot from existing results")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., cic-ids-2018) - overrides feature_file derivation")
    parser.add_argument("--results_file", type=str, help="Path to existing results file (single file)")
    parser.add_argument("--deployment_delay", type=float, default=0.0, help="Single deployment delay (backwards compatibility)")
    parser.add_argument("--delays", type=str, default="0.0", help="Comma-separated list of delays to run (e.g., '0.0,30.0,60.0')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA behavior")
    parser.add_argument("--timestamp", type=str, default=None, help="Timestamp signature for plot_only matching")
    
    args = parser.parse_args()
    
    if args.dataset:
        dataset_name = args.dataset
    else:
        # Extract dataset name from feature_file
        dataset_basename = os.path.basename(args.feature_file)
        dataset_name = dataset_basename.split('.')[0].replace('_X', '').replace('_redist', '').replace('-redistributed', '')
    
    # Normalize common dataset names
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
                dataset_class_map = {
                    'cic-ids-2018': 7,
                    'iscxvpn': 7,
                    'ustc-tfc2016': 12,
                }
                args.num_classes = dataset_class_map.get(args.dataset_name, 7)
                print(f"Falling back to default: {args.num_classes} classes for {args.dataset_name}")
        else:
            dataset_class_map = {
                'cic-ids-2018': 7,
                'iscxvpn': 7,
                'ustc-tfc2016': 12,
            }
            args.num_classes = dataset_class_map.get(args.dataset_name, 7)
            print(f"Label file not found, using default: {args.num_classes} classes for {args.dataset_name}")
    
    # Set default labeler class
    if args.labeler_class is None:
        labeler_class_map = {
            "MLP1": "MLP1Teacher",
            "TextCNN1": "TextCNNTeacher",
            "TextCNN2": "TextCNNTeacher",
            "RNN1": "RNNTeacher",
        }
        args.labeler_class = labeler_class_map.get(args.model_class, f"{args.model_class}Teacher")
    
    # Determine model checkpoint directory
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
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Parse delays
    if args.delays:
        delays = [float(d) for d in args.delays.split(',')]
    else:
        delays = [args.deployment_delay]
    
    results_dict = {}

    if args.plot_only:
        # Load logic slightly more complex for plot_only
        # If user provides --results_file, we assume it's for 0.0 delay or single plot
        # But if they want to plot multiple from existing, they likely need to rely on auto-naming
        if args.results_file:
             results_dict[0.0] = load_results(args.results_file)
        else:
             # Try to find files matching pattern for each delay
             if not args.timestamp:
                 print("Error: --timestamp required for plot_only with multiple delays (unless --results_file is used)")
                 return
             
             for delay in delays:
                 pattern = f"{args.name}_{args.dataset_name}_{args.model_class}_delay_{delay}_{args.timestamp}.jsonl"
                 # Backward compatibility: try without delay in name for 0.0 if not found
                 fpath = os.path.join(results_dir, pattern)
                 if not os.path.exists(fpath) and delay == 0.0:
                      fpath = os.path.join(results_dir, f"{args.name}_{args.dataset_name}_{args.model_class}_original_{args.timestamp}.jsonl")
                 
                 if os.path.exists(fpath):
                     print(f"Loading results for delay {delay}: {fpath}")
                     results_dict[delay] = load_results(fpath)
                 else:
                     print(f"Warning: Results file not found for delay {delay}: {fpath}")

        plot_results(results_dict, figures_dir, f"{args.name}_{args.dataset_name}_{args.model_class}_delays_{args.timestamp}")
        return

    # Clean environment
    clean_env = os.environ.copy()
    for key in list(clean_env.keys()):
        if key.startswith("SHAWARMA_CP_") or key.startswith("SHAWARMA_DP_"):
            del clean_env[key]

    # Run experiments
    for delay in delays:
        # Generate filename specific to delay
        if args.results_file and len(delays) == 1:
             current_results_file = args.results_file
        else:
             current_results_file = f"{results_dir}/{args.name}_{args.dataset_name}_{args.model_class}_delay_{delay}_{timestamp}.jsonl"
        
        run_single_experiment(args, current_results_file, clean_env, deployment_delay=delay)
        results_dict[delay] = load_results(current_results_file)
        
        # Brief pause between experiments
        time.sleep(5)

    plot_results(results_dict, figures_dir, f"{args.name}_{args.dataset_name}_{args.model_class}_delays_{timestamp}")


if __name__ == "__main__":
    main()
