import argparse
import logging
import os
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter, MultipleLocator
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

def get_shawarma_home():
    env_home = os.getenv("Shawarma_HOME")
    if env_home:
        return env_home
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

SHAWARMA_HOME = get_shawarma_home()
DATA_DIR = os.path.join(SHAWARMA_HOME, 'emulation', 'datasets')
DEFAULT_FIGURES_DIR = os.path.join(SHAWARMA_HOME, 'emulation', 'scripts', 'figures')


def setup_academic_plot_style():
    """Apply a paper-friendly matplotlib style (serif fonts, subtle grid, Type 42 fonts)."""
    # Prefer a paper-oriented preset when available.
    try:
        plt.style.use('seaborn-v0_8-paper')
    except Exception:
        try:
            plt.style.use('seaborn-paper')
        except Exception:
            pass

    mpl.rcParams.update({
        # Fonts
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 22,
        'axes.labelsize': 26,
        'axes.titlesize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'mathtext.fontset': 'stix',
        # Vector export compatibility (e.g., camera-ready PDFs)
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        # Axes/lines
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.2,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.8,
        # Layout + saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

def resolve_path(path):
    if path and not os.path.isabs(path):
        return os.path.join(DATA_DIR, path)
    return path

def analyze_distribution(features, labels, window_size, label_names=None):
    """
    Core analysis logic for both CSV and NPY data.
    """
    total_samples = len(labels)
    total_windows = total_samples // window_size
    
    logging.info(f"Total samples: {total_samples}")
    logging.info(f"Total windows: {total_windows}")
    
    stats_data = []

    for w in range(total_windows):
        start_idx = w * window_size
        end_idx = start_idx + window_size
        
        window_y = labels[start_idx:end_idx]
        window_X = features[start_idx:end_idx]
        
        # Get labels in order of appearance, not sorted
        _, unique_indices = np.unique(window_y, return_index=True)
        unique_labels = window_y[np.sort(unique_indices)]
        
        for label in unique_labels:
            label_mask = (window_y == label)
            total_count = np.sum(label_mask)
            
            # Extract features for this label
            label_X = window_X[label_mask]
            
            # Flatten features to check for duplicates (works for both 2D and 3D)
            flattened_X = label_X.reshape(len(label_X), -1)
            
            # Calculate unique rows
            unique_X = np.unique(flattened_X, axis=0)
            unique_count = len(unique_X)
            
            dup_rate = 1.0 - (unique_count / total_count) if total_count > 0 else 0
            
            display_label = label
            if label_names and label in label_names:
                display_label = label_names[label]

            stats_data.append({
                'Window_ID': w,
                'Label': display_label,
                'Total_Count': int(total_count),
                'Unique_Count': int(unique_count),
                'Duplication_Rate': f"{dup_rate:.2%}"
            })

        if w % 10 == 0:
            logging.info(f"Processing window {w}/{total_windows}...")

    return pd.DataFrame(stats_data)


def plot_class_distribution(stats_df, window_stride=1, figures_dir=DEFAULT_FIGURES_DIR):
    """Plot a single stacked-area chart of class share across windows (optionally aggregated)."""
    if stats_df.empty:
        logging.warning("No stats available; skipping plot generation.")
        return

    setup_academic_plot_style()
    os.makedirs(figures_dir, exist_ok=True)

    # Get labels in order of first appearance
    label_order = stats_df.groupby('Label')['Window_ID'].min().sort_values().index.tolist()

    pivot = stats_df.pivot_table(
        index='Window_ID',
        columns='Label',
        values='Total_Count',
        aggfunc='sum',
        fill_value=0,
    ).sort_index()
    
    # Reorder columns to match appearance order
    pivot = pivot.reindex(columns=label_order, fill_value=0)

    if window_stride > 1:
        # Aggregate contiguous windows into bins to reduce points
        grouped = pivot.groupby(pivot.index // window_stride).sum()
        x_axis = grouped.index * window_stride
        counts = grouped
    else:
        x_axis = pivot.index
        counts = pivot

    # Shift to 1-based indexing for plot
    x_axis = x_axis + 1

    window_totals = counts.sum(axis=1)
    # Avoid division by zero; replace zeros with ones temporarily
    safe_totals = window_totals.replace(0, 1)
    percentages = counts.div(safe_totals, axis=0) * 100.0

    label_strings = [str(lbl) for lbl in percentages.columns]

    # Color palette sized to number of classes
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(len(label_strings))]

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.stackplot(
        x_axis,
        [percentages[label].values for label in percentages.columns],
        labels=label_strings,
        colors=colors,
        alpha=0.95,
        linewidth=0.6,
        edgecolor='white',
    )

    ax.set_ylabel('Traffic Share (%)')
    ax.set_xlabel('Window')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # Custom ticks: 0, 20, ...
    x_max = x_axis.max()
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlim(left=0, right=x_max)

    ax.grid(True, axis='y')

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    title = f"Class distribution across windows (stride {window_stride})"
    ax.set_title(title)

    # Legend at bottom right
    ax.legend(
        loc='lower right',
        frameon=True,
        ncol=2,
        fontsize=20,
        title='Class',
        title_fontsize=24,
    )

    fig.tight_layout()

    output_png = os.path.join(figures_dir, f"class_distribution_all_windows_stride{window_stride}.png")
    output_pdf = output_png.replace('.png', '.pdf')
    fig.savefig(output_png)
    fig.savefig(output_pdf)
    plt.close(fig)
    logging.info(f"Saved distribution plot: {output_png}")
    logging.info(f"Saved distribution plot: {output_pdf}")


def get_dataset_tag(input_path: str) -> str:
    """Derive a concise dataset tag from the input filename for plot naming."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    suffixes = [
        "-redistributed",
        "-deduplicated",
        "_X",
        "_y_multi",
        "_y",
        "_labels",
    ]
    # Repeatedly strip suffixes to handle multiple layers like -redistributed_X
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if base.endswith(suf):
                base = base[: -len(suf)]
                changed = True
    return base.rstrip('-_') or "dataset"


def load_label_mapping(input_labels_path, dataset_tag):
    """Build a mapping from numeric label IDs (as stored in y.npy) back to human-readable names.

    Supported formats:
    - Name-to-ID mapping JSON, e.g. {"BENIGN": 0, "Bot": 1}.
      This is inverted to {0: "BENIGN", 1: "Bot"}.
    - ID remap JSON, e.g. {"4": 3, "5": 4} meaning old_id -> new_id.
      When present, we try to combine it with a dataset labels.json (name -> old_id)
      to derive {new_id: name}.
    """
    remap_path = input_labels_path.replace('.npy', '.json')
    labels_paths = [
        os.path.join(os.path.dirname(input_labels_path), "labels.json"),
        os.path.join(os.path.dirname(input_labels_path), dataset_tag, "labels.json"),
        os.path.join(DATA_DIR, dataset_tag, "labels.json"),
    ]

    remap = None
    if os.path.exists(remap_path):
        try:
            with open(remap_path, 'r') as f:
                remap = json.load(f)
            logging.info(f"Loaded label JSON from {remap_path}")
        except Exception as e:
            logging.warning(f"Failed to load label JSON from {remap_path}: {e}")

    name_to_old = None
    for p in labels_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    name_to_old = json.load(f)
                logging.info(f"Loaded dataset labels from {p}")
                break
            except Exception as e:
                logging.warning(f"Failed to load dataset labels from {p}: {e}")

    def _keys_are_numeric_str(d: dict) -> bool:
        try:
            return all(str(k).strip().lstrip('-').isdigit() for k in d.keys())
        except Exception:
            return False

    def _values_are_int_like(d: dict) -> bool:
        try:
            for v in d.values():
                if isinstance(v, bool):
                    return False
                if not isinstance(v, (int, np.integer)):
                    return False
            return True
        except Exception:
            return False

    # Case A: remap JSON is actually name->id (common for CIC/USTC)
    if isinstance(remap, dict) and remap and (not _keys_are_numeric_str(remap)):
        try:
            return {int(v): str(k) for k, v in remap.items()}
        except Exception as e:
            logging.warning(f"Failed to invert name->id mapping from {remap_path}: {e}")

    # Case B: remap JSON is old_id->new_id; combine with labels.json (name->old_id)
    if isinstance(remap, dict) and remap and _keys_are_numeric_str(remap) and _values_are_int_like(remap):
        if isinstance(name_to_old, dict) and name_to_old:
            try:
                old_to_name = {int(old_id): str(name) for name, old_id in name_to_old.items()}
                new_to_name = {}
                for old_id_str, new_id in remap.items():
                    old_id = int(old_id_str)
                    if old_id in old_to_name:
                        new_to_name[int(new_id)] = old_to_name[old_id]
                if new_to_name:
                    return new_to_name
            except Exception as e:
                logging.warning(f"Failed to build new_id->name mapping from remap+labels.json: {e}")

        logging.info("Label remap found but no names; using numeric IDs.")
        return None

    # Case C: no remap; labels.json directly provides name->id for current y.npy
    if isinstance(name_to_old, dict) and name_to_old:
        try:
            return {int(v): str(k) for k, v in name_to_old.items()}
        except Exception as e:
            logging.warning(f"Failed to invert dataset labels.json: {e}")

    logging.info("No label mapping found; using numeric IDs.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Unified Traffic Distribution Analysis (NPY only)")
    parser.add_argument("--input", required=True, help="Path to input NPY features (_X.npy)")
    parser.add_argument("--input_labels", required=True, help="Path to input NPY labels (_y.npy)")
    parser.add_argument("--output", required=True, help="Path to output report (.csv)")
    parser.add_argument("--window_size", type=int, default=4000, help="Window size for analysis")
    parser.add_argument("--plot_stride", type=int, default=1, help="Aggregate this many windows per plotted point (single figure)")
    parser.add_argument(
        "--figures_dir",
        default=None,
        help=f"Directory to save distribution plots (default: {DEFAULT_FIGURES_DIR})",
    )
    
    args = parser.parse_args()

    # Resolve paths relative to emulation/datasets/
    args.input = resolve_path(args.input)
    args.input_labels = resolve_path(args.input_labels)
    args.output = resolve_path(args.output)

    # NPY Loading (CSV support removed)
    logging.info(f"Loading NPY data: {args.input} and {args.input_labels} ...")
    X = np.load(args.input)
    y = np.load(args.input_labels)
    
    dataset_tag = get_dataset_tag(args.input)
    label_names = load_label_mapping(args.input_labels, dataset_tag)
    
    result_df = analyze_distribution(X, y, args.window_size, label_names=label_names)

    # Sort and Save
    result_df = result_df.sort_values(by=['Window_ID', 'Total_Count'], ascending=[True, False])
    
    logging.info(f"Saving report to {args.output} ...")
    result_df.to_csv(args.output, index=False)
    
    figures_dir = args.figures_dir
    if figures_dir:
        if not os.path.isabs(figures_dir):
            figures_dir = os.path.abspath(os.path.join(SHAWARMA_HOME, figures_dir))
    else:
        figures_dir = DEFAULT_FIGURES_DIR

    logging.info(f"Saving distribution plots every {args.plot_stride} windows to {figures_dir} (tag: {dataset_tag}) ...")
    output_file = os.path.join(figures_dir, f"class_distribution_{dataset_tag}_stride{args.plot_stride}.png")

    # Generate plot and move to tagged filename
    plot_class_distribution(result_df, window_stride=args.plot_stride, figures_dir=figures_dir)
    # Rename the default file if it exists
    default_plot = os.path.join(figures_dir, f"class_distribution_all_windows_stride{args.plot_stride}.png")
    if os.path.exists(default_plot):
        os.replace(default_plot, output_file)
    default_pdf = os.path.join(figures_dir, f"class_distribution_all_windows_stride{args.plot_stride}.pdf")
    output_pdf = output_file.replace('.png', '.pdf')
    if os.path.exists(default_pdf):
        os.replace(default_pdf, output_pdf)
    logging.info(f"Final plot saved: {output_file}")
    logging.info(f"Final plot saved: {output_pdf}")
    
    logging.info("\n[Result Preview]")
    print(result_df.head(15).to_string(index=False))
    logging.info("Done.")

if __name__ == "__main__":
    main()
