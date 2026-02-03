# *************************************************************************
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *************************************************************************

import os
import subprocess
import sys
import time
from pathlib import Path
import argparse

def _get_shawarma_home():
    env_home = os.getenv("Shawarma_HOME")
    if env_home:
        return env_home
    # Fallback to repository root if env var is missing (scripts/experiments -> Shawarma)
    return str(Path(__file__).resolve().parents[3])

Shawarma_home = _get_shawarma_home()
os.environ.setdefault("Shawarma_HOME", Shawarma_home)

SEQUENCE_MODELS = {"TextCNN1", "TextCNN2", "RNN1"}

COMMON_SEQUENCE_DEFAULTS = {
    "NUM_CLASSES": "7",
    "SEQUENCE_LENGTH": "9",
    "SEQUENCE_FEATURE_DIM": "2",
    "LEN_VOCAB": "1501",
    "IPD_VOCAB": "2561",
    "LEN_EMBEDDING_BITS": "10",
    "IPD_EMBEDDING_BITS": "8",
}

TEXTCNN_DEFAULTS = {
    "TEXTCNN_NK": "4",
    "TEXTCNN_EBDIN": "4",
}

RNN_DEFAULTS = {
    "RNN_IN": "12",
    "RNN_HIDDEN": "16",
    "RNN_DROPOUT": "0.0",
}

def _env(key, default):
    return os.getenv(key, default)

def _append_sequence_model_args(cmd, model_class, env_prefix):
    if model_class not in SEQUENCE_MODELS:
        return

    def get_param(name, defaults):
        full_key = f"{env_prefix}_{name}"
        default_val = defaults.get(name)
        return _env(full_key, default_val)

    for name in COMMON_SEQUENCE_DEFAULTS:
        cmd.extend([
            f"--{name.lower()}",
            get_param(name, COMMON_SEQUENCE_DEFAULTS),
        ])

    if model_class in ["TextCNN1", "TextCNN2"]:
        for name in TEXTCNN_DEFAULTS:
            cmd.extend([
                f"--{name.lower()}",
                get_param(name, TEXTCNN_DEFAULTS),
            ])
    else:
        for name in RNN_DEFAULTS:
            cmd.extend([
                f"--{name.lower()}",
                get_param(name, RNN_DEFAULTS),
            ])

def run_control_plane_experiment(output_file=None, model_class=None, base_model_path=None, labeler_class=None, labeler_path=None, use_ground_truth_for_drift=False, memory_sample_ratio=None, device=None):
    # TextCNN1, RNN1, MLP1, TextCNN2
    if model_class is None:
        model_class = _env("SHAWARMA_CP_MODEL_CLASS", "RNN1")
    
    if base_model_path is None:
        if model_class == "TextCNN1":
            base_model_path = f"{Shawarma_home}/emulation/models/checkpoint/cnn/best_TextCNN1_custom.pth"
        elif model_class == "TextCNN2":
            base_model_path = f"{Shawarma_home}/emulation/models/checkpoint/cnn/best_TextCNN2_custom.pth"
        elif model_class == "RNN1":
            base_model_path = f"{Shawarma_home}/emulation/models/checkpoint/rnn/best_RNN1_custom.pth"
        else:
            base_model_path = _env("SHAWARMA_CP_BASE_MODEL_PATH", f"{Shawarma_home}/emulation/models/checkpoint/mlp/best_MLP1_custom.pth")

    # TextCNNTeacher, RNNTeacher, MLP1Teacher, MLP1Teacher_multi
    if labeler_class is None:
        labeler_class = _env("SHAWARMA_CP_LABELER_CLASS", "RNNTeacher")
    
    # 
    if labeler_path is None:
        if "TextCNN" in labeler_class:
            labeler_path = f"{Shawarma_home}/emulation/models/checkpoint/cnn/best_TextCNNTeacher_custom.pth"
        elif "RNN" in labeler_class:
            labeler_path = f"{Shawarma_home}/emulation/models/checkpoint/rnn/best_RNNTeacher_custom.pth"
        elif "MLP1Teacher_multi" in labeler_class:
            labeler_path = _env("SHAWARMA_CP_LABELER_PATH", f"{Shawarma_home}/emulation/models/checkpoint/mlp/best_MLP1Teacher_multi_custom.pth")
        else:
            labeler_path = _env("SHAWARMA_CP_LABELER_PATH", f"{Shawarma_home}/emulation/models/checkpoint/mlp/best_MLP1Teacher_custom.pth")
    if output_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"{Shawarma_home}/emulation/scripts/results/control_plane_torchlens-experiment_{timestamp}.jsonl"

    learning_rate = _env("SHAWARMA_CP_LEARNING_RATE", "0.005")
    total_epochs = _env("SHAWARMA_CP_TOTAL_EPOCHS", "60")
    retrain_batch_size = _env("SHAWARMA_CP_RETRAIN_BATCH_SIZE", "512")
    
    # Memory sample ratio for hybrid training strategy
    if memory_sample_ratio is None:
        memory_sample_ratio = _env("SHAWARMA_CP_MEMORY_SAMPLE_RATIO", "0.1")

    # Device selection: control plane defaults to cuda:0
    if device is None:
        device = _env("SHAWARMA_CP_DEVICE", "cuda:0")

    cmd = [
        sys.executable,
        f"{Shawarma_home}/emulation/benchmarks/control_plane_retrain_torchlens.py",

        "--device",
        device,

        "--application_type",
        "intrusion_detection",

        "--job_name",
        "control_plane_torchlens_experiment",

        "--labeler_type",
        "DNN_classifier",

        "--labeler_dnn_class",
        labeler_class,

        "--labeler_dnn_path",
        labeler_path,

        "-r",
        learning_rate,
        "--total_epochs",
        total_epochs,
        
        "--model_class",
        model_class,

        "-m",
        base_model_path,

        # Output File
        "-o",
        output_file,

        "--logdir",
        f"{Shawarma_home}/emulation/logs/control_plane",

        "--test_size",
        "100",

        "--memory_size",
        "5000",
        "--label_diff_threshold",
        "0.05",
        
        "--memory_sample_ratio",
        str(memory_sample_ratio),

        "--retrain_batch_size",
        retrain_batch_size,

        "--use_ground_truth_for_drift",

        "--adaptive_freeze",
    ]

    _append_sequence_model_args(cmd, model_class, "SHAWARMA_CP")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_file", help="Path to output JSONL file")
    parser.add_argument("--model_class", help="Model class name")
    parser.add_argument("--model_path", help="Path to the base model")
    parser.add_argument("--labeler_class", help="Labeler class name")
    parser.add_argument("--labeler_path", help="Path to the labeler model")
    parser.add_argument("--use_ground_truth_for_drift", action="store_true", help="Use ground-truth labels for drift detection")
    parser.add_argument("--memory_sample_ratio", type=float, help="Ratio of memory samples to mix with relabeled samples (0.0-1.0, default: 0.1)")
    parser.add_argument("--device", type=str, help="CUDA device for control plane (default: cuda:0)")
    args = parser.parse_args()

    run_control_plane_experiment(
        output_file=args.output_file,
        model_class=args.model_class,
        base_model_path=args.model_path,
        labeler_class=args.labeler_class,
        labeler_path=args.labeler_path,
        use_ground_truth_for_drift=args.use_ground_truth_for_drift,
        memory_sample_ratio=args.memory_sample_ratio,
        device=args.device,
    )
