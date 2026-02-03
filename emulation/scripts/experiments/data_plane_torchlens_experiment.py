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
import sys
import subprocess
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

    for name, default_val in COMMON_SEQUENCE_DEFAULTS.items():
        cmd.extend([
            f"--{name.lower()}",
            get_param(name, COMMON_SEQUENCE_DEFAULTS),
        ])

    if model_class in ["TextCNN1", "TextCNN2"]:
        for name, default_val in TEXTCNN_DEFAULTS.items():
            cmd.extend([
                f"--{name.lower()}",
                get_param(name, TEXTCNN_DEFAULTS),
            ])
    else:
        for name, default_val in RNN_DEFAULTS.items():
            cmd.extend([
                f"--{name.lower()}",
                get_param(name, RNN_DEFAULTS),
            ])

def run_data_plane_experiment(feature_file="cic-ids-2018-redistributed_X.npy", label_file="cic-ids-2018-redistributed_y_binary.npy", eval_frequency="4000", device=None, linc_rules_path=None, seed=None, deterministic=False):
    # TextCNN1, RNN1, MLP1, TextCNN2
    model_class = _env("SHAWARMA_DP_MODEL_CLASS", "RNN1")
    base_model_path = _env("SHAWARMA_DP_BASE_MODEL_PATH", None)
    if base_model_path is None:
        if model_class == "TextCNN1":
            base_model_path = f"{Shawarma_home}/emulation/models/checkpoint/cnn/best_TextCNN1_custom.pth"
        elif model_class == "TextCNN2":
            base_model_path = f"{Shawarma_home}/emulation/models/checkpoint/cnn/best_TextCNN2_custom.pth"
        elif model_class == "RNN1":
            base_model_path = f"{Shawarma_home}/emulation/models/checkpoint/rnn/best_RNN1_custom.pth"
        else:
            base_model_path = f"{Shawarma_home}/emulation/models/checkpoint/mlp/best_MLP1_custom.pth"
    dataset_root = _env("SHAWARMA_DP_DATA_ROOT", f"{Shawarma_home}/emulation/datasets/")
    
    # LINC rules path from environment or parameter
    if linc_rules_path is None:
        linc_rules_path = _env("SHAWARMA_DP_LINC_RULES_PATH", None)

    # Device selection: data plane defaults to cuda:1 to avoid GPU contention with control plane
    if device is None:
        device = _env("SHAWARMA_DP_DEVICE", "cuda:1")

    cmd = [
        sys.executable,
        f"{Shawarma_home}/emulation/benchmarks/data_plane_inference_torchlens.py",

        "--device",
        device,

        "--application_type",
        "intrusion_detection",

        "--job_name",
        "data_plane_torchlens_experiment",

        "--eval_frequency",
        str(eval_frequency),

        "--batch_size",
        "16",

        "--model_class",
        model_class,

        "-m",
        base_model_path,

        "-i",
        dataset_root,

        "--feature_file",
        feature_file,

        "--label_file",
        label_file,

        "--shutdown_control_plane",
    ]

    if seed is None:
        env_seed = _env("SHAWARMA_DP_SEED", None)
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = None

    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if deterministic:
        cmd.append("--deterministic")
    
    # Add LINC rules path if provided
    if linc_rules_path:
        cmd.extend(["--linc_rules_path", linc_rules_path])

    _append_sequence_model_args(cmd, model_class, "SHAWARMA_DP")
        
    subprocess.run(cmd)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_file", default="cic-ids-2018-redistributed_X.npy", help="Feature file name")
    parser.add_argument("--label_file", default="cic-ids-2018-redistributed_y_binary.npy", help="Label file name")
    parser.add_argument("--eval_frequency", default="4000", help="Evaluation frequency")
    parser.add_argument("--device", type=str, help="CUDA device for data plane (default: cuda:1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA behavior (may reduce performance)")
    args = parser.parse_args()

    run_data_plane_experiment(
        feature_file=args.feature_file, 
        label_file=args.label_file,
        eval_frequency=args.eval_frequency,
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
