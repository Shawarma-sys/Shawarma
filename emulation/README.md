# Shawarma Software Emulation

This directory contains the software emulation code for Shawarma. Below, we walk through how to set up the environment, prepare datasets, train models, run experiments, and generate key figures from the paper.

For hardware testbed experiments, see [`testbed/`](../testbed/).

## Getting Started 

### Environment Setup
To set up the environment, please run the commands below:
```bash
export Shawarma_HOME=$HOME/Shawarma
cd $Shawarma_HOME/
git pull
cd $Shawarma_HOME/emulation/
pip install -e .
cd $Shawarma_HOME/emulation/benchmarks/
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. control_plane.proto
```

### Data Preparation
We provide three datasets for evaluation purposes: `ISCXVPN`, `USTC-TFC2016`, and `CIC-IDS-2018`. Please download the datasets from the following links and place them under the `emulation/datasets/` folder.
* CIC-IDS-2018: [Link](https://www.unb.ca/cic/datasets/ids-2018.html)
* ISCXVPN: [Link](https://www.unb.ca/cic/datasets/vpn.html)
* USTC-TFC2016: [Link](https://github.com/yungshenglu/USTC-TFC2016)

#### Redistributing Datasets
To redistribute the datasets for concept drift simulation, run the following commands:

**CIC-IDS-2018:**
```bash
python emulation/datasets/redistribute_unified.py \
    --input cic-ids-2018/cic-ids-2018.csv \
    --benign_ratio 0.6 \
    --windows_per_attack 20 \
    --binary \
    --skip_attacks 'Infilteration'
```

**ISCXVPN:**
```bash
python emulation/datasets/redistribute_unified.py \
    --input iscxvpn/train/iscxvpn_train_data.npy \
    --input_labels iscxvpn/train/iscxvpn_train_labels.npy \
    --benign_ratio 0.6 \
    --windows_per_attack 20 \
    --skip_attacks '3'
```

**USTC-TFC2016:**
```bash
python emulation/datasets/redistribute_unified.py \
    --input ustc-tfc2016/train/ustc-tfc2016_train_data.npy \
    --input_labels ustc-tfc2016/train/ustc-tfc2016_train_labels.npy \
    --benign_ratio 0.6 \
    --windows_per_attack 20 \
    --skip_attacks '7'
```

### Analyzing Data Distribution
To analyze the distribution of the redistributed dataset:

```bash
# CIC-IDS-2018
python emulation/datasets/analyze_distribution_unified.py \
    --input cic-ids-2018-redistributed_X.npy \
    --input_labels cic-ids-2018-redistributed_y.npy \
    --output report.csv
    
# ISCXVPN
python emulation/datasets/analyze_distribution_unified.py \
    --input iscxvpn-redistributed_X.npy \
    --input_labels iscxvpn-redistributed_y.npy \
    --output report.csv

# USTC-TFC2016
python emulation/datasets/analyze_distribution_unified.py \
    --input ustc-tfc2016-redistributed_X.npy \
    --input_labels ustc-tfc2016-redistributed_y.npy \
    --output report.csv
```

## Model Training

### Training Data Plane Inference Model and Control Plane Labeler Model

The training script will automatically:
1. Train the model on the first 160,000 samples (windows 1-40) for inference models
2. Extract LINC and Helios rules for non-Teacher models
3. Save model checkpoints, LINC rules, and Helios rules to `emulation/models/checkpoint/`

#### MLP Models (CIC-IDS-2018)
```bash
# Train inference model (MLP1)
python emulation/models/train_experiment.py \
    --mode train \
    --model_type mlp \
    --model_class MLP1 \
    --dataset_name cic-ids-2018 \
    --train_data emulation/datasets/cic-ids-2018-redistributed_X.npy \
    --train_labels emulation/datasets/cic-ids-2018-redistributed_y_binary.npy

# Train labeler model (MLP1Teacher)
python emulation/models/train_experiment.py \
    --mode train \
    --model_type mlp \
    --model_class MLP1Teacher \
    --dataset_name cic-ids-2018 \
    --train_data emulation/datasets/cic-ids-2018-redistributed_X.npy \
    --train_labels emulation/datasets/cic-ids-2018-redistributed_y_binary.npy
```

#### CNN Models (ISCXVPN)
```bash
# Train inference model (TextCNN1)
python emulation/models/train_experiment.py \
    --mode train \
    --model_type cnn \
    --model_class TextCNN1 \
    --dataset_name iscxvpn \
    --train_data emulation/datasets/iscxvpn-redistributed_X.npy \
    --train_labels emulation/datasets/iscxvpn-redistributed_y.npy

# Train labeler model (TextCNNTeacher)
python emulation/models/train_experiment.py \
    --mode train \
    --model_type cnn \
    --model_class TextCNNTeacher \
    --dataset_name iscxvpn \
    --train_data emulation/datasets/iscxvpn-redistributed_X.npy \
    --train_labels emulation/datasets/iscxvpn-redistributed_y.npy
```

#### RNN Models (USTC-TFC2016)
```bash
# Train inference model (RNN1)
python emulation/models/train_experiment.py \
    --mode train \
    --model_type rnn \
    --model_class RNN1 \
    --dataset_name ustc-tfc2016 \
    --train_data emulation/datasets/ustc-tfc2016-redistributed_X.npy \
    --train_labels emulation/datasets/ustc-tfc2016-redistributed_y.npy

# Train labeler model (RNNTeacher)
python emulation/models/train_experiment.py \
    --mode train \
    --model_type rnn \
    --model_class RNNTeacher \
    --dataset_name ustc-tfc2016 \
    --train_data emulation/datasets/ustc-tfc2016-redistributed_X.npy \
    --train_labels emulation/datasets/ustc-tfc2016-redistributed_y.npy
```

### Extracting Rules from Existing Models

Extract rules from trained models without retraining:

```bash
# Auto-detect model path and extract both LINC and Helios rules
python emulation/models/train_experiment.py \
    --mode extract \
    --model_type cnn \
    --model_class TextCNN1 \
    --dataset_name iscxvpn

# Extract only LINC or Helios rules
python emulation/models/train_experiment.py \
    --mode extract \
    --model_type mlp \
    --model_class MLP1 \
    --dataset_name cic-ids-2018 \
    --extract_rules linc  # or 'helios' or 'both'

# Manually specify model path
python emulation/models/train_experiment.py \
    --mode extract \
    --model_path emulation/models/checkpoint/mlp/best_MLP1_cic-ids-2018.pth \
    --model_type mlp \
    --model_class MLP1 \
    --dataset_name cic-ids-2018
```

### Model Checkpoints
After training, models and rules are saved to:
- `emulation/models/checkpoint/mlp/` - MLP models
- `emulation/models/checkpoint/cnn/` - CNN models  
- `emulation/models/checkpoint/rnn/` - RNN models

Each directory contains:
- `best_{ModelClass}_{dataset}.pth` - Model weights
- `linc_rules_{ModelClass}_{dataset}.json` - LINC rules (for inference models only)
- `helios_rules_{ModelClass}_{dataset}.json` - Helios rules (for inference models only)

## Running Experiments

### Motivation Experiment
```bash
python emulation/datasets/mixed_attack_generator.py \
    --input cic-ids-2018/cic-ids-2018.csv \     
    --benign_ratio 0.6 \     
    --windows_per_attack 20 \     
    --binary \     
    --skip_attacks 'Infilteration'

python emulation/models/train_experiment.py \
    --mode train \
    --model_type mlp \
    --model_class MLP1 \
    --dataset_name cic-ids-2018 \
    --train_data emulation/datasets/cic-ids-2018-redistributed_mixed_X.npy \
    --train_labels emulation/datasets/cic-ids-2018-redistributed_mixed_y_binary.npy \
    --extract_rules none

python emulation/scripts/run_motivation.py \
    --name mlp1_motivation \
    --delays "0.0,0.03,0.06" \
    --model_class MLP1 \
    --feature_file cic-ids-2018-redistributed_mixed_X.npy \
    --label_file cic-ids-2018-redistributed_mixed_y_binary.npy \
    --cp_device cuda:0 \
    --dp_device cuda:1 \
    --seed 42
```

### Retrain Comparison Experiment (Original vs Adaptive)

Compare Original Retrain vs Adaptive Retrain methods:

**MLP1 on CIC-IDS-2018:**
```bash
python emulation/scripts/run_retrain_comparison.py \
    --name mlp1_comparison \
    --model_class MLP1 \
    --feature_file cic-ids-2018-redistributed_X.npy \
    --label_file cic-ids-2018-redistributed_y_binary.npy \
    --cp_device cuda:0 \
    --dp_device cuda:1 \
    --seed 42
```

**TextCNN1 on ISCXVPN:**
```bash
python emulation/scripts/run_retrain_comparison.py \
    --name textcnn_comparison \
    --model_class TextCNN1 \
    --feature_file iscxvpn-redistributed_X.npy \
    --label_file iscxvpn-redistributed_y.npy \
    --cp_device cuda:0 \
    --dp_device cuda:1 \
    --seed 42
```

**RNN1 on USTC-TFC2016:**
```bash
python emulation/scripts/run_retrain_comparison.py \
    --name rnn_comparison \
    --model_class RNN1 \
    --feature_file ustc-tfc2016-redistributed_X.npy \
    --label_file ustc-tfc2016-redistributed_y.npy \
    --cp_device cuda:0 \
    --dp_device cuda:1 \
    --seed 42
```

#### Replotting Existing Results (Academic Style)

You can regenerate the plots for existing experiments using the `--plot_only` flag. This will generate academic-style figures (Times New Roman, PDF Type 42 features) without re-running the heavy experiments.

**Using Timestamp (Auto-resolve files):**
If you have run the experiments previously, simply provide the timestamp:
```bash
python emulation/scripts/run_retrain_comparison.py \
    --plot_only \
    --name mlp1_comparison \
    --model_class MLP1 \
    --dataset cic-ids-2018 \
    --timestamp 20260121-213531
```

**Using Explicit File Paths:**
If auto-resolution fails or you want to specify exact files:
```bash
python emulation/scripts/run_retrain_comparison.py \
    --plot_only \
    --name mlp1_comparison \
    --model_class MLP1 \
    --dataset cic-ids-2018 \
    --original_results emulation/scripts/results/baselines_mlp1_cic-ids-2018_MLP1_original_20260121-213531.jsonl \
    --adaptive_results emulation/scripts/results/baselines_mlp1_cic-ids-2018_MLP1_adaptive_20260121-213531.jsonl
```

### Baselines Comparison Experiment (LINC vs Helios vs Original vs Adaptive)

Compare all baseline methods including LINC (ICNP 2024) and Helios (WWW 2025):

**MLP1 on CIC-IDS-2018:**
```bash
python emulation/scripts/run_baselines_comparison.py \
    --name baselines_mlp1 \
    --model_class MLP1 \
    --feature_file cic-ids-2018-redistributed_X.npy \
    --label_file cic-ids-2018-redistributed_y_binary.npy \
    --cp_device cuda:0 \
    --dp_device cuda:1 \
    --seed 42
```

**TextCNN1 on ISCXVPN:**
```bash
python emulation/scripts/run_baselines_comparison.py \
    --name baselines_textcnn \
    --model_class TextCNN1 \
    --feature_file iscxvpn-redistributed_X.npy \
    --label_file iscxvpn-redistributed_y.npy \
    --cp_device cuda:0 \
    --dp_device cuda:1 \
    --seed 42
```

**RNN1 on USTC-TFC2016:**
```bash
python emulation/scripts/run_baselines_comparison.py \
    --name baselines_rnn \
    --model_class RNN1 \
    --feature_file ustc-tfc2016-redistributed_X.npy \
    --label_file ustc-tfc2016-redistributed_y.npy \
    --cp_device cuda:0 \
    --dp_device cuda:1 \
    --seed 42
```

### Experiment Options

**Skip specific methods:**
```bash
# Skip LINC experiment
python emulation/scripts/run_baselines_comparison.py --name test --skip_linc

# Skip Helios experiment
python emulation/scripts/run_baselines_comparison.py --name test --skip_helios

# Skip original retrain experiment
python emulation/scripts/run_baselines_comparison.py --name test --skip_original

# Skip adaptive retrain experiment
python emulation/scripts/run_baselines_comparison.py --name test --skip_adaptive
```

**Helios-specific parameters:**
```bash
python emulation/scripts/run_baselines_comparison.py \
    --name test \
    --helios_max_rules 3000 \
    --helios_radio 1.5 \
    --helios_boost_num 6 \
    --helios_prune_rule 3
```

**LINC-specific parameters:**
```bash
python emulation/scripts/run_baselines_comparison.py \
    --name test \
    --linc_rule_threshold 0.3 \
    --linc_max_rules 1000 \
    --linc_update_samples 4000
```

**Plot from existing results:**
```bash
python emulation/scripts/run_baselines_comparison.py \
    --plot_only \
    --name baselines_mlp1 \
    --model_class MLP1 \
    --dataset cic-ids-2018 \
    --timestamp 20260121-213531
```
This command will automatically find results matching the pattern `{name}_{dataset}_{model}_{method}_{timestamp}.jsonl` in the results directory.

### Ablation Experiment
```bash
python emulation/datasets/mixed_attack_generator.py \
    --input cic-ids-2018/cic-ids-2018.csv \     
    --benign_ratio 0.6 \     
    --windows_per_attack 20 \     
    --binary \     
    --skip_attacks 'Infilteration'

python emulation/scripts/run_ablation.py \
    --name mlp1_ablation \
    --model_class MLP1 \
    --feature_file cic-ids-2018-redistributed_mixed_X.npy \
    --label_file cic-ids-2018-redistributed_mixed_y_binary.npy \
    --cp_device cuda:0 \
    --dp_device cuda:1 \
    --seed 42
```


### Experiment Results
Results are saved to:
- `emulation/scripts/results/` - Raw experiment data (JSONL format)
- `emulation/scripts/figures/` - Generated plots (PNG and PDF)

## Directory Structure

The emulation folder is mainly composed of the following sub-folders:

| Directory | Description |
|-----------|-------------|
| `benchmarks/` | Benchmarks for experiments: continuous training pipeline, labeling rule cache, retraining trigger, LINC manager, Helios manager, etc. |
| `datasets/` | Datasets for evaluation (ISCXVPN, USTC-TFC2016, CIC-IDS-2018) |
| `models/` | DNN model checkpoints, LINC rules, and Helios rules |
| `scripts/` | Scripts to run experiments and generate figures |
| `src/` | Core implementation: model architectures, labeling agent, dataset constructors, etc. |
| `logs/` | Training and experiment logs |
