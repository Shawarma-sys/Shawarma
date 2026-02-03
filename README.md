# Artifact for Shawarma

This is the artifact repo for the paper Shawarma.

## Overview

The paper artifact includes Shawarma's software emulation and the FPGA hardware testbed.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `emulation/` | Software emulation for key experiments, including motivation experiments, retrain comparison (Original vs Adaptive), baselines comparison (LINC vs Helios vs Original vs Adaptive), and ablation studies. |
| `testbed/` | FPGA hardware testbed with fixed-point quantization utilities, inference accelerator (FPE), and traffic feature extractor (TFE). |

## Quick Start

### Software Emulation
See [`emulation/README.md`](emulation/README.md) for environment setup, data preparation, model training, and running experiments.

### Hardware Testbed
See [`testbed/readme.md`](testbed/readme.md) for FPGA-based hardware components and quantization tools.

> **Note for AE reviewers:** The `testbed` folder is not required during artifact evaluation since it requires specialized FPGA hardware; however, anyone with the necessary hardware can try out the instructions.