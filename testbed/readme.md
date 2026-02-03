# Shawarma Hardware Testbed

## Overview

This directory hosts the Shawarma FPGA hardware testbed, including:

- **Fixed-point quantization tools** for hardware-friendly neural network deployment
- **Inference accelerator (FPE)** Verilog implementation for low-latency GEMV computation
- **Traffic feature extractor (TFE)** Verilog implementation for real-time packet processing

## Directory Structure

```
testbed/
├── Fixed_Quant/1.1/    # Fix-8 quantization and QAT tools
├── FPE/                # Fast Process Engine (inference accelerator)
└── TFE/                # Traffic Feature Extractor
```

## Components

### Fixed_Quant/1.1

Hardware-oriented Fix-8 quantization and quantization-aware training (QAT). The Fix-8 format uses 1 sign bit, 2 integer bits, and 5 fractional bits, with a representable range of $[-4.0, 3.9675]$ and a minimum increment of $0.03125$.

**Requirements:**
```
PyTorch >= 1.12.1
```

**Files:**
| File | Description |
|------|-------------|
| `fully_fix_linear.py` | Input tensor quantization (`fix_x`) and fully fixed-point linear layer implementation |
| `fully_fix_MLP.py` | Example fixed-point MLP model with `fp_forward`, `fixed_model`, and `fixed_fwd` methods |
| `train_MLP_fix.py` | Fix-8 QAT training (`fix_train_op`) and evaluation (`fixed_fwd_op`) workflows |
| `README.txt` | Additional usage notes |

### FPE/

Verilog source for the inference accelerator (Fast Process Engine). This module targets GEMV computation and supports low-latency inference-path validation on FPGA.

**Architecture:**
```
VPE_top.v              // FPE top module
├── VPE_Ctrler.v       // Controller: instruction decoding and issuing
├── TF_fetcher.v       // Traffic fetcher for mirrored packets
├── Vector_Regfile.v   // Vector register file
├── SIMD_in_Reg.v      // SIMD input register file
├── VPE_Weights_ROM.v  // On-chip memory for NN parameters (BRAM on FPGA)
├── SIMD.v             // Core SIMD computing unit
│   └── Lane.v         // SIMD lane
│       └── Dot_PE.v   // Dot-product processing element
├── VPE_Vector_Adder.v // Inline accumulator
├── VPE_ReLU.v         // ReLU activation module
├── Bias_Adder.v       // Bias addition with overflow handling
└── VPE_Bias_ROM.v     // Bias parameter storage
```

### TFE/

Verilog source for the Traffic Feature Extractor. It uses a pipelined and shared-memory design for real-time flow tracking and feature extraction.

**Architecture:**
```
TFE_top.v                      // TFE top module
├── Wine_Dispenser.v           // Packet parser: splits header and payload
├── Hashing.v                  // CRC32-based hash function
│   └── CRC32_D104.v           // CRC32 implementation
├── Meta_Gen.v                 // Metadata generator (packet arrival time, etc.)
└── TFE_kernel.v               // TFE kernel
    ├── flow_tracker.v         // Flow state tracking
    │   └── tracker.v          // State tracking implementation
    ├── Extreme_Val_Memory_Module.v  // Shared feature cache
    ├── buffer2alu.v           // Data buffer between memory and ALU
    └── ALU_Cluster.v          // Feature operator integration
        ├── Max_Min_Alu.v      // Max/min feature extractor
        ├── Average_Val_Unit.v // Averaging feature extractor
        └── Vec_Feature_Unit.v // Time-series feature extractor
```

## Notes

This testbed is designed for FPGA-based hardware evaluation. For software emulation experiments, refer to the [`emulation/`](../emulation/) directory.