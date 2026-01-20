# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM fine-tuning project for Apple Silicon (MPS) focusing on reinforcement learning from human feedback (RLHF) using GRPO (Group Relative Policy Optimization) training. The primary use case is training models to solve maze navigation tasks using the `reasoning_gym` library.

## Commands

### Environment Setup
```bash
# Uses uv for dependency management
uv sync                           # Install dependencies
source .venv/bin/activate         # Activate virtual environment
```

### Training
```bash
# GRPO training with PyTorch on MPS (main training script)
python training_model_v2.py

# Standard supervised fine-tuning
python pytorch_training.py

# MLX LoRA training (alternative for Apple Silicon)
python mlx-examples/lora/lora.py --model <model_path> --train --iters 600
```

### Notebooks
```bash
jupyter notebook                  # Run notebooks for experimentation
# Main notebook: loading_prompts.ipynb
```

## Architecture

### Training Pipeline
- **training_model_v2.py**: Main GRPO training script using TRL's `GRPOTrainer` with LoRA adapters
- **training_model.py**: Earlier version of GRPO training (reference)
- **pytorch_training.py**: Standard supervised fine-tuning with HuggingFace Trainer
- **loading_prompts.ipynb**: Experimentation notebook for prompt engineering and model testing

### Key Components

**Maze Class** (`training_model_v2.py`): Generates maze navigation prompts from `reasoning_gym` datasets
- Parses grid-based mazes with start/goal positions
- Formats prompts with chat templates for instruction-tuned models
- Supports configurable grid sizes and path distances

**Reward Functions**: Custom reward functions for GRPO training (currently placeholder returning constant rewards)

### Model Storage
- `models/gemma_3.1_4B_instruct/`: Gemma 3.1 4B instruction model (HuggingFace format)
- `models/llama_3.2_3b_instruct_q4/`: Llama 3.2 3B quantized model
- `models/mlx/`: MLX-format models for native Apple Silicon inference
- `output/lora/`: Saved LoRA adapter weights

### Libraries Used
- **TRL**: GRPO trainer for RLHF
- **PEFT**: LoRA adapter configuration
- **Transformers**: Model loading and tokenization
- **MLX-LM**: Alternative inference/training on Apple Silicon
- **reasoning_gym**: Procedural maze generation for training data
- **Weights & Biases**: Experiment tracking (`report_to='wandb'`)

## MPS-Specific Notes

- Models use `device_map='mps'` for Apple Silicon GPU
- Gradient checkpointing enabled to manage memory on unified memory architecture
- Beta parameter set to 0 in GRPO to avoid loading reference model (memory optimization)
- vLLM acceleration not available on MPS (CUDA-only)
