<h1 align="center">LLM Fine-Tune: Maze Navigation via GRPO</h1>

<p align="center">
  <img src="maze.gif" alt="maze" /><br/>
  Can an LLM learn to navigate a grid maze? This repo aims to do that!
</p>

Fine-tune an LLM to navigate 2D grid mazes, moving from a start position (`*`) to a goal (`#`) while avoiding walls (`X`). The entry point is `train.py`, invokable via CLI with configurable parameters. Training metrics are logged to [Weights and Biases](https://wandb.ai/site/). Training uses [TRL's GRPOTrainer](https://huggingface.co/docs/trl) with optional LoRA adapters (via [PEFT](https://huggingface.co/docs/peft/en/index)).

## Project Structure

```
llm_fine_tune/
├── src/llm_fine_tune/
│   ├── train.py          # GRPO training entry point
│   ├── evaluate.py       # Evaluation on held-out mazes
│   ├── dataset.py        # Maze dataset creation and prompt formatting
│   ├── rewards.py        # Reward functions for GRPO
│   └── utils/
│       ├── config.py                          # System prompt and tag constants
│       ├── utils.py                           # Shared helpers (answer extraction, grid parsing)
│       └── plot_direction_sequence_distribution.py  # Direction distribution plotting
├── wandb_configs/
│   ├── sweep_lora.yaml             # W&B sweep config for LoRA fine-tuning
│   └── sweep_full_fine_tune.yaml   # W&B sweep config for full fine-tuning
├── models/                  # Local model weights (downloaded separately)
├── output/                  # Training checkpoints and LoRA adapters
├── eval_results/            # Evaluation output JSON files
└── pyproject.toml           # Project dependencies and build config
```

## Setup
This project requires **Python 3.12.12**. Before running training, make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, then create a virtual environment and install dependencies:

```bash
uv venv
uv sync
```

An `.env` file is already supplied in the project root:
```
DEFAULT_MODEL_PATH=/path/to/your/local/model
DEVICE=mps  # mps | cuda | cpu
```

Download a model into the `models/` folder using the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (some models require access approval):

```bash
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir models/
```
This project was developed and tested with Llama 3.2 1B Instruct.

### Metrics tracking with Weights & Biases
I used Weights & Biases in this project for experiment tracking because the training framework (TRL) integrates with it directly. The current free plan of Weights & Biases offers plenty of space to upload experiment metrics.
To enable logging of fine-tuning metrics, create a W&B account and authenticate:

```bash
wandb login
```

## Maze Format

Mazes are procedurally generated using [Reasoning Gym](https://github.com/open-thought/reasoning-gym), with size controlled by the training parameters you specify. The grid uses the following symbols:
- `*` - start position
- `#` - goal position
- `X` - wall
- `O` - open cell

Example 3x3 maze:
```
O X *
O O #
X O O
```

## Reward Functions

| Function            | Type                                        | Score                      | Purpose                                                                                                                            |
|---------------------|---------------------------------------------|----------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `got_to_end_reward` | Binary                                      | +2 or 0                    | Did the path reach the goal? Foundation of every reward set.                                                                       |
| `format_reward`     | Structured                                  | Up to +1.0 (+0.25 per tag) | Rewards correct `<think>`, `</think>`, `<answer>`, `</answer>` structure. Modified later to require *exactly one* of each.         |
| `binary_got_closer` | Binary                                      | +0.5 or 0                  | Did the path end closer to the goal than the start? Soft nudge in the right direction (Manhattan distance)                         |
| `distance_reward`   | Proportional version of `binary_got_closer` | Up to +3.0                 | `3.0 × (distance_improvement / initial_distance)` - more reward the closer you get.                                                |
| `simulate_path`     | Granular                                    | Variable                   | Step-by-step simulation: +0.5 per valid step, −1.0 per wall hit, +10.0 for reaching goal, −2.0 per extra move after reaching goal. |
| `length_reward`     | Binary                                      | +1 or 0                    | Does the number of directions outputted match the ground truth shortest path?                                                      |
| `validity_reward`   | Proportional                                | Up to +3.0                 | Fraction of output tokens that are valid directions (`up`, `down`, `left`, `right`).                                               |
| `no_answer_reward`  | Penalty                                     | −5.0                       | Fires when no answer is extractable at all.                                                                                        |
| `diversity_reward`  | Regularizer                                 | Variable                   | Penalizes identical completions across a GRPO batch to maintain gradient signal.                                                   |


## Training
Training can be launched in two ways:
1. **Single run with a reward set** - pick a predefined combination of reward functions and train directly.
2. **W&B sweep** - define a hyperparameter search in a YAML config (`wandb_configs/`). Two configs are provided: one for LoRA and one for full fine-tuning.

### Single run
Example command:
```bash
python src/llm_fine_tune/train.py --min_rows 3 --max_rows 3 --min_cols 3 --max_cols 3 --dataset_size 5000 --max_steps 500 --run_name example_run_name --group_name small_maze_sizes --temperature 0.6 --logging_steps 5 --save_steps 300 --learning_rate 1e-8 --max_completion_length 600 --num_generations 6 --generation_batch_size 6 --reward_set 3
```

Three reward sets are available via `--reward_set` (defined in `train.py`):
- `1`: `[got_to_end]`
- `2`: `[got_to_end, format]`
- `3`: `[got_to_end, format, binary_got_closer]`


Starting with fewer reward functions gives you more control and makes it easier to catch reward hacking early. Additional reward functions are available in `rewards.py`.

*If you wish to experiment with specific reward functions, or just to try out different combinations, feel free to mix and match as you please!*

### W&B sweeps

To launch a hyperparameter search, first register the sweep:

```bash
wandb sweep wandb_configs/<sweep_name>.yaml
```

This outputs a sweep ID along with the command to start the agent:

```bash
wandb agent <user>/<project>/<sweep_id>
```

Currently, the sweep only supports the parameters defined in the sweep yaml files (`wandb_configs/sweep_lora.yaml`, `wandb_configs/sweep_full_fine_tune.yaml`). The sweepable parameters are `learning_rate`, `reward_set`, and `temperature`. Any other training arguments must be set as fixed values in the yaml `command` section.
Feel free to modify the code if you wish to explore other parameters.

### Notable arguments

| Argument                   | Default               | Description                                          |
|----------------------------|-----------------------|------------------------------------------------------|
| `--model_path`             | `$DEFAULT_MODEL_PATH` | Path to base model                                   |
| `--reward_set`             | 1                     | Reward function set (1, 2, or 3)                     |
| `--full_fine_tune`         | false                 | Skip LoRA, fine-tune all weights                     |
| `--no_wandb`               | false                 | Disable W&B logging                                  |
| `--resume_from_checkpoint` | None                  | Path to checkpoint directory to resume training from |
| `--wandb_run_id`           | None                  | W&B run ID to resume logging into an existing run    |

For a full list of arguments, run `python src/llm_fine_tune/train.py --help`.

## Evaluation
Testing of the trained model can be performed on a held-out maze dataset. To generate the held-out maze datasets, the seed 12345 is used. For training, the seed [42](https://www.youtube.com/watch?v=aboZctrHfK8) is used.

*If you change the seed, ensure the training and evaluation seeds are different!*

Example:
```bash
python src/llm_fine_tune/evaluate.py \
  --model_path /path/to/base/model \
  --lora_path ./output/my_run/lora \
  --eval_size 100 \
  --min_rows 5 --max_rows 7 \
  --min_cols 5 --max_cols 7 \
  --verbose
```

`--lora_path` is optional, omit it to evaluate the base model without any adapter. When provided, it should point to a directory containing an `adapter_config.json` file, which is automatically created under `<output_dir>/lora` when performing a `train.py` run.

Results are saved to `eval_results/eval_<timestamp>.json` and include:
- Goal-reach accuracy
- Average final Manhattan distance to the goal
- Average valid steps and wall hits

## Hardware

The project has been tested on Apple Silicon (MPS). CUDA support is untested but should work. Update the `DEVICE` environment variable accordingly. The model is loaded in `float16`.
