import reasoning_gym

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM)

from peft import LoraConfig

class Maze:
    def __init__(self, dataset_name="maze", min_dist=3, max_dist=10, size=2, min_grid_size=5, max_grid_size=5, seed=None):
        """Initialize maze with dataset parameters"""
        self.dataset = list(reasoning_gym.create_dataset(
            dataset_name,
            min_dist=min_dist,
            max_dist=max_dist,
            size=size,
            min_grid_size=min_grid_size,
            max_grid_size=max_grid_size,
            seed=seed
        ))

    def get_maze_world_prompt(self, index):
        """Return only the maze visualization text without the question"""
        question = self.dataset[index]['question']
        end_of_grid_text = question.index("What")

        grid_world_prompt = question[:end_of_grid_text].strip()

        return grid_world_prompt

    def get_prompts(self):
        """Returns all prompts in list format"""
        return [self.get_maze_world_prompt(maze_idx) for maze_idx, _ in enumerate(self.dataset)]

# Usage
maze = Maze(size=10, min_dist=2, max_dist=5, min_grid_size=4, max_grid_size=7, seed=1)

mazes = maze.get_prompts()
mazes_dicts = [{"prompt": maze_prompt} for maze_prompt in mazes]
dataset = Dataset.from_list(mazes_dicts)

# dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
#
# print()

def my_stuff(completions: list[list[dict[str, str]]], **kwargs) -> list[float | None]:
    rewards = []
    for content in completions:
        rewards.append(1)

    return rewards

# Select model
model_name_path = "models/gemma_3.1_4B_instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
# device = 'cpu'
# Load tokenizer
tok = AutoTokenizer.from_pretrained(model_name_path, use_fast=True)
tok.pad_token = tok.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name_path, device_map="mps")
# model.to(device)

# Memory optimization for MPS
# model.config.use_cache = False
# if hasattr(model, "gradient_checkpointing_enable"):
#     model.gradient_checkpointing_enable()


# LoRA config
lora_cfg = LoraConfig(
r=8, lora_alpha=16, lora_dropout=0.05,
target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
task_type="CAUSAL_LM"
)
# model = get_peft_model(model, lora_cfg)
#
# training_args = GRPOConfig(
#     output_dir="./output",
#     # Numerical stability fixes
#     bf16=False,
#     fp16=True,  # Use float32 for MPS stability
# )
trainer = GRPOTrainer(
    model=model,
    reward_funcs=my_stuff,
    train_dataset=dataset,
    peft_config=lora_cfg

)
# Check where model parameters actually are
print(f"Model device: {next(model.parameters()).device}")

# After creating trainer, you can also check:
print(f"Trainer device: {trainer.args.device}")
# trainer.train()
# # Save LoRA adapter
# model.save_pretrained("output/lora")
# tok.save_pretrained("output/lora")
# print("Done. LoRA adapter saved to out/lora")
