import reasoning_gym
import re

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM)

from peft import LoraConfig

class Maze:
    def __init__(self, tokenizer, dataset_name="maze", min_dist=3, max_dist=10, size=2, min_grid_size=5, max_grid_size=5, seed=None):
        """Initialize maze with dataset parameters"""
        self.tokenizer = tokenizer
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.size = size
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
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
        grid_world_prompt += """\nOutput ONLY the steps needed to go from the start position to the goal position. Use ONLY the words (up, down, left, right) to solve the maze.\n<example_output>\nup, right, up, up, left, up\n</example_output>"""

        messages = [{"role": "user", "content": grid_world_prompt}]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Adds the assistant turn start
        )
        return formatted_prompt

    def parse_maze_grid(self, question):
        """Parse the maze grid and return grid, start position, and goal position"""
        # Extract the grid from between ``` markers
        grid_match = re.search(r'```\n(.*?)\n```', question, re.DOTALL)
        if not grid_match:
            return None, None, None

        grid_text = grid_match.group(1)
        grid = [list(row) for row in grid_text.strip().split('\n')]

        start_pos = None
        goal_pos = None

        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                if cell == '(':
                    start_pos = (row_idx, col_idx)
                elif cell == 'g':
                    goal_pos = (row_idx, col_idx)

        return grid, start_pos, goal_pos

    def get_dataset_with_metadata(self):
        """Returns dataset with prompts and parsed maze metadata"""
        data_list = []

        for maze_idx, maze_data in enumerate(self.dataset):
            question = maze_data['question']
            grid, start_pos, goal_pos = self.parse_maze_grid(question)

            # Serialize grid as string for dataset storage
            grid_str = '\n'.join([''.join(row) for row in grid]) if grid else ""

            data_list.append({
                "prompt": self.get_maze_world_prompt(maze_idx),
                "grid": grid_str,
                "start_row": start_pos[0] if start_pos else -1,
                "start_col": start_pos[1] if start_pos else -1,
                "goal_row": goal_pos[0] if goal_pos else -1,
                "goal_col": goal_pos[1] if goal_pos else -1,
            })

        return Dataset.from_list(data_list)

    def get_prompts(self):
        """Returns all prompts in list format"""
        return [self.get_maze_world_prompt(maze_idx) for maze_idx, _ in enumerate(self.dataset)]

# Usage
# Select model
model_name_path = "models/gemma_3.1_4B_instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = 'cpu'
# Load tokenizer
tok = AutoTokenizer.from_pretrained(model_name_path, use_fast=True)
tok.pad_token = tok.eos_token

maze = Maze(tokenizer=tok, size=10, min_dist=2, max_dist=5, min_grid_size=4, max_grid_size=7, seed=1)

mazes = maze.get_prompts()
mazes_dicts = [{"prompt": maze_prompt} for maze_prompt in mazes]
dataset = Dataset.from_list(mazes_dicts)

def format_reward_func(prompts, completions: list[list[dict[str, str]]], **kwargs) -> list[float | None]:
    rewards = []
    for i in kwargs:
        print(f' key: {i}, value: {kwargs[i]}')
    for content in completions:
        rewards.append(1)

    return rewards


# Load model
model = AutoModelForCausalLM.from_pretrained(model_name_path, device_map='mps', dtype='auto')
print(f'model"s device {model.device}')
print(f'next model device: {next(model.parameters()).device}')
print(f'dtype: {model.dtype}')

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

training_args = GRPOConfig(
    output_dir="./output",
    beta=0, # if > 0 it would load a second model needed for calculating KL divergence
    num_generations=2, # default is 8
    max_completion_length=128, # default is 256
    per_device_train_batch_size=2, # default is 8
    gradient_checkpointing=True,
    log_completions=True,
    logging_steps=1,
    gradient_accumulation_steps=4,
    max_steps=20,
    report_to='wandb'

    # Numerical stability fixes
    # bf16=True,
    # fp16=True,  # Use float32 for MPS stability
    # use_vllm=True,
    # vllm_mode='colocate' # vLLM is built for CUDA, so tweaking will need to be done for this to work on Mac
)
trainer = GRPOTrainer(
    args=training_args,
    model=model,
    reward_funcs=my_stuff,
    train_dataset=dataset,
    peft_config=lora_cfg
)
# Check where model parameters actually are
print(f"Model device: {next(model.parameters()).device}")
print(f'dtype: {model.dtype}')
# After creating trainer, you can also check:
print(f"Trainer device: {trainer.args.device}")
trainer.train()
# Save LoRA adapter
model.save_pretrained("output/lora")
tok.save_pretrained("output/lora")
print("Done. LoRA adapter saved to out/lora")
