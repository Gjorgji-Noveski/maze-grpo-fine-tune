import argparse
import os
import re
import reasoning_gym
import wandb

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig
from config import SYSTEM_PROMPT, THINK_TAG, ANSWER_TAG

DEFAULT_MODEL_PATH = "models/llama3.2_1B_instruct"
class Maze:
    def __init__(self, tokenizer, dataset_name="shortest_path", min_rows=5, max_rows=8, min_cols=5, max_cols=8, p_blocked=0.4, size=10, seed=None):
        """Initialize maze with dataset parameters"""
        self.tokenizer = tokenizer
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.min_cols = min_cols
        self.max_cols = max_cols
        self.p_blocked = p_blocked
        self.size = size
        self.dataset = reasoning_gym.create_dataset(
            dataset_name,
            min_rows=min_rows,
            max_rows=max_rows,
            min_cols=min_cols,
            max_cols=max_cols,
            p_blocked=p_blocked,
            size=size,
            seed=seed
        )
        self.dataset = self.filter_infeasible()

    def apply_chat_template(self, content):
        """Apply tokenizer chat template to content"""
        content = content.replace("the length of ", "")
        content = content.replace(
            "Your task is to find the shortest path from the start to the destination point in a grid.\n", "")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def filter_infeasible(self):
        """Remove entries with 'infeasible' answers and clean prompts"""
        filtered = []

        for entry in self.dataset:
            if entry['answer'].lower() != 'infeasible':
                question = entry['question']
                start_idx = question.find('If there is no path')
                end_idx = question.find('Your output should be')
                clean_prompt = question[:start_idx] + question[end_idx:]
                entry['prompt'] = self.apply_chat_template(clean_prompt)
                filtered.append(entry)
        return filtered


def simulate_path(prompts, completions, metadata, **kwargs) -> list[float]:
    """Reward function using path simulation through the maze.

    Evaluates path mechanics:
    - valid steps taken
    - wall/boundary hits
    - reaching the goal
    - penalizes moves after reaching goal
    """
    step_reward = 0.5
    wall_penalty = -1.0
    reached_end_reward = 10.0
    overshoot_penalty = -2.0  # Penalty per move after reaching goal

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    rewards = []
    for completion, meta in zip(completions, metadata):
        mat = meta['matrix']
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        # Extract only the final answer for evaluation
        text = extract_answer(text)

        # Find start and goal positions
        start_row, start_col = None, None
        goal_row, goal_col = None, None
        for r, row in enumerate(mat):
            for c, cell in enumerate(row):
                if cell == '*':
                    start_row, start_col = r, c
                elif cell == '#':
                    goal_row, goal_col = r, c

        current_row, current_col = start_row, start_col
        pred_dirs = text.lower().strip().split()

        valid_steps = 0
        wall_hits = 0
        reached_end = False
        moves_after_reaching_end = 0
        rows, cols = len(mat), len(mat[0])

        for direction in pred_dirs:
            if direction not in directions:
                wall_hits += 1
                continue

            dr, dc = directions[direction]
            new_row, new_col = current_row + dr, current_col + dc
            current_row, current_col = new_row, new_col

            # Check validity
            in_bounds = 0 <= new_row < rows and 0 <= new_col < cols
            if in_bounds and mat[new_row][new_col] != 'X':
                valid_steps += 1
            else:
                wall_hits += 1

            # Check if at goal
            if current_row == goal_row and current_col == goal_col:
                reached_end = True
            elif reached_end:
                # Already reached goal but kept moving
                moves_after_reaching_end += 1

        score = (
            (valid_steps * step_reward) +
            (wall_hits * wall_penalty) +
            (reached_end_reward if reached_end else -3.0) +
            (moves_after_reaching_end * overshoot_penalty)
        )
        rewards.append(float(score))

    return rewards


def distance_reward(prompts, completions, metadata, **kwargs) -> list[float]:
    """Reward based on how much closer we got to the goal.

    Returns 0 if no final answer extracted (let format_reward handle that penalty).
    Otherwise, returns positive reward for getting closer, negative for moving away.
    """
    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    rewards = []
    for completion, meta in zip(completions, metadata):
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        text = extract_answer(text)
        pred_dirs = text.lower().strip().split()

        # No answer extracted - return neutral, let format_reward penalize
        if not pred_dirs:
            rewards.append(0.0)
            continue

        mat = meta['matrix']

        # Find start and goal positions
        start_row, start_col = None, None
        goal_row, goal_col = None, None
        for r, row in enumerate(mat):
            for c, cell in enumerate(row):
                if cell == '*':
                    start_row, start_col = r, c
                elif cell == '#':
                    goal_row, goal_col = r, c

        current_row, current_col = start_row, start_col

        # Calculate initial distance to goal (Manhattan)
        initial_distance = abs(start_row - goal_row) + abs(start_col - goal_col)

        # Simulate the path
        for direction in pred_dirs:
            if direction not in directions:
                continue
            dr, dc = directions[direction]
            new_row, new_col = current_row + dr, current_col + dc
            current_row, current_col = new_row, new_col

        # Calculate final distance to goal
        final_distance = abs(current_row - goal_row) + abs(current_col - goal_col)

        # Distance improvement reward: positive if closer, negative if further
        distance_improvement = initial_distance - final_distance
        reward = 3.0 * (distance_improvement / initial_distance)

        rewards.append(float(reward))

    return rewards


def length_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that compares predicted length to ground truth length.

    - If lengths match: +1 reward
    - If lengths differ: penalty equal to the absolute difference
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        # Extract only the final answer for evaluation
        text = extract_answer(text)

        pred_dirs = text.lower().strip().split()
        truth_dirs = ground_truth.lower().strip().split()
        pred_len = len(pred_dirs)
        truth_len = len(truth_dirs)

        diff = abs(pred_len - truth_len)

        if pred_len == truth_len:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def extract_answer(text: str) -> str:
    """Extract text from answer tags defined by ANSWER_TAG constant.

    Returns the content inside the tags, or empty string if not found.
    """
    match = re.search(rf'<{ANSWER_TAG}>(.*?)</{ANSWER_TAG}>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ''

def no_answer_reward(completions, **kwargs) -> list[float]:
    """Penalize completions that have no extractable answer."""
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        answer = extract_answer(text)
        if len(answer.strip()) == 0:
            rewards.append(-5.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """Reward proper tag structure in outputs.

    Checks for <think> and <answer> tags.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        count = 0.0
        if re.search(rf"\s*<{THINK_TAG}>\s*", text):
            count += 0.25
        if re.search(rf"\s*</{THINK_TAG}>\s*", text):
            count += 0.25
        if re.search(rf"\s*<{ANSWER_TAG}>\s*", text):
            count += 0.25
        if re.search(rf"\s*</{ANSWER_TAG}>\s*", text):
            count += 0.25
        rewards.append(count)

    return rewards

def got_to_end_reward(completions, metadata, **kwargs) -> list[float]:
    reached_end_reward = 2.0

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    rewards = []
    for completion, meta in zip(completions, metadata):
        mat = meta['matrix']
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        # Extract only the final answer for evaluation
        text = extract_answer(text)

        # Find start and goal positions
        start_row, start_col = None, None
        goal_row, goal_col = None, None
        for r, row in enumerate(mat):
            for c, cell in enumerate(row):
                if cell == '*':
                    start_row, start_col = r, c
                elif cell == '#':
                    goal_row, goal_col = r, c

        current_row, current_col = start_row, start_col
        pred_dirs = text.lower().strip().split()

        reached_end = False

        for direction in pred_dirs:
            if direction not in directions:
                continue

            dr, dc = directions[direction]
            new_row, new_col = current_row + dr, current_col + dc
            current_row, current_col = new_row, new_col

        # Check if at goal
        if current_row == goal_row and current_col == goal_col:
            reached_end = True

        score = reached_end_reward if reached_end else 0.0
        rewards.append(float(score))

    return rewards


def binary_got_closer(completions, metadata, **kwargs) -> list[float]:
    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    rewards = []
    for completion, meta in zip(completions, metadata):
        mat = meta['matrix']
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        text = extract_answer(text)

        start_row, start_col = None, None
        goal_row, goal_col = None, None
        for r, row in enumerate(mat):
            for c, cell in enumerate(row):
                if cell == '*':
                    start_row, start_col = r, c
                elif cell == '#':
                    goal_row, goal_col = r, c

        initial_dist = abs(start_row - goal_row) + abs(start_col - goal_col)

        current_row, current_col = start_row, start_col
        pred_dirs = text.lower().strip().split()

        for direction in pred_dirs:
            if direction not in directions:
                continue
            dr, dc = directions[direction]
            new_row, new_col = current_row + dr, current_col + dc
            current_row, current_col = new_row, new_col

        final_dist = abs(current_row - goal_row) + abs(current_col - goal_col)
        score = 0.5 if final_dist < initial_dist else 0.0
        rewards.append(float(score))

    return rewards


def validity_reward(prompts, completions, **kwargs) -> list[float]:
    """Reward valid direction tokens in final answer.

    Returns 0 if no answer extracted, otherwise rewards based on
    ratio of valid directions (up, down, left, right) to total tokens.
    """
    valid_directions = {'up', 'down', 'left', 'right'}

    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        final_answer = extract_answer(text)
        tokens = final_answer.lower().strip().split()

        # No answer extracted - return neutral
        if not tokens:
            rewards.append(0.0)
            continue

        valid_count = sum(1 for t in tokens if t in valid_directions)
        validity_ratio = valid_count / len(tokens)

        # Reward high validity ratio (max 3)
        rewards.append(validity_ratio * 3.0)

    return rewards


def diversity_reward(prompts, completions, **kwargs) -> list[float]:
    """Reward unique completions, penalize identical ones to prevent mode collapse.

    Uses random tiebreaking so identical completions get different rewards,
    creating gradient signal even when all outputs collapse to the same text.
    """
    import random
    from collections import Counter

    texts = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)
        # Extract only the final answer for diversity comparison
        final = extract_answer(text)
        texts.append(final.lower().strip())

    counts = Counter(texts)
    n = len(texts)

    # Track how many of each duplicate we've seen for tiebreaking
    seen_counts = {}

    rewards = []
    for text in texts:
        frequency = counts[text] / n

        # Base penalty for duplicates
        base_reward = 1.0 - (2.0 * frequency)

        # Add decreasing reward for each duplicate instance
        # First occurrence gets base, subsequent ones get progressively worse
        if text not in seen_counts:
            seen_counts[text] = 0
        seen_counts[text] += 1

        # Tiebreaker: penalize later duplicates more heavily
        duplicate_penalty = -1.0 * (seen_counts[text] - 1) / n

        # Small random noise to break exact ties in gradient computation
        noise = random.uniform(-0.02, 0.02)

        rewards.append(base_reward + duplicate_penalty + noise)

    return rewards

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for maze navigation")

    # Maze parameters
    maze_group = parser.add_argument_group("Maze")
    maze_group.add_argument("--min_rows", type=int, default=5)
    maze_group.add_argument("--max_rows", type=int, default=7)
    maze_group.add_argument("--min_cols", type=int, default=5)
    maze_group.add_argument("--max_cols", type=int, default=7)
    maze_group.add_argument("--p_blocked", type=float, default=0.3, help="Probability of blocked cells")
    maze_group.add_argument("--dataset_size", type=int, default=10, help="Number of maze samples")
    maze_group.add_argument("--seed", type=int, default=42)

    # Training parameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--max_steps", type=int, default=20)
    train_group.add_argument("--num_generations", type=int, default=8, help="Completions per prompt for GRPO")
    train_group.add_argument("--generation_batch_size", type=int, default=8, help="Completions per forward pass (must be divisible by num_generations)")
    train_group.add_argument("--max_completion_length", type=int, default=512, help="Max tokens for generation (increased for scratchpad reasoning)")
    train_group.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train_group.add_argument("--logging_steps", type=int, default=10)
    train_group.add_argument("--save_steps", type=int, default=20, help="Save checkpoint every N steps")
    train_group.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    train_group.add_argument("--learning_rate", type=float, default=5e-5)
    train_group.add_argument("--beta", type=float, default=0.0, help="KL divergence coefficient (0=no ref model)")
    train_group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for diverse completions")
    train_group.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for generation"
    )
    # LoRA parameters
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    lora_group.add_argument("--lora_alpha", type=int, default=16)
    lora_group.add_argument("--lora_dropout", type=float, default=0.05)

    # Other
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    other_group.add_argument("--output_dir", type=str, default="./output")
    other_group.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    other_group.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    other_group.add_argument("--group_name", type=str, help="Name of the group in wandb")
    other_group.add_argument("--wandb_run_id", type=str, default=None, help="Wandb run ID to resume (find in wandb URL or run overview)")
    other_group.add_argument("--full_fine_tune", action="store_true",  help="If full fine tuning should be performed")
    other_group.add_argument("--reward_set", type=int, default=None, choices=[1, 2, 3],
                             help="Reward function set: 1=[got_to_end], 2=[got_to_end, format], 3=[got_to_end, format, binary_got_closer]")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.no_wandb:
        if args.resume_from_checkpoint and not args.wandb_run_id:
            print("\n" + "=" * 60)
            print("WARNING: RESUMING FROM CHECKPOINT WITHOUT WANDB RUN ID")
            print("Wandb will create a NEW run with fresh graphs!")
            print("To continue existing graphs, pass --wandb_run_id <id>")
            print("=" * 60 + "\n")
        wandb.init(
            project="maze-grpo",
            name=args.run_name,
            group=args.group_name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None
        )
        # Override args with sweep params if running under a wandb sweep
        sweep_config = dict(wandb.config)
        if "learning_rate" in sweep_config:
            args.learning_rate = sweep_config["learning_rate"]
        if "reward_set" in sweep_config:
            args.reward_set = int(sweep_config["reward_set"])
        if not args.run_name:
            args.run_name = f"lr{args.learning_rate}_rewards{args.reward_set}"
            wandb.run.name = args.run_name
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tok.pad_token = tok.eos_token
    maze = Maze(
        tokenizer=tok,
        size=args.dataset_size,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_cols=args.min_cols,
        max_cols=args.max_cols,
        p_blocked=args.p_blocked,
        seed=args.seed
    )
    dataset = Dataset.from_list(maze.dataset)
    from reasoning_gym.dataset import ProceduralDataset
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='mps', dtype='float16')
    print(f'Model device: {model.device}, dtype: {model.dtype}')

    # LoRA config
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    checkpoint_path = args.resume_from_checkpoint

    args.output_dir = os.path.join("output", args.run_name)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,  # Must be divisible by num_generations
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        log_completions=True,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        generation_kwargs={"do_sample": True},
        report_to='none' if args.no_wandb else 'wandb'
    )

    trainer = GRPOTrainer(
        args=training_args,
        model=model,
        reward_funcs={
            1: [got_to_end_reward],
            2: [got_to_end_reward, format_reward],
            3: [got_to_end_reward, format_reward, binary_got_closer],
        }.get(args.reward_set, [got_to_end_reward, format_reward, binary_got_closer]),
        train_dataset=dataset,
        peft_config= None if args.full_fine_tune else lora_cfg
    )

    # Log config to wandb
    if not args.no_wandb:
        wandb.config.update(vars(args), allow_val_change=True)
        wandb.config.update(training_args.to_sanitized_dict(), allow_val_change=True)

    print(f"Trainer device: {trainer.args.device}")
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # Save LoRA adapter
    save_path = f"{args.output_dir}/lora"
    trainer.model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"Done. LoRA adapter saved to {save_path}")
