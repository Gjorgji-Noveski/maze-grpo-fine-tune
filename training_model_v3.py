import argparse
import reasoning_gym
import textdistance
import wandb

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig
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
        system_prompt = "You are an expert maze solver. Your job is to output the sequence of directions to reach the goal."

        messages = [
            {"role": "system", "content": system_prompt},
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

    Key insight: We add distance-based shaping so the model gets gradient signal
    even when it doesn't reach the goal. This prevents mode collapse by rewarding
    outputs that get CLOSER to the goal, not just those that reach it.
    # Currently this does 3 things:
    - simualtes path
    - calcualtes reward for how much we've come closer to the goal or if we ran away from it
    -
    """
    step_reward = 0.5
    wall_penalty = -1.0
    reached_end_reward = 10.0

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    rewards = []
    for completion, meta in zip(completions, metadata):
        reached_end = False
        mat = meta['matrix']
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

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
        rows, cols = len(mat), len(mat[0])

        # Calculate initial distance to goal (Manhattan)
        initial_distance = abs(start_row - goal_row) + abs(start_col - goal_col)

        for direction in pred_dirs:
            if direction not in directions:
                wall_hits += 1
                continue

            dr, dc = directions[direction]
            new_row, new_col = current_row + dr, current_col + dc

            if 0 <= new_row < rows and 0 <= new_col < cols and mat[new_row][new_col] != 'X':
                current_row, current_col = new_row, new_col
                valid_steps += 1
                if mat[new_row][new_col] == '#':
                    reached_end = True
                    break
            else:
                wall_hits += 1

        # Calculate final distance to goal
        final_distance = abs(current_row - goal_row) + abs(current_col - goal_col)

        # Distance improvement reward: positive if we got closer, negative if further
        # Scale by initial distance to normalize across different maze sizes
        distance_improvement = initial_distance - final_distance
        if initial_distance > 0:
            distance_reward = 3.0 * (distance_improvement / initial_distance)
        else:
            distance_reward = 0.0

        score = (
            (valid_steps * step_reward) +
            (wall_hits * wall_penalty) +
            distance_reward +
            (reached_end_reward if reached_end else 0.0)
        )
        rewards.append(float(score))

    return rewards


def length_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that compares predicted length to ground truth length.

    - If lengths match: +2 reward
    - If lengths differ: penalty equal to the absolute difference
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        pred_dirs = text.lower().strip().split()
        truth_dirs = ground_truth.lower().strip().split()

        pred_len = len(pred_dirs)
        truth_len = len(truth_dirs)

        if pred_len == truth_len:
            rewards.append(2.0)
        else:
            diff = abs(pred_len - truth_len)
            rewards.append(-float(diff))

    return rewards


def score_answer(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function using Hamming distance. Returns negative distance."""
    rewards = []
    # print(f"prompts: {prompts}\n completions: {completions}\n answer: {answer}\n the rest of kwargs: {kwargs}")
    for completion, ground_truth in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        pred_dirs = text.lower().strip().split()
        truth_dirs = ground_truth.lower().strip().split()
        distance = textdistance.hamming.distance(pred_dirs, truth_dirs)
        rewards.append(-float(distance))
    return rewards


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """Reward well-formed outputs that use valid direction tokens.

    Encourages the model to at least produce syntactically valid outputs
    rather than gibberish or overly short/long sequences.
    """
    valid_directions = {'up', 'down', 'left', 'right'}

    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        tokens = text.lower().strip().split()

        if len(tokens) == 0:
            rewards.append(-2.0)  # Penalize empty output
            continue

        # Count valid direction tokens
        valid_count = sum(1 for t in tokens if t in valid_directions)
        validity_ratio = valid_count / len(tokens)

        # Reward high validity ratio
        format_score = validity_ratio * 5 # range would be [0, 5], depending on how much percentage of directions are correct

        rewards.append(format_score)

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
        texts.append(text.lower().strip())

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
        duplicate_penalty = -0.5 * (seen_counts[text] - 1) / n

        # Small random noise to break exact ties in gradient computation
        noise = random.uniform(-0.01, 0.01)

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
    train_group.add_argument("--max_completion_length", type=int, default=128)
    train_group.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train_group.add_argument("--logging_steps", type=int, default=10)
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
    other_group.add_argument("--model_path", type=str, default="models/gemma_3.1_4B_instruct")
    other_group.add_argument("--output_dir", type=str, default="./output_v3")
    other_group.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    other_group.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    other_group.add_argument("--group_name", type=str, help="Name of the group in wandb")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.no_wandb:
        wandb.init(
            project="maze-grpo",
            name=args.run_name,
            group=args.group_name
        )
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # tok.pad_token = tok.eos_token # TODO: CHECK IF THIS DOES ANYTHING ON THE NEXT RUN, IT SHOULD SAY THAT THE TOKENIZER DOESNT HAVE PADDING TOKEN
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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='mps', dtype='auto')
    print(f'Model device: {model.device}, dtype: {model.dtype}')

    # LoRA config
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        log_completions=True,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        generation_kwargs={"do_sample": True},
        report_to='none' if args.no_wandb else 'wandb'
    )

    trainer = GRPOTrainer(
        args=training_args,
        model=model,
        reward_funcs=[simulate_path, length_reward, format_reward, diversity_reward],
        train_dataset=dataset,
        peft_config=lora_cfg
    )

    # Log config to wandb
    if not args.no_wandb:
        wandb.config.update(vars(args))

    print(f"Trainer device: {trainer.args.device}")
    trainer.train()

    # Save LoRA adapter
    save_path = f"{args.output_dir}/lora"
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"Done. LoRA adapter saved to {save_path}")
