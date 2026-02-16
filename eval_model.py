"""Evaluation script for maze-solving model.

Generates completions on a held-out maze dataset and reports accuracy.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import reasoning_gym
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import SYSTEM_PROMPT, ANSWER_TAG


def extract_final_answer(text: str) -> str:
    """Extract text from answer tags."""
    match = re.search(rf'<{ANSWER_TAG}>(.*?)</{ANSWER_TAG}>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ''


def create_eval_dataset(tokenizer, size, min_rows, max_rows, min_cols, max_cols, p_blocked, seed):
    """Create evaluation maze dataset.

    Returns:
        tuple: (filtered_dataset, original_size, filtered_size)
    """
    dataset = reasoning_gym.create_dataset(
        "shortest_path",
        min_rows=min_rows,
        max_rows=max_rows,
        min_cols=min_cols,
        max_cols=max_cols,
        p_blocked=p_blocked,
        size=size,
        seed=seed
    )

    original_size = len(dataset)

    # Filter infeasible and format prompts
    filtered = []
    for entry in dataset:
        if entry['answer'].lower() != 'infeasible':
            question = entry['question']
            start_idx = question.find('If there is no path')
            end_idx = question.find('Your output should be')
            clean_prompt = question[:start_idx] + question[end_idx:]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clean_prompt}
            ]
            entry['prompt'] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            filtered.append(entry)

    return filtered, original_size, len(filtered)


def simulate_and_check(completion: str, metadata: dict) -> dict:
    """Simulate path and check if it reaches the goal.

    Returns dict with:
        - reached_goal: bool
        - valid_steps: int
        - wall_hits: int
        - final_distance: int (Manhattan distance to goal)
    """
    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    mat = metadata['matrix']
    text = extract_final_answer(completion)

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

    reached_goal = (current_row == goal_row and current_col == goal_col)
    final_distance = abs(current_row - goal_row) + abs(current_col - goal_col)

    # Calculate percentage of correct directions
    ground_truth_dirs = metadata.get('solution', [])
    correct_dirs = 0
    compare_length = min(len(pred_dirs), len(ground_truth_dirs))
    for i in range(compare_length):
        if pred_dirs[i] == ground_truth_dirs[i].lower():
            correct_dirs += 1

    # Use ground truth length as denominator (what percentage of the solution did we get right)
    direction_accuracy = (correct_dirs / len(ground_truth_dirs) * 100) if ground_truth_dirs else 0.0

    return {
        'reached_goal': reached_goal,
        'valid_steps': valid_steps,
        'wall_hits': wall_hits,
        'final_distance': final_distance,
        'predicted_path': text,
        'ground_truth': ' '.join(ground_truth_dirs),
        'completion': completion,
        'direction_accuracy': direction_accuracy,
    }


def evaluate(model, tokenizer, dataset, max_new_tokens, temperature, device):
    """Run evaluation on dataset and return metrics."""
    results = []

    for entry in tqdm(dataset, desc="Evaluating"):
        prompt = entry['prompt']

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Simulate and check
        result = simulate_and_check(completion, entry['metadata'])
        results.append(result)

    return results


def print_results(results, verbose=False):
    """Print evaluation metrics."""
    total = len(results)
    correct = sum(1 for r in results if r['reached_goal'])
    accuracy = (correct / total) * 100 if total > 0 else 0

    avg_distance = sum(r['final_distance'] for r in results) / total if total > 0 else 0
    avg_valid_steps = sum(r['valid_steps'] for r in results) / total if total > 0 else 0
    avg_wall_hits = sum(r['wall_hits'] for r in results) / total if total > 0 else 0
    # TODO: Remove or fix this, it takes the min(len) between predicted and ground truth when calculating
    # TODO: this metric, for instance: pred: right right down, TRUTH: right right---> this would give 100% direction accuracy
    # avg_direction_accuracy = sum(r['direction_accuracy'] for r in results) / total if total > 0 else 0

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy (reached goal): {accuracy:.1f}% ({correct}/{total})")
    # print(f"Avg direction accuracy: {avg_direction_accuracy:.1f}%")
    print(f"Avg final distance to goal: {avg_distance:.2f}")
    print(f"Avg valid steps: {avg_valid_steps:.2f}")
    print(f"Avg wall hits: {avg_wall_hits:.2f}")
    print("=" * 50)

    if verbose:
        print("\nDetailed results:")
        for i, r in enumerate(results):
            status = "+" if r['reached_goal'] else "x"
            print(f"\n[{i+1}] {status} Distance: {r['final_distance']}, "
                  f"Valid: {r['valid_steps']}, Walls: {r['wall_hits']}")
            print(f"    Predicted: {r['predicted_path']}")
            print(f"    Ground truth: {r['ground_truth']}")

    return {
        "accuracy": accuracy,
        # 'avg_direction_accuracy': avg_direction_accuracy,
        "avg_distance": avg_distance,
        "avg_valid_steps": avg_valid_steps,
        "avg_wall_hits": avg_wall_hits,
        "total": total,
        "correct": correct,
    }


def save_results_json(metrics, args, original_size, filtered_size, results, output_path):
    """Save evaluation results to JSON file."""
    output = {
        'datetime': datetime.now().isoformat(),
        'dataset': {
            'original_size': original_size,
            'filtered_size': filtered_size,
            'seed': args.seed,
            'min_rows': args.min_rows,
            'max_rows': args.max_rows,
            'min_cols': args.min_cols,
            'max_cols': args.max_cols,
            'p_blocked': args.p_blocked,
        },
        'model': {
            'model_path': args.model_path,
            'lora_path': args.lora_path,
        },
        'generation': {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
        },
        'metrics': metrics,
        'per_sample_results': results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate maze-solving model")

    # Model
    parser.add_argument("--model_path", type=str, default="models/llama3.2_1B_instruct",
                        help="Base model path")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter (optional)")

    # Dataset
    parser.add_argument("--eval_size", type=int, default=50,
                        help="Number of evaluation samples")
    parser.add_argument("--min_rows", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=7)
    parser.add_argument("--min_cols", type=int, default=5)
    parser.add_argument("--max_cols", type=int, default=7)
    parser.add_argument("--p_blocked", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=12345,
                        help="Seed for eval dataset (use different from training!)")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 for greedy decoding")

    # Output
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results for each sample")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results JSON (default: eval_results/eval_YYYYMMDD_HHMMSS.json)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        dtype="auto"
    )

    # Load LoRA adapter if specified
    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    print(f"Model loaded. Device: {model.device}, dtype: {model.dtype}")

    # Create eval dataset
    print(f"\nCreating eval dataset with seed={args.seed}, size={args.eval_size}...")
    eval_dataset, original_size, filtered_size = create_eval_dataset(
        tokenizer,
        size=args.eval_size,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_cols=args.min_cols,
        max_cols=args.max_cols,
        p_blocked=args.p_blocked,
        seed=args.seed
    )
    print(f"Eval dataset size: {filtered_size} (filtered from {original_size})")

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate(
        model, tokenizer, eval_dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device
    )

    # Print results
    metrics = print_results(results, verbose=args.verbose)

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_json:
        # Insert timestamp before .json extension
        output_path = Path(args.output_json)
        json_path = (
            output_path.parent
            / f"eval_results/{output_path.stem}_{timestamp}{output_path.suffix}"
        )
    else:
        json_path = f"eval_results/eval_{timestamp}.json"

    save_results_json(metrics, args, original_size, filtered_size, results, json_path)
