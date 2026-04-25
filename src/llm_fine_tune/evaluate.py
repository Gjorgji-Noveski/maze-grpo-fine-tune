"""Evaluation script for maze-solving model.

Generates completions on a held-out maze dataset and reports accuracy.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

from llm_fine_tune.dataset import load_maze_dataset
from llm_fine_tune.rewards import _simulate_directions
from llm_fine_tune.utils.utils import extract_answer, find_starting_and_goal_positions

load_dotenv()


def simulate_and_check(completion: str, metadata: dict) -> dict:
    """Simulate path and check if it reaches the goal.

    Returns dict with:
        - reached_goal: bool
        - valid_steps: int
        - wall_hits: int
        - final_distance: int (Manhattan distance to goal)
    """
    mat = metadata['matrix']
    text = extract_answer(completion)

    start_row, start_col, goal_row, goal_col = find_starting_and_goal_positions(mat)

    current_row, current_col, valid_steps, wall_hits = _simulate_directions(
        mat, (start_row, start_col), text
    )

    reached_goal = (current_row == goal_row and current_col == goal_col)
    final_distance_to_goal = abs(current_row - goal_row) + abs(current_col - goal_col)

    # Calculate percentage of correct directions
    pred_dirs = text.lower().strip().split()
    ground_truth_dirs = metadata.get('solution', [])
    correct_dirs = 0
    compare_length = min(len(pred_dirs), len(ground_truth_dirs))
    for i in range(compare_length):
        if pred_dirs[i] == ground_truth_dirs[i].lower():
            correct_dirs += 1


    return {
        'reached_goal': reached_goal,
        'valid_steps': valid_steps,
        'wall_hits': wall_hits,
        'final_distance_to_goal': final_distance_to_goal,
        'predicted_path': text,
        'ground_truth': ' '.join(ground_truth_dirs),
        'completion': completion,
    }


def evaluate(model, tokenizer, dataset, max_new_tokens, device):
    """Run evaluation on dataset and return metrics. Greedy decoding only."""
    results = []

    for entry in tqdm(dataset, desc="Evaluating"):
        prompt = entry['prompt']

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate (greedy)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Simulate and check
        result = simulate_and_check(completion, entry['metadata'])
        results.append(result)

    return results


def calculate_results(results, verbose=False):
    """Print evaluation metrics."""
    total = len(results)
    correct = sum(1 for r in results if r['reached_goal'])
    accuracy = (correct / total) * 100 if total > 0 else 0

    avg_distance_to_goal = sum(r['final_distance_to_goal'] for r in results) / total if total > 0 else 0
    correct_results = [r for r in results if r['reached_goal']]
    avg_path_efficiency = (
        sum(len(r['ground_truth'].split()) / r['valid_steps'] for r in correct_results)
        / len(correct_results)
        if correct_results else 0
    )
    avg_wall_hits = sum(r['wall_hits'] for r in results) / total if total > 0 else 0

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy (reached goal): {accuracy:.1f}% ({correct}/{total})")
    print(f"Avg final distance to goal: {avg_distance_to_goal:.2f}")
    print(f"Avg path efficiency (correct only): {avg_path_efficiency:.2f}")
    print(f"Avg wall hits: {avg_wall_hits:.2f}")
    print("=" * 50)

    if verbose:
        print("\nDetailed results:")
        for i, r in enumerate(results):
            status = "+" if r['reached_goal'] else "x"
            print(f"\n[{i+1}] {status} Distance: {r['final_distance_to_goal']}, "
                  f"Valid: {r['valid_steps']}, Walls: {r['wall_hits']}")
            print(f"    Predicted: {r['predicted_path']}")
            print(f"    Ground truth: {r['ground_truth']}")

    return {
        "accuracy": accuracy,
        "avg_distance_to_goal": avg_distance_to_goal,
        "avg_path_efficiency_correct_only": avg_path_efficiency,
        "avg_wall_hits": avg_wall_hits,
        "total": total,
        "correct": correct,
    }


def save_results_json(metrics, args, eval_size, results, output_path):
    """Save evaluation results to JSON file."""
    output = {
        'datetime': datetime.now().isoformat(),
        'dataset': {
            'data_path': args.data_path,
            'size': eval_size,
        },
        'model': {
            'model_path': args.model_path,
            'lora_path': args.lora_path,
        },
        'generation': {
            'max_new_tokens': args.max_new_tokens,
            'temperature': 0.0,
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
    parser.add_argument("--model_path", type=str, default=os.getenv("DEFAULT_MODEL_PATH"),
                        help="Base model path")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter (optional)")

    # Dataset
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to eval.json produced by prepare_dataset.py")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Output
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results for each sample")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results JSON (default: eval_results/eval_YYYYMMDD_HHMMSS.json)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = os.getenv("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
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
    else:
        print(f"Loading WITHOUT LoRA adapter...")

    model.eval()
    print(f"Model loaded. Device: {model.device}, dtype: {model.dtype}")

    # Load eval dataset
    print(f"\nLoading eval dataset from {args.data_path}...")
    eval_dataset = load_maze_dataset(args.data_path, tokenizer)
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate(
        model, tokenizer, eval_dataset,
        max_new_tokens=args.max_new_tokens,
        device=device
    )

    # Calculate results
    metrics = calculate_results(results, verbose=args.verbose)

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_json:
        # Insert timestamp before .json extension
        output_path = Path(args.output_json)
        json_path = (
            output_path.parent
            / f"eval_results/{output_path.stem}_{timestamp}.json"
        )
    else:
        json_path = f"eval_results/eval_{timestamp}.json"

    save_results_json(metrics, args, len(eval_dataset), results, json_path)
