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
from peft import PeftModel, get_peft_model, LoraConfig
import json
from safetensors.torch import load_file
from dotenv import load_dotenv

from llm_fine_tune.dataset import create_maze_dataset
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
    final_distance = abs(current_row - goal_row) + abs(current_col - goal_col)

    # Calculate percentage of correct directions
    pred_dirs = text.lower().strip().split()
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


def load_lora_adapter(model, lora_path):
    """Load LoRA adapter, handling key format differences (e.g. from Prime Intellect).

    Prime Intellect saves keys like: model.layers.0.mlp.down_proj.lora_A.weight
    PEFT expects:  base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight

    PeftModel.from_pretrained emits false "missing keys" warnings because it compares
    against the full model state dict (base weights + lora), not just lora keys. Loading
    via get_peft_model + load_state_dict avoids this and correctly loads the weights.
    """
    adapter_file = Path(lora_path) / "adapter_model.safetensors"
    tensors = load_file(adapter_file)

    sample_key = next(iter(tensors))
    needs_remap = ".lora_A.weight" in sample_key or ".lora_B.weight" in sample_key

    with open(Path(lora_path) / "adapter_config.json") as f:
        adapter_cfg_raw = json.load(f)

    lora_cfg = LoraConfig(
        r=adapter_cfg_raw["r"],
        lora_alpha=adapter_cfg_raw["lora_alpha"],
        lora_dropout=adapter_cfg_raw.get("lora_dropout", 0.0),
        target_modules=adapter_cfg_raw["target_modules"],
        bias=adapter_cfg_raw.get("bias", "none"),
        task_type=adapter_cfg_raw.get("task_type", "CAUSAL_LM"),
    )
    peft_model = get_peft_model(model, lora_cfg)

    if needs_remap:
        print("  Remapping LoRA keys (Prime Intellect format: missing 'default' adapter name prefix)...")
        state_dict = {}
        for k, v in tensors.items():
            new_k = k.replace(".lora_A.weight", ".lora_A.default.weight")
            new_k = new_k.replace(".lora_B.weight", ".lora_B.default.weight")
            new_k = "base_model.model." + new_k
            state_dict[new_k] = v
    else:
        state_dict = dict(tensors)

    missing, unexpected = peft_model.load_state_dict(state_dict, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        print(f"  WARNING: {len(lora_missing)} LoRA keys failed to load: {lora_missing[:3]}...")
    else:
        print(f"  LoRA weights loaded successfully ({len(state_dict)} keys).")

    return peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate maze-solving model")

    # Model
    parser.add_argument("--model_path", type=str, default=os.getenv("DEFAULT_MODEL_PATH"),
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
        model = load_lora_adapter(model, args.lora_path)
    else:
        print(f"Loading WITHOUT LoRA adapter...")

    model.eval()
    print(f"Model loaded. Device: {model.device}, dtype: {model.dtype}")

    # Create eval dataset
    print(f"\nCreating eval dataset with seed={args.seed}, size={args.eval_size}...")
    eval_dataset, original_size, filtered_size = create_maze_dataset(
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
            / f"eval_results/{output_path.stem}_{timestamp}.json"
        )
    else:
        json_path = f"eval_results/eval_{timestamp}.json"

    save_results_json(metrics, args, original_size, filtered_size, results, json_path)
