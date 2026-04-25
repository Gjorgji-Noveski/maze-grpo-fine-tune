"""Generate train + eval maze datasets with deduplication.

Run once to produce a versioned data artifact that train.py and evaluate.py load.
Both splits are generated in the same invocation so eval can be deduplicated
against train using exact grid-match signatures.
"""
import argparse
import json
from pathlib import Path

import reasoning_gym

from llm_fine_tune.dataset import _clean_prompt


def _grid_signature(matrix):
    """Hashable signature of a maze grid (includes walls, start, goal)."""
    return tuple(tuple(row) for row in matrix)


def _generate_raw(seed, requested_size, buffer_multiplier, min_rows, max_rows,
                  min_cols, max_cols, p_blocked):
    """Generate buffer*requested mazes, filter infeasibles, return list of entries.

    Returns the entries with cleaned question text. Does not trim — caller trims
    after dedup.
    """
    raw_size = int(requested_size * buffer_multiplier)
    dataset = reasoning_gym.create_dataset(
        "shortest_path",
        min_rows=min_rows,
        max_rows=max_rows,
        min_cols=min_cols,
        max_cols=max_cols,
        p_blocked=p_blocked,
        size=raw_size,
        seed=seed,
    )

    feasible = []
    infeasible_count = 0
    for entry in dataset:
        if entry['answer'].lower() == 'infeasible':
            infeasible_count += 1
            continue
        feasible.append({
            'question': _clean_prompt(entry['question']),
            'answer': entry['answer'],
            'metadata': entry['metadata'],
        })

    return feasible, raw_size, infeasible_count


def _count_internal_duplicates(entries):
    seen = set()
    dupes = 0
    for e in entries:
        sig = _grid_signature(e['metadata']['matrix'])
        if sig in seen:
            dupes += 1
        else:
            seen.add(sig)
    return dupes


def _config_space_estimate(min_rows, max_rows, min_cols, max_cols):
    """Rough lower bound on distinct configurations for the smallest grid in range.

    Uses the smallest grid because that is the bottleneck for collisions.
    Counts: 2^cells wall patterns * cells * (cells - 1) start/goal placements.
    """
    cells = min_rows * min_cols
    return (2 ** cells) * cells * (cells - 1)


def parse_args():
    p = argparse.ArgumentParser(description="Generate train+eval maze datasets")
    p.add_argument("--min_rows", type=int, required=True)
    p.add_argument("--max_rows", type=int, required=True)
    p.add_argument("--min_cols", type=int, required=True)
    p.add_argument("--max_cols", type=int, required=True)
    p.add_argument("--p_blocked", type=float, required=True)

    p.add_argument("--train_size", type=int, required=True)
    p.add_argument("--train_seed", type=int, required=True)
    p.add_argument("--eval_size", type=int, required=True)
    p.add_argument("--eval_seed", type=int, required=True)

    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write train.json, eval.json, config.json")
    p.add_argument("--buffer_multiplier", type=float, default=2.0,
                   help="Generate buffer*size raw mazes before filtering and dedup")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.train_seed == args.eval_seed:
        raise SystemExit("train_seed and eval_seed must differ")

    print(f"Generating train (seed={args.train_seed}, size={args.train_size})...")
    train, train_raw, train_infeasible = _generate_raw(
        args.train_seed, args.train_size, args.buffer_multiplier,
        args.min_rows, args.max_rows, args.min_cols, args.max_cols, args.p_blocked,
    )
    if len(train) < args.train_size:
        raise SystemExit(
            f"Only {len(train)} feasible train mazes after filtering "
            f"{train_raw} raw (infeasible: {train_infeasible}). "
            f"Increase --buffer_multiplier or lower --p_blocked."
        )
    train_internal_dupes = _count_internal_duplicates(train[:args.train_size])
    train = train[:args.train_size]
    train_sigs = {_grid_signature(e['metadata']['matrix']) for e in train}

    print(f"Generating eval (seed={args.eval_seed}, size={args.eval_size})...")
    eval_raw_entries, eval_raw, eval_infeasible = _generate_raw(
        args.eval_seed, args.eval_size, args.buffer_multiplier,
        args.min_rows, args.max_rows, args.min_cols, args.max_cols, args.p_blocked,
    )
    eval_clean = []
    eval_dropped_overlap = 0
    for e in eval_raw_entries:
        if _grid_signature(e['metadata']['matrix']) in train_sigs:
            eval_dropped_overlap += 1
            continue
        eval_clean.append(e)

    if len(eval_clean) < args.eval_size:
        raise SystemExit(
            f"Only {len(eval_clean)} eval mazes after filtering+dedup "
            f"({eval_raw} raw, {eval_infeasible} infeasible, "
            f"{eval_dropped_overlap} overlapping with train). "
            f"Increase --buffer_multiplier or pick a larger maze size."
        )
    eval_internal_dupes = _count_internal_duplicates(eval_clean[:args.eval_size])
    eval_clean = eval_clean[:args.eval_size]

    config_space = _config_space_estimate(
        args.min_rows, args.max_rows, args.min_cols, args.max_cols
    )
    if args.train_size > config_space / 100:
        print(
            f"WARNING: train_size ({args.train_size}) is large relative to the "
            f"~{config_space:.2g} configuration space for the smallest grid in "
            f"range. Within-train duplicates are likely."
        )

    print()
    print("=" * 60)
    print(f"Train: {len(train)} mazes (raw {train_raw}, infeasible {train_infeasible}, "
          f"within-train dupes {train_internal_dupes})")
    print(f"Eval:  {len(eval_clean)} mazes (raw {eval_raw}, infeasible {eval_infeasible}, "
          f"dropped-overlap {eval_dropped_overlap}, within-eval dupes {eval_internal_dupes})")
    print("=" * 60)

    with open(out / "train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(out / "eval.json", "w") as f:
        json.dump(eval_clean, f, indent=2)

    config = {
        "min_rows": args.min_rows,
        "max_rows": args.max_rows,
        "min_cols": args.min_cols,
        "max_cols": args.max_cols,
        "p_blocked": args.p_blocked,
        "train": {
            "size": len(train),
            "seed": args.train_seed,
            "raw_generated": train_raw,
            "infeasible_dropped": train_infeasible,
            "within_split_duplicates": train_internal_dupes,
        },
        "eval": {
            "size": len(eval_clean),
            "seed": args.eval_seed,
            "raw_generated": eval_raw,
            "infeasible_dropped": eval_infeasible,
            "dropped_due_to_train_overlap": eval_dropped_overlap,
            "within_split_duplicates": eval_internal_dupes,
        },
        "buffer_multiplier": args.buffer_multiplier,
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote {out}/train.json, eval.json, config.json")


if __name__ == "__main__":
    main()
