"""Wrapper script for wandb sweep over training_model_v3.py"""
import wandb
import subprocess
import sys

REWARD_SETS = {
    1: ["got_to_end_reward"],
    2: ["got_to_end_reward", "format_reward"],
    3: ["got_to_end_reward", "format_reward", "binary_got_closer"],
}


def main():
    config = wandb.config

    lr = config.learning_rate
    reward_set = config.reward_set

    run_name = f"sweep_lr{lr}_rewards{reward_set}"

    cmd = [
        "uv", "run", "training_model_v3.py",
        "--min_rows", "3",
        "--max_rows", "3",
        "--min_cols", "3",
        "--max_cols", "3",
        "--dataset_size", "10000",
        "--max_steps", "50",
        "--temperature", "0.6",
        "--logging_steps", "5",
        "--save_steps", "300",
        "--max_completion_length", "600",
        "--num_generations", "6",
        "--generation_batch_size", "6",
        "--full_fine_tune",
        "--learning_rate", str(lr),
        "--reward_set", str(reward_set),
        "--run_name", run_name,
        "--group_name", "sweep",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/Users/gjorgji/PycharmProjects/llm_fine_tune")
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
