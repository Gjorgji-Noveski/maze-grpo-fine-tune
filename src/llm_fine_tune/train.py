import argparse
import os
import wandb

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from peft import LoraConfig
from llm_fine_tune.dataset import create_maze_dataset

from llm_fine_tune.rewards import format_reward, got_to_end_reward, binary_got_closer

load_dotenv()


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
    train_group.add_argument("--max_completion_length", type=int, default=512, help="Max tokens for generation")
    train_group.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train_group.add_argument("--logging_steps", type=int, default=10, help="Log metrics every X updates steps")
    train_group.add_argument("--save_steps", type=int, default=20, help="Save checkpoint every N steps")
    train_group.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    train_group.add_argument("--learning_rate", type=float, default=5e-5)
    train_group.add_argument("--beta", type=float, default=0.0, help="KL divergence coefficient (0=no second reference model)")
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
    other_group.add_argument("--model_path", type=str, default=os.getenv("DEFAULT_MODEL_PATH"), help="Path to local model directory")
    other_group.add_argument("--output_dir", type=str, default="./output")
    other_group.add_argument("--run_name", type=str, help="Wandb run name")
    other_group.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    other_group.add_argument("--group_name", type=str, help="Name of the group in wandb")
    other_group.add_argument("--wandb_run_id", type=str, default=None, help="Wandb run ID to resume (find this in the WANDB URL when you open a run or run overview in the WANDB online dashboard)")
    other_group.add_argument("--full_fine_tune", action="store_true",  help="If full fine tuning should be performed")
    other_group.add_argument("--reward_set", type=int, default=1, choices=[1, 2, 3],
                             help="Reward function set. Example: 1=[got_to_end], 2=[got_to_end, format], 3=[got_to_end, format, binary_got_closer]")

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
        for key, value in sweep_config.items():
            if hasattr(args, key):
                setattr(args, key, type(getattr(args, key))(value))
        # Auto-name sweep runs based on swept params
        if sweep_config:
            args.run_name = "_".join(f"{k}{v}" for k, v in sweep_config.items())
            wandb.run.name = args.run_name
    device = os.getenv("DEVICE")
    checkpoint_path = args.resume_from_checkpoint
    args.output_dir = os.path.join(args.output_dir, args.run_name)

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tok.pad_token = tok.eos_token
    maze_data, _, _ = create_maze_dataset(
        tokenizer=tok,
        size=args.dataset_size,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_cols=args.min_cols,
        max_cols=args.max_cols,
        p_blocked=args.p_blocked,
        seed=args.seed
    )
    dataset = Dataset.from_list(maze_data)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=device, dtype='float16')
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

    # Save model
    if args.full_fine_tune:
        save_path = f"{args.output_dir}/model"
    else:
        save_path = f"{args.output_dir}/lora"
    trainer.model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"Done. {'Model' if args.full_fine_tune else 'LoRA adapter'} saved to {save_path}")
