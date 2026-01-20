import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
DataCollatorForLanguageModeling, TrainingArguments, Trainer)

from peft import LoraConfig, get_peft_model
# Select model
model_name_path = "models/gemma_3.1_4B_instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16
# Load tokenizer
tok = AutoTokenizer.from_pretrained(model_name_path, use_fast=True)
if tok.pad_token is None:
tok.pad_token = tok.eos_token
# Load model
model = AutoModelForCausalLM.from_pretrained(model_name_path, dtype='auto', device_map='mps')
args = TrainingArguments(
output_dir="out",
per_device_train_batch_size=1,
gradient_accumulation_steps=16,
learning_rate=2e-4,
num_train_epochs=3,
fp16=False,
bf16=False,
dataloader_pin_memory=False,
logging_steps=10,
save_total_limit=2,
report_to="none"
)
# Memory optimization for MPS
model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
model.gradient_checkpointing_enable()
# LoRA config
lora_cfg = LoraConfig(
r=8, lora_alpha=16, lora_dropout=0.05,
target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
# Dataset (place your own my_data.json in format [{"instruction": "", "input": "", "output": ""}, {....}])
ds = load_dataset("json", data_files="my_data.json", split="train")
def format_example(data):
prompt = f"Instruction: {data['instruction']}\nInput: {data['input']}\nResponse:"
text = prompt + data["output"] + tok.eos_token
return {"input_ids": tok(text, truncation=True, max_length=1024)["input_ids"]}
ds = ds.map(format_example, remove_columns=["instruction", "input", "output"])
collator = DataCollatorForLanguageModeling(tok, mlm=False)
trainer = Trainer(
model=model,
args=args,
train_dataset=ds,
data_collator=collator,
)
trainer.train()
# Save LoRA adapter
model.save_pretrained("out/lora")
tok.save_pretrained("out/lora")
print("Done. LoRA adapter saved to out/lora")