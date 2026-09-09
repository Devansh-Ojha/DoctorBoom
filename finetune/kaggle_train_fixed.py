import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# 1. Force single GPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 2. Load Base Model and Tokenizer with Unsloth
max_seq_length = 2048
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

print("[+] Loading Qwen2.5-Coder-7B in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# 3. Apply QLoRA Target Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 4. Prompt Formatting Function
def format_prompt(example):
    prompt = f"""### Hardware Bug Repair Task
Analyze the hardware failure trace and code context below, diagnose the root cause, and provide a patch diff.

### Failure Information:
{example.get('subject', '')}
{example.get('body', '')}

### Changed Files:
{json.dumps(example.get('changed_files', []), indent=2)}

### Diagnosis & Patch:
{example.get('diff', '')}
"""
    return {"text": prompt}

# 5. Load Dataset
data_path = "/kaggle/input/datasets/devanshojha05/doctorboom-train-data/train.jsonl"
train_data = []

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            train_data.append(json.loads(line))

raw_dataset = Dataset.from_list(train_data)
dataset = raw_dataset.map(format_prompt)

# 6. Configure SFTConfig
sft_args = SFTConfig(
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=5,
    output_dir="outputs",
)

# Pass tokenizer inside SFTTrainer positional args directly (bypasses keyword argument checks)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_args,
)

# Assign tokenizer directly onto trainer instance for Unsloth
trainer.tokenizer = tokenizer

# 7. Run Fine-Tuning Loop
print("[+] Starting QLoRA Fine-Tuning...")
trainer.train()

# 8. Save Model Artifact
output_dir = "/kaggle/working/doctorboom_qwen2.5_coder_lora"
print(f"[+] Saving model weights to {output_dir}...")
model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
print("[+] Fine-Tuning Complete!")
