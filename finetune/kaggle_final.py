import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer

# 1. Force single GPU execution AND disable multi-GPU/Triton memory splits
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda:0"

# 2. Load Base Model with device_map="cuda:0" explicitly
max_seq_length = 2048
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

print("[+] Loading Qwen2.5-Coder-7B in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    device_map="cuda:0",
)

# Fix BOS token duplication during tokenization
tokenizer.padding_side = "right"

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

# 4. Tokenize with Single Pass (Prevents BOS duplication & loss zero-out)
def tokenize_function(example):
    prompt_text = (
        "### Hardware Bug Repair Task\n"
        "Analyze the hardware failure trace and code context below, diagnose the root cause, and provide a patch diff.\n\n"
        f"### Failure Information:\n{example.get('subject', '')}\n{example.get('body', '')}\n\n"
        f"### Changed Files:\n{json.dumps(example.get('changed_files', []), indent=2)}\n\n"
        "### Diagnosis & Patch:\n"
    )
    target_text = f"{example.get('diff', '')}{tokenizer.eos_token}"
    full_text = prompt_text + target_text

    # Single pass tokenization prevents BOS token duplication
    full_tokens = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)

    input_ids = full_tokens["input_ids"]
    labels = list(input_ids)

    # Mask out prompt portion (-100)
    prompt_len = len(prompt_tokens["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }

# 5. Load Dataset & Tokenize
data_path = "/kaggle/input/datasets/devanshojha05/doctorboom-train-data/train.jsonl"
train_data = []

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            train_data.append(json.loads(line))

raw_dataset = Dataset.from_list(train_data)
dataset = raw_dataset.map(tokenize_function, remove_columns=raw_dataset.column_names)

# 6. Native Trainer (Explicitly on GPU)
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        output_dir="outputs",
        remove_unused_columns=True,
    ),
)

# 7. Run Fine-Tuning Loop
print("[+] Starting QLoRA Fine-Tuning...")
trainer.train()

# 8. Save LoRA Adapter Artifacts
output_dir = "/kaggle/working/doctorboom_qwen2.5_coder_lora"
print(f"[+] Saving LoRA adapter weights to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[+] Fine-Tuning & Saving Complete!")
