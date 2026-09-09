import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

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

def train():
    max_seq_length = 2048
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    print(f"[+] Loading model {model_name} with Unsloth 4-bit QLoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    train_data = []
    with open("dataset/train.jsonl", "r") as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))

    raw_dataset = Dataset.from_list(train_data)
    dataset = raw_dataset.map(format_prompt)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir="outputs",
        ),
    )

    print("[+] Starting QLoRA Fine-Tuning...")
    trainer.train()
    print("[+] Saving Fine-Tuned Model Checkpoint...")
    model.save_pretrained_merged("doctorboom_qwen2.5_coder_lora", tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    train()
