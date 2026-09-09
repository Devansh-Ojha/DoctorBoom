import os
import json
import torch
import gc
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# 1. Memory Optimization Environment Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda:0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear residual VRAM memory allocations
gc.collect()
torch.cuda.empty_cache()

# 2. Load Model in 4-bit with reduced sequence length
max_seq_length = 1024  # Reduced from 2048 to prevent VRAM OOM
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

print("[+] Loading Qwen2.5-Coder-7B in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    device_map="cuda:0",
)

# 3. Apply QLoRA Adapters
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
def format_prompts(examples):
    texts = []
    for subject, body, changed_files, diff in zip(
        examples["subject"], examples["body"], examples["changed_files"], examples["diff"]
    ):
        text = f"""### Hardware Bug Repair Task
Analyze the hardware failure trace and code context below, diagnose the root cause, and provide a patch diff.

### Failure Information:
{subject}
{body}

### Changed Files:
{json.dumps(changed_files, indent=2)}

### Diagnosis & Patch:
{diff}"""
        texts.append(text)
    return {"text": texts}

# 5. Load Dataset
data_path = "/kaggle/input/datasets/devanshojha05/doctorboom-train-data/train.jsonl"
train_data = []

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            train_data.append(json.loads(line))

raw_dataset = Dataset.from_list(train_data)
dataset = raw_dataset.map(format_prompts, batched=True)

# 6. SFTConfig Optimized for Low VRAM (Batch size 1, 8 Grad Accumulation)
sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    per_device_train_batch_size=1,        # Reduced from 2 to 1 (uses 50% less VRAM!)
    gradient_accumulation_steps=8,         # Maintained effective batch size (1 x 8 = 8)
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=5,
    output_dir="outputs",
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

# 7. Run Fine-Tuning
print("[+] Starting QLoRA Fine-Tuning...")
trainer.train()

# 8. Save Artifacts
output_dir = "/kaggle/working/doctorboom_qwen2.5_coder_lora"
print(f"[+] Saving LoRA adapter weights to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[+] Fine-Tuning & Saving Complete!")
