import os
import json
import torch
from unsloth import FastLanguageModel

# 1. Option A: Save standalone LoRA adapters (Fastest & Smallest)
output_lora_dir = "/kaggle/working/doctorboom_qwen2.5_coder_lora"
print(f"[+] Saving LoRA adapter weights to {output_lora_dir}...")
model.save_pretrained(output_lora_dir)
tokenizer.save_pretrained(output_lora_dir)
print("[+] LoRA Adapter Saved Successfully!")

# 2. Loading saved LoRA adapters back into Unsloth for evaluation:
print(f"[+] Re-loading base model + saved adapter for evaluation...")
eval_model, eval_tokenizer = FastLanguageModel.from_pretrained(
    model_name=output_lora_dir,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(eval_model)
print("[+] Model loaded successfully and ready for evaluation!")
