# Cell: Push Fine-Tuned Model Adapters to Hugging Face Hub

# 1. Login to Hugging Face (Paste your HF Write Token when prompted)
from huggingface_hub import notebook_login
notebook_login()

# 2. Push fine-tuned LoRA adapter weights directly from Kaggle
repo_id = "Devansh-Ojha/doctorboom-qwen2.5-coder-7b-lora"
print(f"[+] Pushing fine-tuned model to Hugging Face: {repo_id}...")

model.push_to_hub_merged(
    repo_id,
    tokenizer,
    save_method="lora",  # Pushes lightweight LoRA adapter (~100 MB)
    token=True
)

print(f"[+] Model successfully published to https://huggingface.co/{repo_id}")
