# Fine-Tuning Qwen2.5-Coder (7B) with Unsloth QLoRA for Hardware Repair

This directory contains the Unsloth QLoRA fine-tuning notebook and script for training Qwen2.5-Coder-7B on hardware bug-fix diffs mined from `rocket-chip` / `boom`.

## Fine-Tuning Configuration (Kaggle / Colab GPU)

- **Model Base**: `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Quantization**: 4-bit (QLoRA)
- **LoRA Rank**: `r = 16`, `lora_alpha = 16`
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Format**:
  - **Input**: Failure trace / bug description + context.
  - **Output**: Hardware bug diagnosis + target file patch diff.

## Executing Fine-Tuning

```bash
# Export train dataset
python dataset/split_dataset.py

# Launch fine-tuning (Kaggle T4/P100 or Colab L4/A100)
python finetune/train_qlora.py
```
