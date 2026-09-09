import os
import json
import time
import torch
from typing import List, Dict, Any, Optional
from unsloth import FastLanguageModel
from agent.loop import BaseLLM, RepairLoop

class QwenFineTunedLLM(BaseLLM):
    def __init__(self, model_path: str = "doctorboom_qwen2.5_coder_lora", max_seq_length: int = 2048):
        print(f"[+] Loading Fine-Tuned Model from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def propose_patch(self, failure_trace: str, context: str) -> str:
        prompt = f"""### Hardware Bug Repair Task
Analyze the hardware failure trace and code context below, diagnose the root cause, and provide a patch diff.

### Failure Information:
{failure_trace}

### Changed Files / Context:
{context}

### Diagnosis & Patch:
"""
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=512, use_cache=True)
        response = self.tokenizer.batch_decode(outputs)
        # Extract generated patch following prompt
        generated = response[0].split("### Diagnosis & Patch:")[-1].strip()
        return generated

def run_eval(eval_jsonl: str, model_path: str, output_log: str, max_iterations: int = 3):
    model = QwenFineTunedLLM(model_path=model_path)
    loop = RepairLoop(model=model, max_iterations=max_iterations)

    eval_tasks = []
    with open(eval_jsonl, "r") as f:
        for line in f:
            if line.strip():
                eval_tasks.append(json.loads(line))

    print(f"[+] Starting Fine-Tuned Model Eval on {len(eval_tasks)} held-out tasks...")

    results = []
    start_time = time.time()

    for idx, task in enumerate(eval_tasks):
        task_id = task.get("commit_hash", f"task_{idx}")
        trace = f"{task.get('subject', '')}\n\n{task.get('body', '')}"
        context = json.dumps(task.get('changed_files', []), indent=2)

        task_input = {
            "id": task_id,
            "failure_trace": trace,
            "context": context
        }

        print(f"\n--- Running Task {idx+1}/{len(eval_tasks)}: {task_id[:8]} ---")
        t0 = time.time()
        res = loop.run_repair(task_input)
        elapsed = time.time() - t0
        res["wall_clock_sec"] = elapsed
        results.append(res)

    total_time = time.time() - start_time
    summary = {
        "model": "Fine-Tuned Qwen2.5-Coder-7B (QLoRA)",
        "total_tasks": len(eval_tasks),
        "total_wall_clock_sec": total_time,
        "results": results
    }

    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    with open(output_log, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Fine-Tuned Eval Complete! Logged results to {output_log}")

if __name__ == "__main__":
    run_eval(
        eval_jsonl="dataset/eval.jsonl",
        model_path="doctorboom_qwen2.5_coder_lora",
        output_log="eval/finetuned_qwen_results.json"
    )
