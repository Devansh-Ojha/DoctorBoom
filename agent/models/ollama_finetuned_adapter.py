import os
import json
import time
from typing import List, Dict, Any, Optional
import urllib.request
from agent.loop import BaseLLM, RepairLoop

class OllamaFineTunedLLM(BaseLLM):
    """
    Adapter for running your fine-tuned GGUF/Ollama model locally on CPU/Mac metal.
    100% free, local, no external GPUs or APIs needed!
    """
    def __init__(self, model_name: str = "doctorboom-qwen", ollama_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def propose_patch(self, failure_trace: str, context: str) -> str:
        prompt = f"""### Hardware Bug Repair Task
Analyze the hardware failure trace and code context below, diagnose the root cause, and provide a patch diff.

### Failure Information:
{failure_trace}

### Changed Files:
{context}

### Diagnosis & Patch:
"""
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(self.ollama_url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except Exception as e:
            return f"/* Error running local fine-tuned model via Ollama: {e} */"

def run_eval(eval_jsonl: str, output_log: str, max_iterations: int = 3):
    model = OllamaFineTunedLLM()
    loop = RepairLoop(model=model, max_iterations=max_iterations)

    eval_tasks = []
    with open(eval_jsonl, "r") as f:
        for line in f:
            if line.strip():
                eval_tasks.append(json.loads(line))

    print(f"[+] Running Fine-Tuned Model Eval on {len(eval_tasks)} held-out tasks...")

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

        print(f"--- Running Fine-Tuned Task {idx+1}/{len(eval_tasks)}: {task_id[:8]} ---")
        t0 = time.time()
        res = loop.run_repair(task_input)
        res["wall_clock_sec"] = time.time() - t0
        results.append(res)

    total_time = time.time() - start_time
    summary = {
        "model": "Fine-Tuned DoctorBoom Qwen (Ollama Local)",
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
        output_log="eval/finetuned_qwen_results.json"
    )
