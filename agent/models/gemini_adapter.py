import os
import json
import time
from typing import List, Dict, Any, Optional
from google import genai
from agent.loop import BaseLLM, RepairLoop

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or argument is required")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def propose_patch(self, failure_trace: str, context: str) -> str:
        prompt = f"""You are an expert hardware engineer debugging RISC-V Chisel/RTL code.

Failure Information / Bug Description:
{failure_trace}

Context / Code Snippet:
{context}

Analyze the hardware failure trace, diagnose the bug, and propose a precise code patch diff to fix the issue.
"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

def run_eval(eval_jsonl: str, output_log: str, max_iterations: int = 3):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[!] Warning: GEMINI_API_KEY not set. Set GEMINI_API_KEY to run live baseline eval.")
        return

    model = GeminiLLM(api_key=api_key)
    loop = RepairLoop(model=model, max_iterations=max_iterations)

    eval_tasks = []
    with open(eval_jsonl, "r") as f:
        for line in f:
            if line.strip():
                eval_tasks.append(json.loads(line))

    print(f"[+] Starting Baseline Eval with Gemini on {len(eval_tasks)} held-out tasks...")

    results = []
    start_time = time.time()

    for idx, task in enumerate(eval_tasks):
        task_id = task.get("commit_hash", f"task_{idx}")
        trace = f"Commit Message:\n{task.get('subject', '')}\n\n{task.get('body', '')}"
        context = f"Changed Files:\n{json.dumps(task.get('changed_files', []), indent=2)}"

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
        "total_tasks": len(eval_tasks),
        "total_wall_clock_sec": total_time,
        "results": results
    }

    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    with open(output_log, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Baseline Eval Complete! Logged results to {output_log}")

if __name__ == "__main__":
    run_eval(
        eval_jsonl="/Users/devojha/DoctorBoom/dataset/eval.jsonl",
        output_log="/Users/devojha/DoctorBoom/eval/baseline_gemini_results.json"
    )
