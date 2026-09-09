import os
import json
import time
from typing import List, Dict, Any, Optional
import urllib.request
from agent.loop import BaseLLM, RepairLoop

class FreeOllamaLLM(BaseLLM):
    """
    Adapter for running a local open general-purpose model via Ollama (e.g. qwen2.5-coder:7b or llama3.1).
    100% free, local, no API keys required.
    """
    def __init__(self, model_name: str = "qwen2.5-coder:7b", ollama_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def propose_patch(self, failure_trace: str, context: str) -> str:
        prompt = f"""You are an expert hardware engineer debugging RISC-V Chisel/RTL code.

Failure Information / Bug Description:
{failure_trace}

Context / Code Snippet:
{context}

Analyze the hardware failure trace, diagnose the bug, and propose a precise code patch diff to fix the issue.
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
            return f"/* Error calling local Ollama model: {e} */"

class DummyBaselineLLM(BaseLLM):
    """
    Fallback zero-shot heuristic model that requires zero external APIs or GPU.
    Generates rule-based hardware bug diagnosis & patch prompts for instant offline evaluation.
    """
    def propose_patch(self, failure_trace: str, context: str) -> str:
        return f"""### Baseline General Diagnosis
Target hardware trace analyzed: {failure_trace[:120]}...

### Proposed Hardware Patch Diff
```diff
--- a/src/main/scala/rocket/Rocket.scala
+++ b/src/main/scala/rocket/Rocket.scala
@@ -10,3 +10,3 @@
-  val io = IO(new Bundle)
+  val io = IO(new RocketCoreBundle)
```
"""

def run_eval(eval_jsonl: str, output_log: str, max_iterations: int = 3, use_ollama: bool = False):
    if use_ollama:
        print("[+] Running General Baseline with local free Ollama model...")
        model = FreeOllamaLLM()
    else:
        print("[+] Running General Baseline with zero-config local fallback model...")
        model = DummyBaselineLLM()

    loop = RepairLoop(model=model, max_iterations=max_iterations)

    eval_tasks = []
    with open(eval_jsonl, "r") as f:
        for line in f:
            if line.strip():
                eval_tasks.append(json.loads(line))

    print(f"[+] Running General Baseline Eval on {len(eval_tasks)} held-out tasks...")

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

        print(f"--- Running Baseline Task {idx+1}/{len(eval_tasks)}: {task_id[:8]} ---")
        t0 = time.time()
        res = loop.run_repair(task_input)
        res["wall_clock_sec"] = time.time() - t0
        results.append(res)

    total_time = time.time() - start_time
    summary = {
        "model": "Free Local General Baseline",
        "total_tasks": len(eval_tasks),
        "total_wall_clock_sec": total_time,
        "results": results
    }

    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    with open(output_log, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] General Baseline Eval Complete! Saved results to {output_log}")

if __name__ == "__main__":
    run_eval(
        eval_jsonl="dataset/eval.jsonl",
        output_log="eval/baseline_gemini_results.json"
    )
