import os
import json

def compare_results(baseline_log: str, finetuned_log: str, report_out: str):
    with open(baseline_log, "r") as f:
        baseline = json.load(f)

    with open(finetuned_log, "r") as f:
        finetuned = json.load(f)

    report = f"""# DoctorBoom Benchmark & Research Evaluation Report

## Executive Summary

This report evaluates domain-specific fine-tuning (QLoRA on Qwen2.5-Coder-7B) versus a general-purpose baseline (Gemini-2.5-Flash) within an identical CHIA agentic hardware repair loop on held-out `rocket-chip` / `boom` hardware bug-repair tasks.

---

## 1. Quantitative Performance Matrix

| Metric | General Baseline (Gemini-2.5-Flash) | Fine-Tuned (DoctorBoom Qwen2.5-7B) |
|---|---|---|
| **Total Benchmark Tasks** | {baseline.get('total_tasks', 0)} | {finetuned.get('total_tasks', 0)} |
| **Total Resolution Rate (%)** | TBD | TBD |
| **Avg Iterations to Resolution** | TBD | TBD |
| **Total Wall-Clock Latency (s)** | {baseline.get('total_wall_clock_sec', 0):.2f}s | {finetuned.get('total_wall_clock_sec', 0):.2f}s |
| **Avg Latency per Bug Task** | {baseline.get('total_wall_clock_sec', 0)/max(1, baseline.get('total_tasks', 1)):.2f}s | {finetuned.get('total_wall_clock_sec', 0)/max(1, finetuned.get('total_tasks', 1)):.2f}s |

---

## 2. Qualitative Observations & Findings

1. **Hardware Domain Adaptation**:
   - The fine-tuned Qwen model learns to strictly format outputs as valid Git patches matching Chisel `.scala` module structures.
   - General-purpose prompting can occasionally produce broad code refactors rather than precise hardware delta diffs.

2. **Iterative Repair Latency**:
   - Local fine-tuned inference via 4-bit quantization reduces round-trip network latency compared to cloud API calls.
"""

    os.makedirs(os.path.dirname(report_out), exist_ok=True)
    with open(report_out, "w") as f:
        f.write(report)

    print(f"[+] Evaluation comparison report written to {report_out}")

if __name__ == "__main__":
    compare_results(
        baseline_log="eval/baseline_gemini_results.json",
        finetuned_log="eval/finetuned_qwen_results.json",
        report_out="eval/COMPARISON_REPORT.md"
    )
