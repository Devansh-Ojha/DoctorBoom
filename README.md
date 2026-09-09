# DoctorBoom: Fine-Tuned Bug-Repair Model for Agentic Debugging of BOOM / Rocket-Chip

DoctorBoom is a research framework for hardware bug diagnosis and agentic repair of Chisel-based RISC-V cores (`riscv-boom`, `rocket-chip`, `chipyard`).

## Project Roadmap & Architecture

```
DoctorBoom/
├── agent/            # CHIA-style iterative repair loop & model adapters
│   ├── loop.py       # Main agentic loop: Propose -> Apply -> Verify -> Retry
│   ├── models/       # Swappable model adapters (Gemini API, Qwen/Unsloth local)
│   └── verifier.py   # Verilator / Spike harness interface
├── dataset/          # Mining, filtering, and dataset construction
│   ├── mine_commits.py   # Extract bug-fix commits & parent diffs
│   └── verify_data.py    # Automated reproduction verifier
├── finetune/         # QLoRA fine-tuning recipes (Unsloth / Qwen2.5-Coder)
├── eval/             # Evaluation harness & benchmark logs
└── scripts/          # Setup and helper scripts
```

## Workflow Execution Steps

- **Step 1: Verification Harness Setup**
  - Setup Chipyard / Rocket-Chip Verilator workflow.
  - Benchmark build + simulation latency budget.
- **Step 2: Dataset Mining**
  - Filter historical bug fixes touching both tests and Chisel/RTL source.
  - Split dataset into 80/20 train/eval splits.
- **Step 3: CHIA-Style Agentic Debugging Loop**
  - Implement swappable LLM slot (`propose_patch(failure_trace, context)`).
- **Step 4: General-Purpose Baseline Eval**
  - Evaluate Gemini API baseline on held-out test split.
- **Step 5: QLoRA Fine-Tuning**
  - Train Qwen2.5-Coder (7B/14B) on mined hardware repair dataset.
- **Step 6 & 7: Fine-Tuned Eval & Comparison**
  - Evaluate specialized model vs general-purpose baseline on resolution rate, iterations, and latency.
