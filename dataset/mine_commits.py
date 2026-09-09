import os
import sys
import json
import re
import argparse
import subprocess
from typing import List, Dict, Any, Optional

def run_git_cmd(args: List[str], cwd: str) -> str:
    res = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, check=True)
    return res.stdout

def mine_bug_commits(repo_dir: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Mines bug-fix commits from git log that touch both test files and Chisel/RTL source files.
    """
    print(f"[+] Mining git commits in {repo_dir}...")
    log_cmd = [
        "log",
        "--all",
        f"-n{limit}",
        "--grep=fix\\|bug\\|regression\\|incorrect",
        "-i",
        "--format=COMMIT_SEP%n%H%n%s%n%b%nDIFF_SEP"
    ]
    raw_log = run_git_cmd(log_cmd, repo_dir)
    raw_commits = raw_log.split("COMMIT_SEP\n")

    mined_examples = []
    
    for raw in raw_commits:
        if not raw.strip():
            continue
        parts = raw.split("\nDIFF_SEP\n")
        header_lines = parts[0].strip().split("\n")
        commit_hash = header_lines[0]
        subject = header_lines[1] if len(header_lines) > 1 else ""
        body = "\n".join(header_lines[2:]) if len(header_lines) > 2 else ""

        # Get changed files
        changed_files_str = run_git_cmd(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash], repo_dir)
        changed_files = [f.strip() for f in changed_files_str.strip().split("\n") if f.strip()]

        has_rtl_source = any(re.search(r'\.(scala|v|sv)$', f) for f in changed_files if not "test" in f.lower())
        has_test_touch = any("test" in f.lower() or "spec" in f.lower() for f in changed_files)

        if has_rtl_source and has_test_touch:
            diff_str = run_git_cmd(["show", commit_hash], repo_dir)
            parent_hash = run_git_cmd(["rev-parse", f"{commit_hash}^"], repo_dir).strip()

            mined_examples.append({
                "commit_hash": commit_hash,
                "parent_hash": parent_hash,
                "subject": subject,
                "body": body,
                "changed_files": changed_files,
                "diff": diff_str
            })
            print(f"  [Match] {commit_hash[:8]}: {subject[:60]}")

    print(f"[+] Total usable candidate bug-fix commits found: {len(mined_examples)}")
    return mined_examples

def main():
    parser = argparse.ArgumentParser(description="Mine bug-fix commits from Rocket-Chip / BOOM repos")
    parser.add_argument("--repo", type=str, required=True, help="Path to repository clone")
    parser.add_argument("--output", type=str, default="dataset/mined_candidates.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, default=1000, help="Max commits to scan")
    args = parser.parse_args()

    candidates = mine_bug_commits(args.repo, args.limit)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"[+] Saved {len(candidates)} candidates to {args.output}")

if __name__ == "__main__":
    main()
