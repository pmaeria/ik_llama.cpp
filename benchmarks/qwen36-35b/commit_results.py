#!/usr/bin/env python3
"""Commit and optionally push one EXP-0001 evidence bundle on a dedicated branch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def run(argv: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(map(str, argv)),
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(map(str, argv))}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], check=check)


def sanitize(value: str) -> str:
    value = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-._")
    return value or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--push", action="store_true", help="Push the new evidence branch to origin")
    parser.add_argument("--allow-partial", action="store_true", help="Commit PARTIAL_RESULTS for remote diagnosis")
    parser.add_argument("--allow-head-mismatch", action="store_true")
    parser.add_argument("--branch", help="Override the generated evidence branch name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {result_dir}")
    if not result_dir.is_relative_to(REPO_ROOT):
        raise ValueError(f"result directory must be inside the repository: {REPO_ROOT}")

    manifest_path = result_dir / "manifest.json"
    summary_path = result_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise ValueError("result directory is missing manifest.json or summary.json")
    ready = (result_dir / "READY_TO_COMMIT").is_file()
    partial = (result_dir / "PARTIAL_RESULTS").is_file()
    if not ready and not (args.allow_partial and partial):
        raise ValueError("result is not marked READY_TO_COMMIT; pass --allow-partial to commit diagnostic results")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "EXP-0001":
        raise ValueError(f"unexpected experiment: {manifest.get('experiment_id')}")
    target = manifest.get("target", {}).get("checkpoint")
    if target != "Qwen3.6-35B-A3B":
        raise ValueError(f"unexpected checkpoint target: {target}")

    current_head = git("rev-parse", "HEAD").stdout.strip()
    source_head = manifest.get("source", {}).get("commit", "")
    if current_head != source_head and not args.allow_head_mismatch:
        raise ValueError(
            f"HEAD changed after measurement: current {current_head}, measured {source_head}. "
            "Return to the measured commit or pass --allow-head-mismatch with a documented reason."
        )

    tracked_changes = git("status", "--porcelain=v1", "--untracked-files=no").stdout.strip()
    if tracked_changes:
        raise ValueError("tracked source files changed after measurement:\n" + tracked_changes)

    machine = sanitize(str(manifest.get("machine_id", "unknown")))
    run_id = sanitize(str(manifest.get("run_id", result_dir.name)))
    branch = args.branch or f"bench-results/exp-0001-{machine}-{run_id}"

    branch_exists = git("show-ref", "--verify", f"refs/heads/{branch}", check=False).returncode == 0
    if branch_exists:
        raise ValueError(f"local branch already exists: {branch}")

    relative = result_dir.relative_to(REPO_ROOT)
    git("switch", "-c", branch)
    try:
        git("add", "--", str(relative))
        staged = git("diff", "--cached", "--name-only").stdout.splitlines()
        if not staged:
            raise ValueError("no evidence files were staged")
        unexpected = [path for path in staged if not Path(path).is_relative_to(relative)]
        if unexpected:
            raise ValueError(f"refusing to commit files outside result bundle: {unexpected}")
        status_word = "partial" if partial and not ready else "results"
        message = f"bench: add EXP-0001 Qwen3.6 {status_word} ({machine})"
        git("commit", "-m", message)
    except Exception:
        git("switch", "-", check=False)
        git("branch", "-D", branch, check=False)
        raise

    commit = git("rev-parse", "HEAD").stdout.strip()
    if args.push:
        push = git("push", "-u", "origin", branch, check=False)
        if push.returncode != 0:
            print(f"EVIDENCE_BRANCH={branch}")
            print(f"EVIDENCE_COMMIT={commit}")
            print("Push failed; the evidence commit is safe locally.", file=sys.stderr)
            print(push.stderr, file=sys.stderr)
            print(f"Run: git push -u origin {branch}", file=sys.stderr)
            return 3

    print(f"EVIDENCE_BRANCH={branch}")
    print(f"EVIDENCE_COMMIT={commit}")
    print(f"RESULT_DIR={relative}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
