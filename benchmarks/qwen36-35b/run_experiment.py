#!/usr/bin/env python3
"""Run EXP-0001 for Qwen3.6-35B-A3B and write a commit-ready evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_SPEC = HERE / "experiments.json"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._") or "unknown"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def run(argv: Sequence[str], timeout: int | float = 60) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [str(x) for x in argv],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "argv": [str(x) for x in argv],
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            "seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "argv": [str(x) for x in argv],
            "returncode": None,
            "stdout": out,
            "stderr": err,
            "timed_out": True,
            "seconds": time.monotonic() - started,
        }
    except OSError as exc:
        return {
            "argv": [str(x) for x in argv],
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "timed_out": False,
            "seconds": time.monotonic() - started,
        }


def git(*args: str) -> str:
    result = run(["git", *args], 30)
    return result["stdout"].strip() if result["returncode"] == 0 else ""


def source_info() -> dict[str, Any]:
    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "describe": git("describe", "--always", "--dirty", "--tags"),
        "remote": git("remote", "get-url", "origin"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def sha256(path: Path, cache: Path | None = None) -> str:
    key = f"{path.resolve()}|{path.stat().st_size}|{path.stat().st_mtime_ns}"
    values: dict[str, str] = {}
    if cache and cache.exists():
        try:
            values = json.loads(cache.read_text(encoding="utf-8"))
            if re.fullmatch(r"[0-9a-f]{64}", values.get(key, "")):
                return values[key]
        except (OSError, json.JSONDecodeError):
            values = {}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        values[key] = result
        write_json(cache, values)
    return result


def find_binary(build: Path, name: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    names = [name, name + ".exe"]
    candidates = [build / "bin" / n for n in names] + [build / "bin" / "Release" / n for n in names] + [build / "Release" / n for n in names]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    found = [path for n in names for path in build.rglob(n)] if build.exists() else []
    if found:
        return sorted(found, key=lambda p: ("Release" not in p.parts, len(p.parts), str(p)))[0].resolve()
    raise FileNotFoundError(f"could not find {name} under {build}")


def validate_model(path: Path, spec: dict[str, Any], allow_mismatch: bool) -> None:
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError(f"model must be an existing GGUF: {path}")
    name = path.name.lower().replace("_", "-")
    target = spec["target"]
    rejected = [x.lower().replace("_", "-") for x in target.get("rejected_filename_terms", [])]
    if any(x in name for x in rejected):
        raise ValueError(f"refusing Qwen3.5 model for Qwen3.6 experiment: {path.name}")
    required = [x.lower().replace("_", "-") for x in target.get("required_filename_terms", [])]
    missing = [x for x in required if x not in name]
    if missing and not allow_mismatch:
        raise ValueError(
            f"filename does not verify Qwen3.6-35B-A3B; missing {missing}: {path.name}. "
            "Use --allow-model-name-mismatch only after checking model metadata."
        )


def physical_threads() -> int:
    logical = os.cpu_count() or 4
    return max(1, logical // 2 if logical > 4 else logical)


def memory_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    return None


def probe(argv: Sequence[str]) -> dict[str, Any]:
    result = run(argv, 30)
    return {
        "argv": result["argv"],
        "returncode": result["returncode"],
        "stdout": result["stdout"][:100_000],
        "stderr": result["stderr"][:20_000],
    }


def hardware(machine: str, bench: Path, cli: Path, source: dict[str, Any]) -> dict[str, Any]:
    probes = {
        "nvidia_smi_list": ["nvidia-smi", "-L"],
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pci.bus_id,pstate,clocks.max.sm,power.limit",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_smi_pcie": [
            "nvidia-smi",
            "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_topology": ["nvidia-smi", "topo", "-m"],
        "nvcc": ["nvcc", "--version"],
        "cmake": ["cmake", "--version"],
    }
    if sys.platform.startswith("linux"):
        probes["lscpu"] = ["lscpu"]
        probes["uname"] = ["uname", "-a"]
    return {
        "captured_at": now(),
        "machine_id": machine,
        "hostname_sha256_prefix": hashlib.sha256(platform.node().encode()).hexdigest()[:16],
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "threads_used": physical_threads(),
        "memory_bytes": memory_bytes(),
        "source": source,
        "binaries": {
            "llama_bench": {"filename": bench.name, "sha256": sha256(bench), "size_bytes": bench.stat().st_size},
            "llama_cli": {"filename": cli.name, "sha256": sha256(cli), "size_bytes": cli.stat().st_size},
        },
        "probes": {name: probe(argv) for name, argv in probes.items()},
    }


def sample_gpu(stop: threading.Event, path: Path, interval: float) -> None:
    fields = [
        "timestamp",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "clocks.current.sm",
        "clocks.current.memory",
        "pstate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*fields, "error"])
        while not stop.is_set():
            result = run(
                ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                max(5, interval * 4),
            )
            if result["returncode"] == 0:
                for line in result["stdout"].splitlines():
                    if line.strip():
                        writer.writerow([*next(csv.reader([line])), ""])
            else:
                writer.writerow([*([""] * len(fields)), result["stderr"][:500]])
            handle.flush()
            stop.wait(interval)


def parse_bench_json(text: str) -> list[dict[str, Any]]:
    values = [text.strip()]
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        values.append(text[a : b + 1])
    errors: list[str] = []
    for value in values:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return parsed
    raise ValueError("; ".join(errors[-2:]) or "no JSON array found")


def render(values: list[str], margin: int) -> list[str]:
    return [str(x).replace("{fit_margin_mib}", str(margin)) for x in values]


def normalize_cli(stdout: str, prompt: str) -> str:
    value = ANSI.sub("", stdout).replace("\r", "")
    if prompt.strip() in value:
        value = value.replace(prompt.strip(), "", 1)
    ignored = ("llama_", "ggml_", "common_", "system_info:", "main:", "build:", "print_info:")
    lines = [line.rstrip() for line in value.splitlines() if not line.strip().startswith(ignored)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())


def correctness(
    cli: Path,
    model: Path,
    prompt_names: list[str],
    cli_args: list[str],
    raw: Path,
    context: int,
    tokens: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    help_result = run([str(cli), "--help"], 60)
    help_text = help_result["stdout"] + help_result["stderr"]
    items: list[dict[str, Any]] = []
    for prompt_name in prompt_names:
        prompt_path = HERE / "prompts" / prompt_name
        prompt = prompt_path.read_text(encoding="utf-8")
        argv = [str(cli), "-m", str(model), *cli_args, "-c", str(context), "-n", str(tokens)]
        if "--seed" in help_text:
            argv += ["--seed", str(seed)]
        if "--temp" in help_text:
            argv += ["--temp", "0"]
        if "--no-display-prompt" in help_text:
            argv += ["--no-display-prompt"]
        if "--simple-io" in help_text:
            argv += ["--simple-io"]
        if "--no-conversation" in help_text:
            argv += ["--no-conversation"]
        if "--file" in help_text:
            argv += ["-f", str(prompt_path)]
        else:
            argv += ["-p", prompt]
        result = run(argv, timeout)
        stem = clean_id(prompt_path.stem)
        write_text(raw / f"{stem}.stdout.txt", result["stdout"])
        write_text(raw / f"{stem}.stderr.txt", result["stderr"])
        normalized = normalize_cli(result["stdout"], prompt)
        write_text(raw / f"{stem}.normalized.txt", normalized + ("\n" if normalized else ""))
        items.append(
            {
                "prompt": prompt_name,
                "command": argv,
                "returncode": result["returncode"],
                "timed_out": result["timed_out"],
                "seconds": result["seconds"],
                "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                "normalized_length": len(normalized),
                "normalized_preview": normalized[:500],
                "success": result["returncode"] == 0 and not result["timed_out"] and bool(normalized),
            }
        )
    return {"captured_at": now(), "runs": items, "success": bool(items) and all(x["success"] for x in items)}


def summarize(invocations: list[dict[str, Any]], required: set[str]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    success_count: dict[str, int] = defaultdict(int)
    for item in invocations:
        if item["status"] != "ok":
            failures.append({key: item.get(key) for key in ("config_id", "round", "status", "returncode", "timed_out", "stderr_file", "parse_error")})
            continue
        success_count[item["config_id"]] += 1
        for record in item["records"]:
            try:
                groups[(item["config_id"], str(record.get("test", "unknown")))].append(float(record["avg_ts"]))
            except (KeyError, TypeError, ValueError):
                pass
    rows: list[dict[str, Any]] = []
    by_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (config_id, test), values in sorted(groups.items()):
        row = {
            "config_id": config_id,
            "test": test,
            "records": len(values),
            "avg_ts_values": values,
            "median_avg_ts": statistics.median(values),
            "mean_avg_ts": statistics.fmean(values),
            "min_avg_ts": min(values),
            "max_avg_ts": max(values),
            "stdev_avg_ts": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        rows.append(row)
        by_test[test].append(row)
    best = {
        test: {"config_id": winner["config_id"], "median_avg_ts": winner["median_avg_ts"]}
        for test, candidates in by_test.items()
        if (winner := max(candidates, key=lambda x: x["median_avg_ts"]))
    }
    return {
        "groups": rows,
        "best_by_test": best,
        "failures": failures,
        "successful_invocations_by_config": dict(success_count),
        "required_config_failures": sorted(x for x in required if success_count.get(x, 0) == 0),
    }


def report(manifest: dict[str, Any], summary: dict[str, Any], check: dict[str, Any]) -> str:
    lines = [
        "# EXP-0001 Result",
        "",
        f"- Status: **{manifest['status']}**",
        "- Target: **Qwen3.6-35B-A3B**",
        f"- Machine: `{manifest['machine_id']}`",
        f"- Source commit: `{manifest['source']['commit']}`",
        f"- Model: `{manifest['model']['filename']}`",
        f"- Model SHA-256: `{manifest['model']['sha256']}`",
        "",
        "| Config | Test | Median t/s | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("groups", []):
        lines.append(f"| `{row['config_id']}` | `{row['test']}` | {row['median_avg_ts']:.3f} | {row['min_avg_ts']:.3f} | {row['max_avg_ts']:.3f} |")
    lines += ["", f"Correctness: **{'pass' if check.get('success') else 'fail'}**", ""]
    for item in check.get("runs", []):
        lines.append(f"- `{item['prompt']}`: {'pass' if item['success'] else 'fail'} `{item['normalized_sha256']}`")
    return "\n".join(lines) + "\n"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", choices=("smoke", "baseline", "extended"), default="baseline")
    parser.add_argument("--machine-id", default="paul-4060ti")
    parser.add_argument("--experiment-file", default=str(DEFAULT_SPEC))
    parser.add_argument("--build-dir", default=str(ROOT / "build"))
    parser.add_argument("--bench-bin")
    parser.add_argument("--cli-bin")
    parser.add_argument("--output-root", default=str(ROOT / "evidence"))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--fit-margin-mib", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--test", action="append", dest="tests")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--allow-model-name-mismatch", action="store_true")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if not shutil.which("git"):
        raise RuntimeError("git is required")
    spec = json.loads(Path(args.experiment_file).expanduser().read_text(encoding="utf-8"))
    suite = spec["suites"][args.suite]
    defaults = spec["defaults"]
    model = Path(args.model).expanduser().resolve()
    validate_model(model, spec, args.allow_model_name_mismatch)
    build = Path(args.build_dir).expanduser().resolve()
    bench = find_binary(build, "llama-bench", args.bench_bin)
    cli = find_binary(build, "llama-cli", args.cli_bin)
    source = source_info()
    if not source["commit"]:
        raise RuntimeError("not inside a Git repository")

    configs = args.configs or list(suite["configs"])
    tests = args.tests or list(suite["tests"])
    unknown_configs = set(configs) - set(spec["configs"])
    unknown_tests = set(tests) - set(spec["tests"])
    if unknown_configs or unknown_tests:
        raise ValueError(f"unknown configs={sorted(unknown_configs)} tests={sorted(unknown_tests)}")
    rounds = args.rounds or int(suite.get("rounds", defaults["rounds"]))
    margin = args.fit_margin_mib or int(defaults["fit_margin_mib"])
    timeout = args.timeout or int(defaults["timeout_seconds"])
    machine = clean_id(args.machine_id)

    result = Path(args.output_root).expanduser().resolve() / spec["experiment_id"] / machine / f"{stamp()}-{source['commit'][:10]}"
    counter = 1
    base = result
    while result.exists():
        result = Path(str(base) + f"-{counter}")
        counter += 1
    raw_bench = result / "raw" / "bench"
    raw_cli = result / "raw" / "correctness"
    raw_bench.mkdir(parents=True)
    raw_cli.mkdir(parents=True)

    print(f"Hashing/verifying model: {model.name}", flush=True)
    model_hash = sha256(model, ROOT / ".bench-cache" / "qwen36-model-hashes.json")
    hw = hardware(machine, bench, cli, source)
    write_json(result / "hardware.json", hw)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "status": "running",
        "created_at": now(),
        "run_id": result.name,
        "machine_id": machine,
        "suite": args.suite,
        "rounds": rounds,
        "configs": configs,
        "tests": tests,
        "fit_margin_mib": margin,
        "notes": args.notes,
        "source": source,
        "target": spec["target"],
        "model": {"filename": model.name, "sha256": model_hash, "size_bytes": model.stat().st_size},
        "binaries": hw["binaries"],
        "result_path": str(result.relative_to(ROOT)) if result.is_relative_to(ROOT) else str(result),
    }
    write_json(result / "manifest.json", manifest)

    test_args = ["-p", "0", "-n", "0"]
    for test in tests:
        test_args += spec["tests"][test]["args"]
    common = [
        "-m", str(model),
        "-b", str(defaults["batch_size"]),
        "-ub", str(defaults["ubatch_size"]),
        "-t", str(physical_threads()),
        "-r", str(defaults["repetitions_per_process"]),
        "-w", "1" if defaults["warmup"] else "0",
        "-o", "json", "-oe", "none",
        *test_args,
    ]

    stop = threading.Event()
    sampler: threading.Thread | None = None
    if not args.no_gpu_sampling and shutil.which("nvidia-smi"):
        sampler = threading.Thread(
            target=sample_gpu,
            args=(stop, result / "gpu_samples.csv", float(defaults["gpu_sample_interval_seconds"])),
            daemon=True,
        )
        sampler.start()

    invocations: list[dict[str, Any]] = []
    rng = random.Random(int(defaults["seed"]))
    try:
        for round_id in range(1, rounds + 1):
            order = list(configs)
            rng.shuffle(order)
            for config_id in order:
                cfg = spec["configs"][config_id]
                argv = [str(bench), *common, *render(cfg["bench_args"], margin)]
                stem = f"round-{round_id:02d}__{clean_id(config_id)}"
                print(f"[{round_id}/{rounds}] {config_id}", flush=True)
                outcome = run(argv, timeout)
                stdout_path = raw_bench / f"{stem}.stdout.json"
                stderr_path = raw_bench / f"{stem}.stderr.txt"
                write_text(stdout_path, outcome["stdout"])
                write_text(stderr_path, outcome["stderr"])
                item: dict[str, Any] = {
                    "captured_at": now(),
                    "config_id": config_id,
                    "required": bool(cfg.get("required")),
                    "round": round_id,
                    "command": argv,
                    "returncode": outcome["returncode"],
                    "timed_out": outcome["timed_out"],
                    "seconds": outcome["seconds"],
                    "stdout_file": str(stdout_path.relative_to(result)),
                    "stderr_file": str(stderr_path.relative_to(result)),
                    "status": "failed",
                    "records": [],
                }
                if outcome["returncode"] == 0 and not outcome["timed_out"]:
                    try:
                        item["records"] = parse_bench_json(outcome["stdout"])
                        item["status"] = "ok" if item["records"] else "empty"
                    except ValueError as exc:
                        item["status"] = "parse_error"
                        item["parse_error"] = str(exc)
                elif outcome["timed_out"]:
                    item["status"] = "timeout"
                invocations.append(item)
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=5)

    write_text(result / "runs.jsonl", "".join(json.dumps(x, sort_keys=True) + "\n" for x in invocations))
    required = {x for x in configs if spec["configs"][x].get("required")}
    summary = summarize(invocations, required)
    required_runs = [x for x in invocations if x["config_id"] in required]
    required_ok = (
        len(required_runs) == len(required) * rounds
        and all(x["status"] == "ok" and len(x["records"]) >= len(tests) for x in required_runs)
    )
    summary.update(
        {
            "experiment_id": spec["experiment_id"],
            "suite": args.suite,
            "captured_at": now(),
            "required_expected_invocations": len(required) * rounds,
            "required_successful_invocations": sum(x["status"] == "ok" for x in required_runs),
        }
    )

    if args.skip_correctness:
        check = {"captured_at": now(), "skipped": True, "success": True, "runs": []}
    elif required_ok:
        check = correctness(
            cli,
            model,
            list(suite["correctness_prompts"]),
            render(spec["correctness"]["base_cli_args"], margin),
            raw_cli,
            int(defaults["correctness_context"]),
            int(defaults["correctness_tokens"]),
            int(defaults["seed"]),
            timeout,
        )
    else:
        check = {"captured_at": now(), "skipped": True, "reason": "required benchmark failed", "success": False, "runs": []}

    optional_failures = [x for x in summary["failures"] if x["config_id"] not in required]
    ready = required_ok and check["success"]
    status = "complete_with_optional_failures" if ready and optional_failures else "complete" if ready else "blocked"
    manifest.update({"status": status, "completed_at": now(), "ready_to_commit": ready, "optional_failures": len(optional_failures)})
    write_json(result / "manifest.json", manifest)
    write_json(result / "summary.json", summary)
    write_json(result / "correctness.json", check)
    write_text(result / "agent_report.md", report(manifest, summary, check))
    write_text(result / ("READY_TO_COMMIT" if ready else "PARTIAL_RESULTS"), status + "\n")

    print(f"RESULT_DIR={result}")
    print(f"STATUS={status}")
    if ready:
        print(f"NEXT=python3 {HERE / 'commit_results.py'} --result-dir {result} --push")
        return 0
    print("Commit with --allow-partial for remote diagnosis.")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
