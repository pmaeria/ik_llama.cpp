# Qwen3.6-35B-A3B GPU Experiments

This directory is the reproducible handoff between code changes made through GitHub and measurements made on the target RTX 4060 Ti 16 GB machine.

The active checkpoint target is **Qwen3.6-35B-A3B**. The runner rejects filenames that identify Qwen3.5. The source tree may still use the internal architecture name `qwen35moe`; that does not change the checkpoint requirement.

## EXP-0001

EXP-0001 establishes the best current non-speculative `ik_llama.cpp` baseline before inference changes.

It collects:

- exact Git, model, and binary fingerprints;
- OS, CPU, RAM, GPU, driver, CUDA, and PCIe information;
- native `llama-bench` JSON for PP, TG, 8K-context TG, and 32K-context TG;
- interleaved configuration rounds;
- deterministic `llama-cli` output hashes;
- raw stdout/stderr and normalized summaries;
- periodic GPU utilization, VRAM, clock, temperature, and power samples.

The default matrix compares current `--fit` behaviour with q8 KV, active-expert-only offload disabled, fused MoE disabled, and graph reuse disabled. Optional candidates may fail or run out of memory; `fit-default` is required.

## Files

```text
experiments.json          experiment definitions
run_experiment.py         build-independent runner and evidence collector
commit_results.py         creates and pushes an append-only evidence branch
compare_results.py        compares normalized result bundles
results-schema.json       manifest schema
prompts/                  deterministic correctness prompts
LOCAL_AGENT_HANDOFF.md    authoritative instructions for the local coding agent
```

## Evidence layout

```text
evidence/
  EXP-0001/
    <machine-id>/
      <timestamp>-<source-commit>/
        manifest.json
        hardware.json
        runs.jsonl
        summary.json
        correctness.json
        gpu_samples.csv
        agent_report.md
        raw/
        READY_TO_COMMIT
```

Evidence is append-only. Never overwrite an older run. Each bundle identifies the source commit and model SHA-256 it measured.

## Manual invocation

The local agent should follow [`LOCAL_AGENT_HANDOFF.md`](LOCAL_AGENT_HANDOFF.md). A direct baseline invocation is:

```bash
python3 benchmarks/qwen36-35b/run_experiment.py \
  --suite baseline \
  --model /path/to/Qwen3.6-35B-A3B-....gguf \
  --machine-id paul-4060ti \
  --build-dir build
```

When the runner prints `STATUS=complete` or `STATUS=complete_with_optional_failures`, commit it with:

```bash
python3 benchmarks/qwen36-35b/commit_results.py \
  --result-dir /absolute/path/printed/by/the/runner \
  --push
```

A blocked bundle may still be pushed for diagnosis with `--allow-partial`.
