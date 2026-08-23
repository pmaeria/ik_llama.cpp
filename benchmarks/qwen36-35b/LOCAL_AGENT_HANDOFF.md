# Local Agent Handoff — EXP-0001

## Your role

You are the local execution agent on the RTX 4060 Ti machine. Execute this handoff completely, collect the evidence bundle, commit it on a dedicated results branch, and push it to `origin`.

Do not redesign the benchmark, edit source files, tune kernels, or interpret the result. Your output is the pushed evidence branch and commit. Analysis happens later in the web environment.

## Exact target

- checkpoint: **Qwen3.6-35B-A3B**
- GPU: **one NVIDIA RTX 4060 Ti 16 GB**
- backend: CUDA
- runtime branch: `plan/qwen-single-gpu-inference-roadmap`
- experiment: `EXP-0001`
- primary suite: `baseline`
- MTP: disabled

The source code may contain an internal architecture identifier such as `qwen35moe`. That does not authorize use of a Qwen3.5 checkpoint.

### Never use

- Qwen3.5-35B-A3B
- a dense 27B checkpoint
- a non-GGUF checkpoint
- a different GPU
- an MTP/speculative command
- a newly downloaded model when an existing Qwen3.6-35B-A3B GGUF is available

## Completion contract

A successful handoff ends with these values printed by `commit_results.py`:

```text
EVIDENCE_BRANCH=...
EVIDENCE_COMMIT=...
RESULT_DIR=...
```

Push that branch. In your final response report only:

```text
STATUS=<complete | complete_with_optional_failures | blocked>
EVIDENCE_BRANCH=<branch>
EVIDENCE_COMMIT=<sha>
RESULT_DIR=<path>
MODEL_FILE=<basename only>
MODEL_SHA256=<sha256>
```

Do not summarize benchmark conclusions.

---

# 1. Prepare a clean source checkout

Start from the repository containing this file.

```bash
git fetch origin --prune
git switch plan/qwen-single-gpu-inference-roadmap
git pull --ff-only origin plan/qwen-single-gpu-inference-roadmap
```

Record the source commit:

```bash
git rev-parse HEAD
```

Check the working tree:

```bash
git status --short
```

If tracked files are modified, do **not** stash, discard, or overwrite the user's work. Create a clean sibling worktree instead:

```bash
git fetch origin --prune
git worktree add ../ik_llama-exp0001 origin/plan/qwen-single-gpu-inference-roadmap
cd ../ik_llama-exp0001
```

Untracked build directories are acceptable. Do not begin measurements with tracked source changes.

Confirm that the active plan names Qwen3.6:

```bash
grep -n "Qwen3.6-35B-A3B" PLAN.md benchmarks/qwen36-35b/experiments.json
```

A PowerShell agent should use the equivalent `git` commands and `Select-String`.

---

# 2. Verify the machine

Run:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader
```

Continue only when the selected device is an RTX 4060 Ti with approximately 16 GB VRAM. Do not include another CUDA device through `CUDA_VISIBLE_DEVICES`.

When several NVIDIA GPUs are installed, identify the RTX 4060 Ti and run the experiment with only that device visible. On Linux:

```bash
export CUDA_VISIBLE_DEVICES=<index-of-4060-ti>
```

On PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = "<index-of-4060-ti>"
```

Do not use a remote or virtual GPU.

---

# 3. Locate the existing Qwen3.6-35B-A3B GGUF

Search the user's normal model locations first. Do not search the whole root filesystem blindly.

Typical Linux search:

```bash
find "$HOME" /mnt /media /data /models \
  -type f \( -iname '*Qwen*3.6*35B*A3B*.gguf' -o -iname '*Qwen3.6*35B*.gguf' \) \
  2>/dev/null
```

Typical PowerShell search:

```powershell
Get-ChildItem $HOME,D:\,E:\ -Filter *.gguf -File -Recurse -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match 'Qwen.*3\.6.*35B' }
```

Reject any result whose name identifies Qwen3.5.

When the model is split, use the first GGUF shard, normally containing `00001-of-...`.

When multiple exact Qwen3.6-35B-A3B GGUFs exist:

1. Inspect existing launch scripts, shell history, UI configurations, or recent commands to identify the model already used on this machine.
2. Prefer the practical approximately 4-bit model already used for hybrid inference.
3. Do not select a model solely because its filename sorts first.
4. Do not download a replacement merely to make the experiment easier.
5. Record the chosen basename in your execution notes.

Set an absolute path:

```bash
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-....gguf
```

PowerShell:

```powershell
$MODEL = "D:\absolute\path\to\Qwen3.6-35B-A3B-....gguf"
```

The runner rejects filenames that identify Qwen3.5. When an existing file omits `3.6` or `35B` from its filename, independently verify its GGUF metadata before using `--allow-model-name-mismatch`. Document that verification in `--notes`.

If no existing Qwen3.6-35B-A3B GGUF can be found, stop. Do not substitute Qwen3.5. Report the missing model rather than downloading tens of gigabytes without instruction.

---

# 4. Build the measured binaries

Use a Release CUDA build from the exact checked-out commit.

Linux or macOS shell with NVIDIA CUDA available:

```bash
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j "$(nproc)" \
  --target llama-bench llama-cli
```

PowerShell:

```powershell
cmake -S . -B build -DGGML_CUDA=ON
cmake --build build --config Release -j --target llama-bench llama-cli
```

If the generator does not expose individual targets, build all targets:

```bash
cmake --build build --config Release -j
```

Confirm both binaries exist. Common locations are:

```text
build/bin/llama-bench
build/bin/llama-cli
build/bin/Release/llama-bench.exe
build/bin/Release/llama-cli.exe
```

Run the benchmark help once and confirm the build has these options:

```bash
build/bin/llama-bench --help
```

Required concepts include JSON output, `--fit`, `--fit-margin`, fused MoE, graph reuse, cache types, and offload-only-active-experts.

Do not run an older binary from another checkout.

---

# 5. Run the smoke suite

Use Python 3.10 or newer. On Windows, replace `python3` with `py -3` when necessary.

Linux example:

```bash
python3 benchmarks/qwen36-35b/run_experiment.py \
  --suite smoke \
  --model "$MODEL" \
  --machine-id paul-4060ti \
  --build-dir build \
  --notes "EXP-0001 smoke; exact Qwen3.6-35B-A3B GGUF"
```

PowerShell example:

```powershell
py -3 benchmarks/qwen36-35b/run_experiment.py `
  --suite smoke `
  --model $MODEL `
  --machine-id paul-4060ti `
  --build-dir build `
  --notes "EXP-0001 smoke; exact Qwen3.6-35B-A3B GGUF"
```

Expected terminal ending:

```text
RESULT_DIR=...
STATUS=complete
```

`complete_with_optional_failures` is also acceptable, although the smoke suite normally has no optional configuration.

## One permitted OOM recovery

If and only if `fit-default` fails from CUDA out-of-memory, rerun smoke once with a larger safety margin:

```bash
python3 benchmarks/qwen36-35b/run_experiment.py \
  --suite smoke \
  --model "$MODEL" \
  --machine-id paul-4060ti \
  --build-dir build \
  --fit-margin-mib 1536 \
  --notes "EXP-0001 smoke; fit margin raised to 1536 MiB after default OOM"
```

Use the same 1536 MiB margin for the baseline suite if this recovery succeeds.

Do not change model quant, context lengths, batch sizes, active expert count, or experiment JSON to recover from OOM.

If smoke remains blocked, proceed to section 8 and commit the latest partial bundle for diagnosis.

---

# 6. Run the baseline suite

Use the fit margin that passed smoke: normally 1024 MiB, or 1536 MiB after the single permitted recovery.

Normal command:

```bash
python3 benchmarks/qwen36-35b/run_experiment.py \
  --suite baseline \
  --rounds 2 \
  --model "$MODEL" \
  --machine-id paul-4060ti \
  --build-dir build \
  --fit-margin-mib 1024 \
  --notes "EXP-0001 baseline; Qwen3.6-35B-A3B; MTP disabled"
```

PowerShell uses the same arguments with PowerShell line continuations.

The suite intentionally attempts optional controls. An optional configuration may OOM, time out, or fail to load. The runner records it and continues. Do not manually remove an optional result.

Do not use the `extended` suite in this handoff.

---

# 7. Validate the evidence bundle

The runner prints an absolute `RESULT_DIR`. Inspect that exact baseline directory.

Required files:

```text
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

When GPU sampling was unavailable, `gpu_samples.csv` may be absent; the hardware probe must still record why.

Validate with Python:

```bash
python3 - <<'PY'
import json
from pathlib import Path

result = Path("PASTE_RESULT_DIR_HERE")
manifest = json.loads((result / "manifest.json").read_text())
summary = json.loads((result / "summary.json").read_text())
correctness = json.loads((result / "correctness.json").read_text())

assert manifest["experiment_id"] == "EXP-0001"
assert manifest["target"]["checkpoint"] == "Qwen3.6-35B-A3B"
assert manifest["target"]["mtp"] is False
assert manifest["ready_to_commit"] is True
assert (result / "READY_TO_COMMIT").is_file()
assert correctness["success"] is True
assert summary["required_successful_invocations"] == summary["required_expected_invocations"]
print(manifest["status"])
print(manifest["model"]["filename"])
print(manifest["model"]["sha256"])
PY
```

For PowerShell, put the same Python code in a temporary `.py` file or use `py -3 -c`.

Do not edit generated JSON to make validation pass.

---

# 8. Commit and push

For a successful baseline bundle:

```bash
python3 benchmarks/qwen36-35b/commit_results.py \
  --result-dir "PASTE_RESULT_DIR_HERE" \
  --push
```

The helper:

1. verifies the bundle target is Qwen3.6-35B-A3B;
2. verifies `HEAD` still matches the measured source commit;
3. refuses tracked source changes;
4. creates a dedicated `bench-results/...` branch;
5. stages only the selected result directory;
6. commits it;
7. pushes it to `origin`.

If smoke or baseline is blocked, commit the latest diagnostic bundle instead:

```bash
python3 benchmarks/qwen36-35b/commit_results.py \
  --result-dir "PASTE_RESULT_DIR_HERE" \
  --allow-partial \
  --push
```

Do not add `.bench-cache`, model files, build outputs, or unrelated logs to the commit.

If the helper creates the commit but push authentication fails, run the exact `git push -u origin <branch>` command it prints. Do not recreate the measurement.

---

# 9. Return control to the web environment

After push succeeds, obtain the manifest fields:

```bash
python3 - <<'PY'
import json
from pathlib import Path
result = Path("PASTE_RESULT_DIR_HERE")
m = json.loads((result / "manifest.json").read_text())
print("STATUS=" + m["status"])
print("MODEL_FILE=" + m["model"]["filename"])
print("MODEL_SHA256=" + m["model"]["sha256"])
PY
```

Combine those with the `EVIDENCE_BRANCH`, `EVIDENCE_COMMIT`, and `RESULT_DIR` printed by the commit helper.

Do not analyze which configuration won. The next web turn will fetch the branch, inspect raw failures and summaries, and decide EXP-0002.
