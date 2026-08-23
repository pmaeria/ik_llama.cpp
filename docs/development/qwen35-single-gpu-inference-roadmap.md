# Qwen 35B Single-GPU Inference Roadmap

Status: design proposal

Primary target:

- Model: Qwen3.6-35B-A3B and architecture-compatible Qwen3.5/3.6 MoE checkpoints
- Hardware: one NVIDIA RTX 4060 Ti 16 GB (Ada, SM89)
- Workload: one interactive request at a time, batch size 1, text generation and coding-agent traffic
- Backend: CUDA
- Baseline: current `ik_llama.cpp`, MTP disabled, existing GGUF quantization unchanged

## Objective

Improve end-to-end single-request latency beyond the best current `ik_llama.cpp` configuration for Qwen3.6-35B-A3B on a 16 GB consumer GPU.

The first project is deliberately **GPU-first**:

- CUDA executes the model computation.
- System RAM is a backing store for expert weights that cannot fit in VRAM.
- The initial design does not split MoE computation between CPU and GPU.
- The initial design does not depend on MTP.
- New weight or context quantization formats are out of the critical path, but the design must not prevent them later.

The central hypothesis is that the current runtime can be improved by turning expert placement from a mostly load-time decision into an adaptive, persistent GPU-residency system.

## Existing foundation

The fork already provides most of the difficult general runtime machinery:

- `--fit` and `--fit-margin` for automatic VRAM fitting;
- full-tensor and MoE-specific CPU/GPU placement controls;
- `only_active_experts` scheduling, which can transfer only selected expert slices;
- fused MoE and merged up/gate paths;
- CUDA graph reuse;
- fused Gated DeltaNet support;
- Linux page-cache expert prefetch;
- `llama-bench` and `llama-sweep-bench`;
- speculative paths, including MTP, which remain available as optional experiments.

The missing capability is a persistent cache of individual host-resident experts in VRAM, with asynchronous miss handling and a decode path designed around fixed GPU buffers.

This roadmap borrows architectural ideas from FreeToken and TokenSpeed, but the implementation should be native to `ik_llama.cpp` and preserve its GGUF, quantization, scheduler, and CUDA abstractions.

## Non-goals for the initial project

The following are explicitly deferred:

- Apple Metal or MLX support;
- multiple GPUs;
- high-concurrency serving and continuous batching;
- CPU execution of selected expert misses;
- a new weight quantization format;
- RotorQuant, TurboQuant, or another new KV codec;
- live Gated DeltaNet state quantization;
- MTP as a required acceleration path;
- aggressive prompt or KV-token pruning;
- support for every MoE architecture in the first implementation.

The first implementation may recognize one exact architecture signature and fall back to the existing generic path for everything else.

## Performance contract

Every performance comparison must keep these constant:

- checkpoint and GGUF files;
- weight quantization;
- context size and cache types;
- prompt and requested output length;
- sampling settings;
- number of parallel slots;
- generated greedy tokens for correctness tests.

A change is not a win if it silently reduces precision, context, output length, or model quality.

Primary metrics:

- prompt-processing tokens/s;
- time to first token;
- generated tokens/s;
- p50 and p95 inter-token latency;
- peak and steady VRAM;
- host-to-device bytes per generated token;
- time blocked on host-to-device transfers;
- GPU kernel time and GPU idle time;
- expert-cache hit rate and reuse distance;
- end-to-end latency of a recorded coding-agent trace.

Initial success gates:

1. Instrumentation disabled: zero measurable regression.
2. Instrumentation enabled: less than 1% median throughput overhead.
3. Static expert caching: a repeatable generation improvement, or a clear trace-based reason to stop before building a dynamic cache.
4. Dynamic expert caching: at least 15% median TG improvement over the best current configuration on two representative workloads, with no more than 3% prompt-processing regression.
5. Agent trace: at least 10% end-to-end latency improvement without output or stability regressions.

These are project gates, not promises. If expert locality is too weak to satisfy them, the project should pivot to transfer overlap and Qwen-specific CUDA kernels rather than preserve a failed cache design.

# Phase 0: Reproducible GPU baseline

Before adding a cache, establish the best existing configuration on the target machine.

## Benchmark matrix

Run at least:

| Prompt | Output | Purpose |
|---:|---:|---|
| 512 | 512 | short interactive generation |
| 8K | 512 | normal coding-agent turn |
| 32K | 512 | long-context agent turn |
| recorded trace | natural | real end-to-end behaviour |

Use several output shapes:

- deterministic code generation;
- JSON or tool-call output;
- open prose;
- repeated editing or continuation, where expert locality may differ.

Compare the best combinations of existing options, including:

- `--fit` and explicit MoE placement;
- `--fit-margin`;
- active-expert transfer on and off;
- fused MoE;
- merged up/gate experts;
- graph reuse;
- pinned versus non-pinned host weights where relevant;
- q8/q4 cache settings already supported by the runtime;
- MTP off for the primary baseline.

## Deliverables

Create a small benchmark package under `tools/qwen35-gpu-bench/` containing:

- a manifest describing hardware, driver, CUDA, commit, checkpoint, GGUF hashes, and flags;
- repeatable benchmark scripts;
- a JSON result schema;
- correctness prompts and expected greedy token hashes;
- optional Nsight Systems launch scripts;
- a report generator comparing revisions.

The harness must report prompt processing and generation separately. It must not use a warmed prefix unless that condition is explicit in the result.

# Phase 1: Expert-routing and transfer observability

Do not implement caching until the runtime can answer where the time and bytes go.

## Runtime telemetry

Behind an opt-in flag, record:

- selected expert IDs by layer and token;
- frequency of each `(layer, expert)` pair;
- reuse distance in tokens and in routed expert accesses;
- number of distinct experts selected per layer over moving windows;
- bytes copied from host to device for each expert tensor;
- copy launch time, completion time, and scheduler wait time;
- time spent in shared-expert, routed-expert, and reduction kernels;
- CUDA graph capture and fallback events;
- GPU-resident expert layers versus host-resident expert layers;
- prompt-processing and decoding statistics separately.

Telemetry should be streamable to a compact binary or JSON-lines trace. Full per-token tracing must be optional; aggregate counters should be cheap enough for ordinary benchmarking.

## Offline cache simulator

Build a trace replay tool before writing the cache. For candidate budgets from roughly 256 MiB to 4 GiB, simulate:

- existing whole-layer placement;
- per-layer static hot sets;
- global static hot sets;
- per-layer LRU or CLOCK;
- global segmented LRU or CLOCK;
- session-local versus cross-session profiles.

The simulator must account for an **expert bundle**, not an abstract expert ID. A cache entry includes all tensors required to execute that expert, such as gate/up and down weights, in their actual GGUF types and byte sizes.

It must also account for a crucial trade-off: reserving VRAM for individual expert slots means `--fit` may keep fewer complete expert layers resident. The simulator should compare total predicted transfer bytes for both layouts rather than treating cache memory as free.

## Decision gate

Proceed to the GPU cache only if traces show one or more of:

- strong repeated use of a manageable hot set;
- useful locality within an interactive session;
- a large difference between static per-layer and current whole-layer placement;
- enough transfer stall that overlap can plausibly improve latency even at modest hit rates.

If locality is weak, skip directly to asynchronous staging and kernel specialization.

# Phase 2: Static GPU expert mirror

The first behavioural change should be a static cache, not an LRU.

## Design

At model initialization:

1. Reserve a fixed VRAM budget before `--fit` finalizes tensor placement.
2. Let `--fit` place always-active and whole-layer tensors using the reduced budget.
3. Allocate fixed-address CUDA buffers for a configured number of expert bundles.
4. Populate the buffers from a saved routing profile or a short warm-up trace.
5. Build an immutable `(layer, expert) -> slot` map for the request or server lifetime.

Only experts whose full layer remains in host memory need cache entries. Fully GPU-resident expert layers continue through the existing path.

## Execution

For each MoE layer during batch-one decoding:

- classify selected experts as resident-full-layer, cached, or cold;
- execute resident and cached experts from VRAM;
- use the existing active-expert transfer path for cold experts;
- combine partial outputs exactly as the existing MoE graph does.

Prompt processing should initially bypass the individual-expert cache. Large prompt batches often activate most experts, and a decode-oriented cache can reduce PP performance if it consumes workspace or causes extra graph splits.

## CLI proposal

All names are provisional:

```text
--moe-gpu-cache-mib N
--moe-gpu-cache-policy static
--moe-gpu-cache-profile FILE
--moe-gpu-cache-stats FILE
```

The feature remains disabled by default.

## Implementation constraint

No allocation, tensor construction, or buffer-address change may occur inside the token loop. Static addresses are required for predictable latency and later CUDA graph capture.

# Phase 3: Dynamic GPU expert cache

After the static proof works, allow entries to change between and during requests.

## Cache unit

Cache one complete expert bundle per slot. For Qwen3.6-35B-A3B this normally means a shared slot index across the packed gate/up and down pools.

The cache manager should preserve source metadata:

- layer and expert ID;
- `ggml_type`;
- block size and strides;
- packed byte ranges for each required tensor;
- source host pointer;
- destination slot offsets;
- last-use and frequency counters.

This avoids hard-coding Q4 and leaves the door open for current and future GGUF quants.

## GPU memory layout

Use preallocated pools:

- persistent gate/up expert slots;
- persistent down expert slots;
- one or two transient miss-staging banks sized for the maximum experts selected by one token;
- GPU-resident mapping and remap tables;
- fixed scratch and reduction buffers.

Start with per-layer quotas or a segmented global policy. A naive global LRU may evict an expert before the next token reaches the same layer, so the trace simulator must choose the first policy.

## Miss pipeline

The initial miss path remains GPU-compute-only:

1. Router/top-k produces selected expert IDs.
2. A small classification step separates cache hits and misses and remaps hit IDs to slots.
3. Copy miss bundles from pinned host memory to the fixed staging bank on a dedicated CUDA copy stream.
4. While copies run, compute the shared expert and cached/resident routed experts.
5. Wait on CUDA events only when the miss kernels need the staged weights.
6. Compute misses on the GPU.
7. Accumulate hit, miss, and shared-expert outputs exactly.
8. Promote selected misses into persistent slots according to the policy.

No selected expert is computed on the CPU in this phase.

## Policies

Implement in this order:

1. static profile;
2. per-layer CLOCK;
3. segmented global CLOCK;
4. LRU only if traces show it is worth the extra bookkeeping;
5. optional frequency/recency hybrid across request boundaries.

All policies operate on fixed buffers. Eviction changes metadata and contents, not pointers.

## Failure behaviour

On an unsupported tensor layout, quant type, model signature, or CUDA capability:

- log one clear reason;
- disable the cache;
- use the existing execution path;
- never silently return an approximate result.

# Phase 4: Separate prefill and decode strategies

A single expert-memory policy is unlikely to be best for both phases.

## Decode mode

Use persistent individual-expert residency, small fixed staging banks, and batch-one kernels.

## Prefill mode

For sufficiently large prompts:

- bypass individual-expert promotion;
- stream complete expert tensors or complete expert layers through one or two fixed GPU buffers;
- overlap transfer of the next layer with computation of the current layer when graph dependencies permit;
- retain enough workspace for the fastest existing PP kernels;
- collect routing statistics to seed the decode cache when prefill ends.

At the PP-to-TG transition, populate the decode hot set from the recent prompt profile without reallocating the cache pools.

The benchmark gate is end-to-end time to first token. A decode cache is not useful if reserving it makes a normal coding-agent prompt materially slower.

# Phase 5: Qwen/SM89 decode specialization

Only after the memory path is measured and stable should the project specialize kernels.

## Architecture profile

Introduce an internal profile selected by an explicit architecture signature, for example:

```text
architecture: qwen35moe-compatible
layers: 40 main text layers
experts: 256
experts selected: 8
hidden size: 2048
expert intermediate size: 512
hybrid Gated DeltaNet/full-attention pattern: compatible
hardware: CUDA SM89
workload: batch 1 decode
```

The profile must reject incompatible shapes rather than trusting a model name.

## Candidate optimizations

Profile first, then consider:

- fused router, top-k, cache-hit classification, and slot remapping;
- a batch-one quantized expert GEMV path for the exact Qwen shapes;
- fused cached-hit and staged-miss reduction;
- persistent CUDA graphs containing the fixed cache/staging buffers;
- fewer host synchronizations between routing, copies, and expert execution;
- parallel Gated DeltaNet input projections if they remain a measured bottleneck;
- Qwen-specific fusion around Gated DeltaNet gates, normalization, and state update only where the existing fused path leaves measurable overhead.

Do not replace a mature generic kernel with a specialized kernel until Nsight traces and A/B tests show a win on SM89.

## Quantization boundary

The first optimized kernel may target one existing, widely used GGUF type to prove the path, but the cache API must expose type/block/stride metadata and retain the generic fallback. Creating a new quantizer is a separate project.

# Phase 6: GPU-memory autotuning

Once the cache and kernels work, add an opt-in startup tuner.

The tuner should measure or infer:

- available VRAM after model, context, and worst-case graph reservation;
- host-to-device bandwidth for the actual pinned expert representation;
- selected-expert transfer latency;
- GPU cached-expert and staged-expert kernel throughput;
- routing locality from a short calibration trace;
- expected prompt versus generation mix.

It then chooses:

- cache budget;
- whole-layer placement versus individual slots;
- per-layer slot quotas;
- static versus CLOCK policy;
- number of copy streams and staging banks;
- prefill bypass threshold.

Autotuning should happen at startup or request boundaries, never by reallocating GPU buffers in the middle of generation.

A future flag might be:

```text
--moe-gpu-cache-auto
```

The selected plan and the reasons for it must be printed in a machine-readable form.

# Deferred research lanes

These remain intentionally possible but should not distract from the initial GPU inference work.

## Existing and new weight quants

The cache should copy and execute packed GGUF expert data without converting it to a private fixed format. This keeps compatibility with current IK/IQ/K quants and allows later experiments with QTIP-, ParoQuant-, or other rotation-based formats.

A future mixed-precision policy could keep frequently used or sensitive experts at higher precision, but precision must be fixed per expert. Output must never depend on whether an expert happened to be cached.

## Context/KV compression

Future work may add SAW-like INT4, TurboQuant, Rotor-derived, or compressed-domain attention codecs. This is separate from the expert cache and should use its own correctness and performance gates.

## Recurrent-state checkpoints

Older inactive Gated DeltaNet checkpoints may eventually use BF16, INT8, or mixed precision. The live recurrent state remains at the model's expected precision until long-generation quality is demonstrated.

## Speculation

N-gram or suffix speculation is cheap enough to benchmark alongside the final engine. MTP remains optional because it consumes VRAM and can be slower despite high acceptance. The GPU-memory controller may later let speculation compete with expert slots for memory, but the expert-cache project must stand on its own with MTP disabled.

# Expected code seams

The exact design will be refined during Phase 1, but likely integration points are:

- `common/common.h` and `common/common.cpp`: CLI and configuration;
- `include/llama.h`: public/internal parameter plumbing;
- `src/llama-model-loader.cpp` and model placement code: reserve cache VRAM before `--fit` and register source expert ranges;
- `src/llama-build-context.cpp`: expose/cache-aware MoE graph construction;
- `ggml/src/ggml-backend.cpp`: selected-expert scheduling, asynchronous copy dependencies, and instrumentation;
- `ggml/src/ggml-moe-prefetch.*`: reuse source-range and host-residency knowledge without conflating OS page prefetch with VRAM caching;
- `ggml/src/ggml-cuda/mmq_id.cu` and adjacent CUDA MoE files: cached-slot and staged-miss execution;
- `examples/llama-bench` and `examples/sweep-bench`: reproducible metrics and cache counters;
- new focused files such as `src/llama-moe-gpu-cache.*` and `ggml/src/ggml-cuda/moe-cache.*` rather than expanding unrelated scheduler code indefinitely.

# Patch sequence

Keep each pull request independently benchmarkable and revertible:

1. Roadmap and benchmark specification.
2. Benchmark harness and correctness corpus.
3. Expert-routing/transfer telemetry.
4. Offline cache simulator and profile format.
5. Static GPU expert mirror.
6. Dynamic fixed-buffer cache and CLOCK policy.
7. Asynchronous miss staging and hit/shared-expert overlap.
8. Prefill/decode policy split.
9. Qwen35/SM89 fused routing and expert path.
10. Startup autotuner.

Each patch must preserve the feature-off baseline and include before/after results on the target 4060 Ti.

# First coding milestone

The first implementation milestone is **not** the cache. It is a trace and benchmark package that can answer:

1. How many expert bytes are transferred per generated token today?
2. How much generation time is blocked on those transfers?
3. What hit rate would static, per-layer, and global caches achieve at 256 MiB, 512 MiB, 1 GiB, 2 GiB, and 4 GiB?
4. How much whole-layer residency must be surrendered for each cache budget?
5. Does the routing distribution remain stable across code, JSON/tool calls, and prose?
6. Is the larger opportunity persistent residency, transfer/computation overlap, or a Qwen-specific CUDA kernel?

Only those measurements should decide the second implementation milestone.