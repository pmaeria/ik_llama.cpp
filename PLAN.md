# GPU-First Qwen Inference Acceleration Plan

## Mission

Extend `ik_llama.cpp` into a faster **single-NVIDIA-GPU runtime for Qwen hybrid MoE models that exceed VRAM**, beginning with:

- **GPU:** NVIDIA RTX 4060 Ti 16 GB, Ada, SM89
- **Primary model:** Qwen3.6-35B-A3B
- **Compatible family target:** Qwen3.5/3.6 hybrid Gated DeltaNet + MoE checkpoints with the same execution structure
- **Workload:** batch-one local inference, especially coding agents and tool-using sessions
- **Typical context:** 8K-32K initially, with 64K+ measured separately
- **Primary objective:** improve end-to-end latency and token generation speed without changing the model's routing or output semantics

The project is **GPU-first**, not CPU-first. System RAM is initially treated as a backing store for model weights that cannot fit in VRAM. CPU execution remains available as a miss/fallback path only when real measurements show that it beats transferring the same work to the GPU.

The initial project is not a new inference runtime. It builds on the existing strengths of `ik_llama.cpp`:

- GGUF loading and mature quant support
- `--fit` and manual tensor placement
- hybrid CPU/GPU execution
- active-expert-only offload
- fused MoE and Qwen Gated DeltaNet paths
- CUDA graph reuse
- prompt/context checkpoints
- existing speculative modes

The central research question is:

> Given a fixed 16 GB VRAM budget, which Qwen expert weights should be resident now, which should be fetched, and how should the GPU execute hits and misses with the least idle time?

---

## Scope

### In scope for the first programme

1. Reproducible on-device benchmarking and profiling.
2. Expert-routing and transfer telemetry with negligible overhead when disabled.
3. A trace-driven cache simulator to establish whether expert locality is exploitable before invasive kernel work.
4. A static hot-expert GPU cache as the first proof of value.
5. A dynamic per-expert GPU cache using stable buffers and device-side indirection.
6. Asynchronous expert promotion, prefetch, and eviction.
7. A calibrated miss policy that chooses between GPU transfer and existing CPU execution.
8. A separate high-throughput prefill mode using layer streaming/double buffering where useful.
9. Qwen/SM89 kernel specialisation only after profiling identifies the actual bottlenecks.
10. An adaptive VRAM budget that accounts for model tensors, compute workspace, context, and expert cache.

### Explicitly deferred

These remain valid future directions, but they must not distract from the initial inference work:

- new weight quantisation formats
- RotorQuant, TurboQuant, SAW-INT4, KIVI, or other new KV codecs
- live Gated DeltaNet state quantisation
- prompt summarisation or lossy context pruning
- Apple Silicon / Metal support
- multi-GPU execution
- high-concurrency serving
- model retraining or routing approximation
- mandatory MTP integration

Existing GGUF weight quants and existing `q8_0`/`q4_0` context-cache options remain benchmark variables. We will preserve clean interfaces so new weight and context codecs can be added later, but no new codec is required to prove the first inference gains.

---

## Non-negotiable engineering rules

### Measure before optimising

Every optimisation must have:

- a named baseline
- a reproducible command
- a fixed model and quant
- a representative workload
- correctness checks
- memory accounting
- before/after timing on the target GPU

No optimisation is accepted on intuition or a synthetic microbenchmark alone.

### Keep output semantics exact

The initial expert-cache work must not:

- alter router logits
- reduce the number of selected experts
- change routing weights
- substitute an approximate expert
- change weight precision depending on cache state
- silently alter sampling settings

The same quantised expert bytes must be used whether an expert is resident, transferred, or executed through the fallback path. Small floating-point differences from operation ordering may occur, but greedy output and logits must be checked against the baseline.

### Optimise end-to-end behaviour

Token generation throughput is important, but the primary score is useful local-agent latency. The benchmark suite must report:

- time to first token
- prompt-processing throughput
- token-generation throughput
- p50/p95 inter-token latency
- request wall time
- peak and steady VRAM
- host RAM and pinned-memory use
- host-to-device bytes per generated token
- GPU utilisation and idle gaps
- cache hit rate and miss cost

An optimisation that improves decode by 10% while making long agent prefills 20% slower is not automatically a win.

### Keep the fast path narrow

The first implementation may be specialised for:

- one CUDA device
- batch size one
- Qwen3.5/3.6 MoE
- supported existing quant layouts
- fixed expert shapes

Generality should be added after the specialised path is proven. Unsupported combinations must fail closed or use the existing path.

---

## Current integration points

The first implementation should reuse current execution and allocation machinery rather than create a parallel model stack.

Likely integration points include:

- `src/llama-load-tensors.cpp`
  - existing `--fit`, CPU-MoE overrides, tensor placement, fused expert tensor construction
- `src/llama-model-loader.cpp`
  - weight provenance, GGUF offsets, host-backed tensors, pinned-memory loading
- `src/llama-expert-io.h`
  - existing expert ranges and deferred expert metadata
- `src/graphs/build_qwen35.cpp`
  - Qwen3.5/3.6 Gated DeltaNet and MoE graph construction
  - calls into `llm_build_std_moe_ffn`
- `src/llama-build-context.cpp` and `src/llama-build-context.h`
  - common MoE graph construction and routing tensors
- `ggml/src/ggml-backend.cpp`
  - active-expert scheduling, backend copies, graph scheduling
- `ggml/src/ggml-cuda/`
  - indirect quantised matmul, fused MoE, routing, copy, and future cache kernels
- `examples/llama-bench/` and `examples/sweep-bench/`
  - repeatable performance and context-length measurements

The existing on-demand tensor-reload machinery is useful prior art for tensor provenance, backend buffer replacement, and graph invalidation. The hot expert cache must not use file hot-swapping in the token-generation loop, but it may reuse or extract lower-level metadata concepts.

---

# Phase 0 - Pin the baseline

## Goal

Create a trusted benchmark and correctness harness before changing execution.

## Model and runtime matrix

Use one pinned Qwen3.6-35B-A3B GGUF checkpoint and one pinned quant around the practical hybrid range, approximately 4.0-4.3 bits per weight. Do not change model or quant while comparing runtime changes.

Record:

- exact model repository and file hash
- exact `ik_llama.cpp` commit
- compiler and CUDA versions
- NVIDIA driver
- CPU and RAM configuration
- PCIe negotiated generation and lane width
- operating system
- all command-line flags

Baseline modes:

1. Current `--fit`, MTP off.
2. Best manually tuned `--n-cpu-moe` or tensor-override configuration, MTP off.
3. Existing active-expert offload on and off.
4. Existing fused MoE on and off.
5. Existing graph reuse on and off.
6. Existing q8 and q4 context cache where they fit.

MTP is disabled for the primary baseline. It is measured later as an optional competitor for the same VRAM.

## Workload matrix

At minimum:

| Workload | Prompt | Output | Purpose |
|---|---:|---:|---|
| Short chat | 512-1K | 256 | steady batch-one decode |
| Code generation | 2K-4K | 512 | structured output and likely expert locality |
| Tool/JSON | 4K-8K | 256 | agent-shaped predictable output |
| Long coding turn | 16K | 512 | realistic TTFT + decode |
| Long agent trace | 32K | 256 | context and repeated-turn behaviour |
| Open prose | 2K | 800 | less predictable decode control |

Run greedy decoding first for determinism, then a fixed sampled configuration for realistic use.

## Required outputs

Create a benchmark artefact that records:

```text
commit
model hash
command
prompt class
prompt tokens
output tokens
PP tok/s
TG tok/s
TTFT
wall time
peak VRAM
host RAM
H2D/D2H bytes
GPU utilisation
```

Suggested repository additions:

```text
benchmarks/qwen35a3b/
  README.md
  prompts/
  run_matrix.py
  parse_results.py
  results-schema.json
```

## Exit gate

Phase 0 is complete when a fresh checkout can reproduce the baseline within an agreed variance band and greedy outputs are stable.

---

# Phase 1 - Add expert-path observability

## Goal

Discover where time and bandwidth are actually going, without changing model output.

## Telemetry to collect

Per layer and globally:

- selected expert IDs
- routing weights
- number of unique selected experts per token/block
- expert reuse distance
- consecutive-token reuse
- per-layer expert frequency
- bytes represented by each selected expert
- CPU-resident versus GPU-resident selection counts
- bytes copied host-to-device
- time waiting for copies
- GPU expert-kernel time
- CPU expert-kernel time
- shared-expert time
- router/top-k time
- Gated DeltaNet time
- full-attention time

Use CUDA events for GPU timing and NVTX ranges for Nsight Systems/Compute. Avoid device synchronisation solely for logging in normal execution. Aggregate counters on device or per request and copy them out after the measured region.

## Trace format

Add an optional machine-readable expert trace. It should contain enough information to simulate cache policies offline without retaining prompts or generated text.

Example logical record:

```json
{
  "token_index": 42,
  "layer": 17,
  "experts": [11, 28, 63, 91, 107, 153, 188, 221],
  "weights": [0.21, 0.18, 0.14, 0.12, 0.11, 0.09, 0.08, 0.07]
}
```

## CLI surface

Tentative flags:

```text
--expert-stats
--expert-trace FILE
--expert-trace-max-tokens N
--expert-nvtx
```

They must be off by default.

## Exit gate

- Disabled telemetry adds no measurable overhead.
- Enabled aggregate telemetry adds less than 2% overhead.
- Trace output is deterministic under greedy decoding.
- We can attribute most token time to named stages.

---

# Phase 2 - Offline cache simulation

## Goal

Prove that a dynamic cache has enough locality to justify invasive runtime work.

Implement a trace simulator for:

- LRU
- segmented LRU
- LFU with ageing
- static top-N per layer
- global static top-N
- recency/frequency hybrid
- layer-weighted budgets
- optional pinning of consistently hot experts

Simulate both count-based and byte-based cache capacities. The real cache is constrained by bytes, not number of experts.

Report:

- expert hit rate
- byte-weighted hit rate
- estimated H2D bytes avoided per token
- compulsory versus capacity misses
- per-layer value of one additional MiB
- cache churn
- expected warm-up duration
- hit rate by workload type

Use measured transfer and kernel costs from the target machine to estimate latency rather than treating every hit as equally valuable.

## Important decision gate

Do not proceed directly to a global dynamic cache if traces show little reuse. Possible pivots include:

- static per-layer hot sets
- next-layer prefetch without persistence
- larger full-layer streaming buffers
- more aggressive existing `--fit` placement
- kernel work rather than cache work

## Exit gate

Proceed when at least one practical cache budget predicts a material reduction in H2D traffic on representative coding-agent traces.

---

# Phase 3 - Static hot-expert GPU cache

## Goal

Build the simplest exact proof that per-expert GPU residency can beat whole-tensor placement.

A static hot set avoids eviction, background copies, and policy races. It validates the graph and kernel changes before adding a dynamic controller.

## Cache unit

Treat an expert as one atomic logical object containing all tensors required to execute it:

- gate/up, or fused gate-up
- down projection
- associated scales/metadata

A cache entry is keyed by at least:

```text
(layer, expert_id, tensor_layout, quant_type, model_generation)
```

Sibling tensors must be promoted and evicted together.

## Preferred first implementation

Use a trace-generated static hot set and fixed VRAM buffers allocated at startup.

For every layer:

1. Router produces original expert IDs and weights.
2. A device-resident map identifies hot hits and cold misses.
3. Hot experts execute from compact GPU-resident storage.
4. Cold experts use the existing exact fallback path.
5. Weighted outputs are combined with the existing shared-expert result.

Do not change the router or reduce expert count.

## Two implementation shapes to test

### A. Layer-local compact hot banks

Each layer receives a small compact bank containing its chosen experts. This is simpler and may work with fewer kernel changes.

### B. Global fixed-size slots with indirection

All layers share one slot pool. This uses VRAM more efficiently but requires kernels to follow `(layer, expert) -> slot` indirection.

Start with A unless the trace simulator shows that fixed per-layer partitioning wastes too much VRAM.

## CUDA graph requirement

Cache buffers and mapping-table addresses must remain stable. Updating cache contents later must not require a new graph shape. The graph may read changing metadata from fixed device buffers.

## Exit gate

On at least two representative workloads:

- decode improves by at least 5% over the best Phase 0 baseline
- prompt processing regresses by less than 2%, or the feature is disabled for prefill
- greedy output matches baseline
- no unbounded memory growth occurs

If the static cache cannot clear this gate, do not build a dynamic LRU yet.

---

# Phase 4 - Dynamic expert cache

## Goal

Turn the static proof into an adaptive cache that learns the current session's routing distribution.

## Design constraints

- All VRAM is allocated up front.
- Slot addresses are stable.
- Entries cannot be evicted while referenced by an in-flight kernel.
- Promotion and eviction are asynchronous where possible.
- A cache miss for the current token must always have an exact fallback.
- Cache policy must be optional and fully bypassable.

## Promotion strategy

The safest initial policy is **promote for future tokens**:

1. Execute the current miss using the existing path.
2. Schedule an asynchronous copy into a cache slot.
3. Mark the entry valid only after a CUDA event completes.
4. Use it on a later token.

This avoids stalling the current token purely to populate the cache. A later policy may decide that immediate transfer is worthwhile.

## Eviction strategy

Start with segmented LRU:

- probationary segment for new entries
- protected segment for repeated hits
- workload/session reset hooks
- optional pinned static entries

Track byte cost and transfer cost, not just access count.

## Device metadata

Maintain fixed-address device tables for:

- expert-to-slot mapping
- slot state
- generation/version
- ready event or epoch
- compact expert index used by kernels

Host policy decisions may update these tables asynchronously, but the token path should not depend on a CPU round trip when all selected experts hit.

## Tentative CLI

```text
--expert-cache-mib N
--expert-cache-policy off|static|lru|slru
--expert-cache-warmup-tokens N
--expert-cache-static-map FILE
--expert-cache-stats
```

## Exit gate

- Dynamic cache matches or beats the best static policy across mixed workloads.
- It does not regress a no-locality workload materially.
- It survives long agent sessions without stale mappings, corruption, or memory leaks.
- CUDA graph reuse remains effective.

---

# Phase 5 - Calibrated GPU miss execution

## Goal

For a non-resident selected expert, decide whether to:

1. copy it to the GPU and execute there,
2. use the existing CPU path,
3. split independent misses between CPU and GPU so both work concurrently.

This adopts the useful FreeToken principle without replacing `ik_llama.cpp`'s execution stack.

## Machine calibration

At startup or through a separate calibration tool, measure:

- pinned H2D latency and bandwidth for real expert sizes
- pageable H2D fallback
- GPU quantised expert GEMV/GEMM latency at relevant token counts
- CPU expert latency at relevant thread counts
- overlap efficiency between CPU execution, H2D copies, and GPU kernels
- costs for fused gate-up and down layouts

Persist calibration by hardware, driver, build, model signature, and quant.

## GPU-first policy

The default policy should favour GPU execution when:

```text
copy_time + gpu_compute_time < cpu_compute_time
```

but the real scheduler must account for overlap and queue state. Several misses from the same token are independent and may be partitioned.

## Streams

Likely stream structure:

- main model/compute stream
- expert-copy stream A
- expert-copy stream B or promotion stream
- events connecting copied experts to compute

Avoid global synchronisation. Use fixed staging buffers and preallocated workspaces.

## Exit gate

The adaptive miss policy must outperform the best fixed strategy (`always transfer` or `always CPU`) on a mixed benchmark set. If it cannot, keep the simpler fixed winner.

---

# Phase 6 - Prefill-specific expert streaming

## Goal

Prevent a decode-optimised cache from harming prompt processing.

During large prefills, many or nearly all experts may be selected across the batch. A small LRU can churn and provide little value. The runtime should treat prefill and single-token decode as different execution regimes.

## Initial strategy

At a configurable token-count threshold:

- bypass decode-cache insertion
- reserve two layer-sized GPU staging buffers
- stream the next layer's required expert data while the current layer computes
- use pinned host memory
- restore or retain the decode cache after prefill according to measured cost

Possible policies:

```text
retain: keep decode cache and use separate prefill buffers
shrink: temporarily donate cold cache slots to prefill workspace
rebuild: release decode contents and warm from subsequent decode routing
```

Start with `retain` because it is easiest to reason about. Add elastic reuse only if memory pressure requires it.

## Exit gate

- Long-prompt throughput is no worse than baseline.
- Decode begins without an excessive cache-rewarm penalty.
- TTFT improves on representative 8K-32K prompts or the mode remains disabled.

---

# Phase 7 - Qwen/SM89 GPU kernel specialisation

## Goal

Optimise only the kernels proven dominant after the memory system is working.

Candidate targets include:

1. Fused router projection + softmax/top-k for batch one.
2. Expert-hit remapping and compact dispatch.
3. Fused quantised gate-up + SiLU + multiply for Qwen expert shapes.
4. Fused/combined down projection and weighted reduction.
5. Overlap of shared expert with routed experts.
6. Packed multi-expert GEMV for one-token decode.
7. Qwen Gated DeltaNet projection/update fusion if it remains a major cost.
8. Reduced host scheduling and launch overhead.

The existing `mmq_id`, fused MoE, and DeltaNet implementations are the starting point. New kernels are justified only by profiler traces and must include a fallback for unsupported quant/layout combinations.

## Kernel policy

Optimise for the actual target first:

```text
SM89
batch = 1
Qwen3.6-35B-A3B dimensions
8 selected experts
existing practical GGUF quant
```

Do not compromise correctness or maintainability to support every architecture in the first patch.

## Exit gate

Each kernel lands independently with:

- isolated correctness tests
- microbenchmarks
- end-to-end impact
- no regression on the fallback path

---

# Phase 8 - Adaptive VRAM controller

## Goal

Make `--fit` account for the expert cache as a first-class GPU resource.

The final VRAM budget is shared by:

```text
fixed non-expert model tensors
resident expert tensors selected by existing placement
new dynamic expert cache
KV cache
Gated DeltaNet state
compute buffers
CUDA graph allocations
optional speculative state
safety margin
```

## Initial implementation

Do not dynamically change KV precision or context length. Compute a safe startup budget:

1. Reserve fixed model and compute requirements.
2. Reserve the requested context and current cache precision.
3. Reserve graph/safety headroom.
4. Assign the remaining budget to expert-cache slots.
5. Refuse or reduce the cache cleanly if the budget is insufficient.

Tentative UX:

```text
--expert-cache-mib auto
--expert-cache-min-mib N
--expert-cache-max-mib N
--gpu-fit-margin N
```

## Later extension

Once stable, the controller may compare the marginal value of:

- one more expert-cache slot
- more context
- higher KV precision
- MTP state
- larger prefill workspace

That is a later optimisation, not a prerequisite for the first cache.

---

# Speculation policy

MTP is deliberately outside the critical path of the first programme.

Reasons:

- it consumes extra model/cache/state memory
- recurrent checkpoint storage grows with speculative depth
- Qwen MoE verification can touch a larger union of experts
- high acceptance does not guarantee a speedup
- the same VRAM may produce more value as expert-cache capacity

After Phase 6, benchmark these as competitors:

1. no speculation
2. suffix or n-gram speculation
3. MTP with `n_max=1`
4. autotuned shallow MTP with a strict VRAM cap
5. self-speculation followed by one-token MTP fallback

The controller should retain MTP only when it improves end-to-end latency on the current workload. It is not enabled simply because the checkpoint contains an MTP head.

---

# Quantisation and context-compression extension points

No new quantisation work is required in the initial phases. However, avoid designs that make it impossible later.

## Weight/expert codec boundary

The expert cache must copy and execute experts through metadata that includes:

```text
quant type
block size
row layout
scale layout
fused or split gate/up layout
bytes per expert
supported CUDA kernel
```

This leaves room for future IQ, QTIP/EXL3-like, ParoQuant, or other expert representations without redesigning the cache policy.

## Context codec boundary

Future work may add a context-cache abstraction for:

- q8/q4 baselines
- rotated INT4
- TurboQuant/Rotor-derived codecs
- mixed K/V precision
- cold recurrent-checkpoint compression

The first inference patches should not couple expert-cache metadata to a specific KV representation.

## Context-management boundary

Semantic Gated DeltaNet checkpoints and block-based prompt reuse remain valuable future work, but should be developed after GPU expert execution is measured and stable.

---

# Proposed source layout

Names are tentative and should follow repository conventions after the first implementation review.

```text
src/
  llama-expert-cache.h
  llama-expert-cache.cpp
  llama-expert-policy.h
  llama-expert-policy.cpp
  llama-expert-profile.h
  llama-expert-profile.cpp

ggml/src/ggml-cuda/
  expert-cache.cu
  expert-cache.cuh

examples/
  expert-cache-bench/

benchmarks/
  qwen35a3b/

tests/
  test-expert-cache.cpp
  test-expert-policy.cpp
```

Keep policy, storage, and CUDA execution separate:

- **profile** records facts
- **policy** decides residency and miss handling
- **cache** owns fixed buffers and lifetime
- **CUDA kernels** execute mapped experts

---

# Correctness strategy

## Unit-level

- expert ID remapping
- slot allocation/eviction
- event/epoch handling
- sibling tensor atomicity
- quant block offsets
- cache hit/miss partitioning
- weighted output merge
- policy decisions from fixed calibration inputs

## Model-level

For short deterministic prompts:

- compare router selections against baseline
- compare per-layer MoE outputs within tolerance
- compare final logits
- compare greedy token sequence

Test:

- no hits
- all hits
- mixed hits/misses
- repeated eviction
- cache disabled
- context reuse
- graph reuse
- long-running server requests

## Failure behaviour

On allocation, unsupported-layout, or cache-consistency failure:

- log a precise reason
- disable the optimisation or fall back for that request
- never continue with an incomplete expert tuple
- never silently change routing

---

# Performance acceptance gates

These are project gates, not promised results.

## Instrumentation

- less than 2% overhead when aggregate stats are enabled
- no measurable overhead when disabled

## Static cache

- at least 5% decode improvement on two representative workloads
- less than 2% prefill regression, or automatic prefill bypass

## Dynamic cache

- beats static allocation on a mixed/session-changing trace
- no material regression on a low-locality trace

## Miss controller

- beats the best fixed CPU/GPU miss policy on the mixed suite

## Project success target

A successful first release should aim for:

- **15-30% lower representative end-to-end latency** versus the best current `ik_llama.cpp` configuration on the same model/quant/hardware, or
- a comparably strong decode gain with no TTFT/prefill regression.

A larger gain is possible if current execution is transfer-bound, but it must not be assumed.

---

# Key risks and planned mitigations

## Expert locality is weaker than expected

Mitigation: Phase 2 simulator is a hard decision gate. Pivot to prefetch or kernels if persistent caching is not valuable.

## Cache indirection slows GPU hits

Mitigation: prove a static layer-local bank first; fuse mapping into the dispatch kernel only after measurement.

## CUDA graph recapture removes the gain

Mitigation: allocate fixed buffers and fixed-address maps up front. Update contents and metadata in place.

## Prefill churn overwhelms the cache

Mitigation: use separate prefill mode and disable cache insertion for large batches.

## Pinned host memory helps prefill but harms decode or system stability

Mitigation: measure separately, cap pinned allocation, and expose an explicit fallback.

## Quant layouts make per-expert slices awkward

Mitigation: support one known Qwen GGUF layout first, validate block alignment, and add layout descriptors before generalisation.

## Cache VRAM displaces more valuable resources

Mitigation: report marginal gain per MiB and make cache allocation explicit. MTP is disabled first under pressure.

## Floating-point accumulation order changes greedy output

Mitigation: preserve operation order where practical, compare intermediate outputs, and keep an exact fallback.

---

# Milestone sequence

1. **M0 - Baseline harness**
   - pinned checkpoint, scripts, schema, reproducible measurements
2. **M1 - Expert telemetry**
   - routing traces, CUDA timings, H2D counters, NVTX
3. **M2 - Cache simulator**
   - offline policies and predicted value per MiB
4. **M3 - Static hot cache**
   - fixed hot experts, exact hot/cold split, end-to-end proof
5. **M4 - Dynamic cache**
   - fixed slot pool, asynchronous promotion, safe eviction
6. **M5 - Calibrated miss scheduler**
   - GPU transfer versus CPU fallback, concurrent execution where useful
7. **M6 - Prefill mode**
   - double-buffered layer streaming and cache-bypass policy
8. **M7 - Qwen/SM89 kernels**
   - only profiler-proven bottlenecks
9. **M8 - Adaptive VRAM budget**
   - integration with `--fit` and safe automatic sizing
10. **M9 - Optional extensions**
    - speculation, context codecs, recurrent checkpoint compression, new quants

Each milestone should be reviewable and benchmarkable independently. Avoid one giant branch that combines policy, kernels, quantisation, and context changes.

---

# First concrete implementation task

The first code change after this plan should be **telemetry only**:

1. Add per-layer selected-expert counters.
2. Add host-to-device expert-byte counters.
3. Add optional per-token expert trace output.
4. Add CUDA-event timing around the routed expert path.
5. Add a benchmark parser that produces a compact JSON summary.

This gives the next coding agent a measurable target and prevents us from building a cache that the routing traces do not justify.

---

## Definition of done for the initial programme

The GPU-first inference programme is complete when:

- Qwen3.6-35B-A3B runs stably on the 16 GB RTX 4060 Ti with the pinned GGUF quant.
- The expert cache and miss policy are optional and have clean fallbacks.
- Greedy correctness and routing semantics are preserved.
- Long coding-agent traces show stable memory use.
- Benchmark results are reproducible from repository scripts.
- The best new configuration materially beats the best pre-change `ik_llama.cpp` configuration on the target machine.
- Quantisation and context-codec interfaces remain open, but no unproven quant method is required for the result.
