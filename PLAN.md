# GPU-First Qwen3.6-35B-A3B Inference Acceleration Plan

## Mission

Extend `ik_llama.cpp` into a faster single-NVIDIA-GPU runtime for **Qwen3.6-35B-A3B** when the model exceeds VRAM.

Initial target:

- checkpoint: Qwen3.6-35B-A3B only
- GPU: NVIDIA RTX 4060 Ti 16 GB, Ada SM89
- runtime: CUDA, one device, one active inference context
- workload: batch-one local inference, especially coding agents and tool-using sessions
- context: 8K-32K first; 64K+ measured separately
- model format: one pinned GGUF and quant for each experiment
- speculation: disabled for the primary baseline
- objective: reduce end-to-end latency while preserving exact routing and model semantics

The repository may internally call the shared implementation `qwen35moe`; that is an architecture label. The checkpoint under test must be **Qwen3.6-35B-A3B**, never a Qwen3.5 checkpoint.

This is a GPU-first programme. System RAM is backing storage for weights that do not fit in VRAM. The first implementation keeps selected-expert computation on the GPU. New CPU/GPU co-execution policy is deferred until the best GPU-oriented path has been measured.

The central question is:

> Given a fixed 16 GB VRAM budget, which Qwen3.6 expert weights should be resident, which should be staged, and how can CUDA execute hits and misses without idle gaps or graph recapture?

## Existing foundation

The project builds on current `ik_llama.cpp` rather than creating a parallel model stack:

- GGUF loading and IK/IQ/K quant support
- `--fit` and explicit tensor placement
- hybrid CPU/GPU model placement
- active-expert-only offload
- fused MoE paths
- Qwen Gated DeltaNet support
- CUDA graph reuse
- prompt and recurrent checkpoints
- existing speculative modes
- `llama-bench` and `sweep-bench`

The likely missing opportunity is a graph-compatible GPU expert residency and staging layer that is more granular than whole expert tensors or whole layers.

---

# Non-negotiable rules

## Keep comparisons exact

Performance comparisons must keep constant:

- checkpoint and GGUF hash
- tensor types and runtime repacking state
- context size and K/V cache types
- prompt and output lengths
- sampling settings
- active expert count and routing weights
- number of slots and devices

The cache must use the same quantized expert bytes whether an expert is resident, staged, or handled by the existing fallback. Cache state must not change precision or routing.

## Measure end-to-end behaviour

Every result must report, where applicable:

- prompt-processing tokens/s
- token-generation tokens/s
- time to first token
- request wall time
- p50 and p95 inter-token latency
- peak and steady VRAM
- host RAM and pinned-memory use
- H2D bytes per generated token
- H2D wait time and overlap
- GPU kernel time and idle gaps
- CUDA graph capture/replay status
- cache hit rate, admission rate, churn, and miss cost

A decode gain that materially worsens realistic coding-agent prefill is not automatically accepted.

## Fixed-shape graph before dynamic policy

Stable buffer addresses are necessary but not sufficient for CUDA graph reuse. The optimized decode graph must also retain a fixed topology and fixed launch shapes.

The first viable design should assume:

```text
8 routed lanes per MoE layer
8 hit lanes
8 miss/staging lanes
fixed masks
fixed mapping tables
fixed output and reduction buffers
```

IDs, masks, epochs, and slot mappings may change. Graph node count, buffer addresses, and tensor shapes must not change in the token loop.

## Hard stop conditions

Do not continue a complex cache implementation merely because it appears in this roadmap. Every stage has a decision gate and a cheaper pivot.

---

# Primary hypotheses

## H1 - Current decode is materially limited by expert movement

Measure selected expert bytes, H2D copy time, scheduler waits, and GPU idle gaps.

Stop persistent-cache work when a perfect elimination of avoidable expert-transfer stalls predicts less than roughly 8-10% end-to-end improvement.

## H2 - Routing locality is valuable at realistic VRAM budgets

Simulate cache budgets only after subtracting the complete expert layers and workspace displaced by those budgets.

Proceed when a practical budget predicts material H2D savings on more than one representative workload.

## H3 - Compact dispatch is cheaper than the transfers it avoids

A high theoretical hit rate is insufficient. Remapping, fragmented launches, masking, reduction, events, and graph constraints must be included in the cost model.

The one-layer CUDA proof must beat the equivalent uncached path before model-wide work.

## H4 - Decode gains survive prompt-heavy agent workloads

Benchmark synthetic PP/TG separately and recorded multi-turn coding-agent traces end to end.

Keep prefill and decode policies separate when required.

---

# Scope

## Initial programme

1. Reproducible local benchmark and evidence protocol.
2. Exact checkpoint, quant, build, driver, and hardware fingerprinting.
3. Roofline and transfer microbenchmarks.
4. Expert-routing, copy, and CUDA-path telemetry.
5. Offline placement and cache simulation.
6. A one-layer fixed-shape CUDA feasibility proof.
7. Profile-guided whole-layer placement and static hot experts.
8. Fixed staging buffers and asynchronous GPU miss execution.
9. Dynamic expert caching only when static evidence justifies it.
10. Separate prefill and decode use of the same VRAM pool.
11. Qwen3.6/SM89 kernel specialization only for measured bottlenecks.
12. Integration with `--fit` and startup autotuning.

## Deferred

- Qwen3.5 checkpoints
- Apple Metal or MLX
- multiple GPUs
- high-concurrency serving
- new CPU/GPU q-star-style co-execution
- model retraining or approximate routing
- new weight quantization formats
- RotorQuant, TurboQuant, SAW-INT4, KIVI, or other new KV codecs
- live Gated DeltaNet state quantization
- lossy prompt or KV-token pruning
- mandatory MTP

Existing weight and context quants remain benchmark variables. Interfaces should leave room for future codecs without making them prerequisites.

---

# Evidence loop

The web/GitHub environment produces code and experiment definitions. The RTX 4060 Ti is the measurement authority.

```text
commit experiment
    -> local agent follows handoff
    -> build and run on RTX 4060 Ti
    -> collect normalized evidence
    -> commit and push result branch
    -> analyse evidence here
    -> commit next experiment
```

Every evidence bundle is append-only and identifies:

- experiment ID
- source commit
- baseline commit
- model SHA-256
- exact command and flags
- OS, CPU, RAM, GPU, PCIe, CUDA, driver, compiler
- cold/warm state
- raw output and normalized summary

The local-agent protocol lives at:

```text
benchmarks/qwen36-35b/LOCAL_AGENT_HANDOFF.md
```

---

# M0 - Exact baseline and external references

## Goal

Establish the best existing `ik_llama.cpp` configuration before runtime changes.

Pin:

- one Qwen3.6-35B-A3B GGUF filename and SHA-256
- one source checkpoint revision
- whether MTP tensors exist, while leaving MTP disabled
- exact tensor quant/layout profile
- merged versus unmerged gate/up state
- runtime repacking state
- context and cache types
- `n_batch`, `n_ubatch`, thread count, and fit margin

Initial workload set:

| Test | Prompt | Output | Purpose |
|---|---:|---:|---|
| PP | 512 | 0 | short prefill |
| TG | 0 | 256 | steady batch-one decode |
| GP | 8K | 256 | ordinary agent context |
| GP | 32K | 256 | long agent context |
| CLI correctness | fixed prompts | 96-128 | deterministic output hashes |

Compare at least:

- current `--fit` default candidate
- tighter and safer fit margins where they load
- active-expert-only offload enabled versus disabled
- fused MoE enabled versus disabled
- graph reuse enabled versus disabled
- conservative existing K/V cache alternatives where useful

Benchmark comparable FreeToken or other runtimes only when the same checkpoint representation and semantics make the comparison honest.

## Exit gate

A fresh checkout and local agent can reproduce the selected baseline within a documented variance band, and deterministic outputs remain stable.

---

# M1 - Architecture audit, roofline, and telemetry

## Roofline first

For the pinned GGUF, calculate:

- bytes in one complete expert bundle
- host-resident MoE layers under each candidate placement
- selected expert bytes per generated token
- pinned and pageable H2D bandwidth
- per-copy launch latency
- cached and staged expert kernel latency
- available overlap with shared expert, Gated DeltaNet, and other work
- complete expert residency displaced by each candidate cache budget

Estimate upper bounds for:

```text
perfect cache
perfect transfer overlap
realistic simulated cache
```

## Runtime telemetry

Behind opt-in flags, collect:

- selected expert IDs and routing weights
- reuse distance measured globally and per visit to the same layer
- expert frequency and second-touch rate
- host/GPU residency at selection time
- H2D bytes, start/end times, and wait time
- routed and shared expert kernel time
- router/top-k time
- Gated DeltaNet time
- full-attention time
- graph capture, replay, and fallback events

Use CUDA events and NVTX. Do not synchronize the device solely for logging during normal execution.

## Exit gate

Most decode time can be attributed to named stages, and the perfect-transfer-removal bound justifies further work.

---

# M2 - Offline placement and cache simulation

The simulator must compare more than eviction policies:

1. current whole-layer placement
2. profile-guided whole-layer placement
3. static per-layer hot experts
4. transfer staging without persistence
5. per-layer cache with second-touch admission
6. global or segmented cache
7. CLOCK, segmented LRU, and frequency/recency hybrids

Use byte capacities, not expert counts. Include:

- expert bundle sizes
- whole layers displaced by cache reservation
- compulsory and capacity misses
- admission and eviction traffic
- host memory and PCIe cost
- compact-dispatch overhead estimate
- warm-up duration
- hit rate and value per MiB by workload

## Decision tree

```text
better whole-layer placement sufficient? -> stop there
transfer overlap sufficient?             -> stop there
static hot experts sufficient?           -> stop there
session changes justify dynamic policy?  -> build dynamic cache
```

---

# M3 - One-layer fixed-shape CUDA proof

Before a model-wide cache, build a synthetic or isolated Qwen3.6 MoE-layer test.

Prove all hit counts from zero to eight using the same graph topology:

- cache-slot and staging-slot indirection
- fixed masks and eight routed lanes
- expert 0 and repeated-ID edge cases
- atomic gate/up/down sibling handling
- exact weighted reduction
- no synchronous router D2H read
- no allocation or tensor construction in the token loop
- CUDA graph replay without recapture
- output agreement with the current MoE path

## Exit gate

The all-hit and mixed hit/miss paths are correct and at least one realistic mixture is faster than the corresponding baseline operation.

---

# M4 - Model-wide static optimization

Implement and compare independently:

- profile-guided complete-layer placement
- layer-local static hot banks
- global fixed static slots when per-layer partitioning wastes too much VRAM
- fixed miss-staging buffers without persistent admission

Prompt processing initially bypasses cache promotion.

## Exit gate

At least two representative workloads improve materially without unacceptable PP regression. Otherwise pivot to transfer overlap or kernels and do not build a dynamic cache.

---

# M5 - Asynchronous GPU miss pipeline

Initial miss semantics:

```text
cache hit  -> GPU resident slot
cache miss -> fixed staging bank -> GPU execution
fallback   -> existing exact runtime path
```

No new CPU executor is introduced here.

Use:

- one main compute stream
- one or two expert-copy/promotion streams
- fixed staging buffers
- CUDA events rather than global synchronization
- shared-expert and hit computation while misses transfer
- mapping epochs so a slot cannot be reused while an earlier graph reads it

Promotions may be committed one token later to avoid a CPU round trip in the critical path.

---

# M6 - Dynamic cache, only if justified

Admission is as important as eviction. Do not insert every miss.

Candidate first policy:

```text
per-layer or segmented pool
second-touch admission
CLOCK or segmented-LRU eviction
protected and probationary entries
no promotion during large prefill
optional pinned static hot set
```

All storage is preallocated. Eviction changes metadata and contents, never graph shapes or buffer addresses.

## Exit gate

Dynamic policy beats the best static/transfer-only result on session-changing traces and does not materially regress low-locality workloads.

---

# M7 - Prefill/decode use of the VRAM pool

Prefill and decode are different regimes.

For large prefills:

- bypass decode-cache admission
- use the cache pool as one or two layer/expert streaming banks where safe
- overlap next-layer transfer with current-layer computation
- preserve enough workspace for the fastest PP kernels
- collect recent routing to seed decode state

At the phase transition, reuse preallocated memory without process restart or mid-token allocation.

Start with a conservative retained-cache policy. Add elastic repurposing only after correctness and memory accounting are stable.

---

# M8 - Qwen3.6/SM89 kernel specialization

Only optimize profiler-proven residual bottlenecks.

Candidate targets:

- fused router, softmax/top-k, hit classification, and slot remapping
- fixed-width batch-one quantized expert GEMV
- fused cached-hit/staged-miss weighted reduction
- overlap of shared and routed experts
- reduced host launch and scheduler overhead
- Qwen Gated DeltaNet projection/update fusion where still dominant

The first optimized kernel may support one exact existing GGUF type and layout. Unsupported types must use the generic fallback.

---

# M9 - `--fit` integration and autotuning

Treat VRAM as a single budget shared by:

```text
fixed non-expert tensors
whole resident expert tensors
expert cache/staging pool
K/V cache
Gated DeltaNet state
compute workspace
CUDA graph allocations
safety margin
optional speculative state
```

At startup, choose:

- whole-layer versus granular expert residency
- expert pool size
- per-layer quotas
- staging-bank count
- prefill threshold
- policy mode

Never reallocate graph-visible buffers during token generation.

---

# Deferred extension points

## Weight/expert codecs

Cache metadata must retain:

```text
ggml type
block size
row and scale layout
fused/split gate-up representation
bytes per expert bundle
supported CUDA path
```

This leaves room for existing and future weight quants without making cache precision depend on residency.

## Context codecs

Future work may add rotated INT4, TurboQuant/Rotor-derived codecs, mixed K/V precision, or compressed cold recurrent checkpoints. Expert-cache metadata must remain independent of KV representation.

## Speculation

After the non-speculative engine is stable, compare:

- no speculation
- suffix/ngram speculation
- shallow MTP under a strict VRAM cap

MTP remains enabled only when it improves end-to-end latency on the current workload. It is not part of the initial success requirement.

---

# Source seams

Likely integration points:

- `src/llama-load-tensors.cpp`: placement and expert tensor creation
- `src/llama-model-loader.cpp`: GGUF provenance and host-backed ranges
- `src/llama-expert-io.h`: expert file ranges and deferred bytes
- `src/graphs/build_qwen35.cpp`: internal shared Qwen hybrid graph implementation used by Qwen3.6
- `src/llama-build-context.*`: common MoE graph construction
- `ggml/src/ggml-backend.cpp`: active-expert scheduling and copies
- `ggml/src/ggml-cuda/`: indirect quantized MoE and future slot/staging kernels
- `examples/llama-bench/`: native JSON PP/TG measurements
- `benchmarks/qwen36-35b/`: experiment runner and evidence protocol

Policy, storage, and CUDA execution should remain separate:

```text
profile records facts
policy decides residency/admission
gpu pool owns fixed buffers and epochs
CUDA kernels execute mapped experts
```

---

# Milestones

1. **EXP-0001 / M0:** baseline harness and local-agent evidence round trip
2. **M1:** roofline, transfer microbenchmarks, and expert telemetry
3. **M2:** placement/cache simulator and hard go/no-go result
4. **M3:** one-layer fixed-shape CUDA proof
5. **M4:** model-wide static alternatives
6. **M5:** asynchronous GPU miss staging
7. **M6:** dynamic admission/eviction if justified
8. **M7:** prefill/decode pool reuse
9. **M8:** Qwen3.6/SM89 kernels
10. **M9:** `--fit` integration and startup autotuning

Each milestone must be independently reviewable, benchmarkable, and revertible.

---

# Immediate task: EXP-0001

EXP-0001 adds no inference optimization. It creates:

- deterministic benchmark definitions
- hardware and model fingerprinting
- native `llama-bench` JSON collection
- deterministic `llama-cli` output hashes
- raw and normalized evidence bundles
- a local-agent handoff that builds, runs, commits, and pushes results

The next optimization decision will be made from that committed evidence, not from assumptions.