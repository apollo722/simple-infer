# Hugging Face Model Runner Baseline Plan

This document captures the first implementation plan for `simple-infer`.
The immediate goal is to build a Hugging Face accuracy baseline that runs
through the real framework path, while keeping contracts modular enough for
future custom runners.

No implementation should shortcut directly from the public engine API into
`transformers.generate()` for the main runtime path.

## Goal

Build the first end-to-end inference path:

```text
LLMEngine
  -> InputProcessor
  -> EngineCore
  -> Scheduler
  -> LocalExecutor
  -> Worker
  -> HFModelRunner
  -> Sampler
  -> OutputProcessor
```

The HF runner is a correctness and contract baseline. It can be slow at first,
but it should prove request lifecycle, worker/executor boundaries, sampling,
and output formatting.

Phase 1 should intentionally use placeholder scheduler and KV-cache contracts.
Contributors should not attempt to complete real continuous batching, chunked
prefill, async scheduling, or paged KV cache behavior in this phase.

## Phase 1 Scope

Phase 1 is a synchronous, single-process, single-worker HF baseline.

It should include:

- End-to-end `LLMEngine.generate()` for text completion.
- Typed configs and boundary dataclasses.
- Input tokenization through a Hugging Face tokenizer.
- A simple engine core loop.
- A pass-through scheduler that schedules one request step at a time.
- Placeholder/logical KV cache objects that preserve future interfaces.
- Local executor and worker pass-through layers.
- `HFModelRunner` using Hugging Face model forward, not
  `transformers.generate()`.
- Sampler support for at least greedy decoding.
- Output detokenization and final sync outputs.
- Fake-runner tests for framework behavior.
- Optional local tiny-HF smoke test for model accuracy.

It should not include:

- Real chunked prefill.
- Real continuous batching.
- Real KV block allocation pressure or eviction behavior.
- Hugging Face `past_key_values` caching.
- Async engine or async driver.
- Streaming output.
- Multi-worker or distributed execution.

## Non-Goals For The First Pass

- No distributed execution.
- No async engine.
- No paged-attention kernel.
- No true GPU KV cache integration.
- No Hugging Face `past_key_values` integration.
- No real chunked prefill implementation.
- No real continuous batching implementation.
- No prefix caching.
- No preemption.
- No streaming output.
- No OpenAI-compatible HTTP API.
- No direct dependency on protocol payloads below the engine/input layer.

## Proposed Package Layout

```text
src/simple_infer/
  __init__.py

  config/
    __init__.py
    model.py
    engine.py
    scheduler.py
    cache.py
    parallel.py

  protocol/
    __init__.py
    requests.py
    outputs.py

  sampling/
    __init__.py
    params.py
    sampler.py

  inputs/
    __init__.py
    base.py
    processor.py
    hf_tokenizer.py

  engine/
    __init__.py
    llm_engine.py

  core/
    __init__.py
    engine_core.py
    request.py
    outputs.py

  scheduler/
    __init__.py
    base.py
    scheduler.py
    outputs.py

  kv_cache/
    __init__.py
    base.py
    manager.py
    blocks.py

  executor/
    __init__.py
    base.py
    local.py

  worker/
    __init__.py
    worker.py

  model_runner/
    __init__.py
    base.py
    hf_runner.py
    inputs.py
    outputs.py

  outputs/
    __init__.py
    base.py
    processor.py

  metrics/
    __init__.py
    stats.py

  testing/
    __init__.py
    fake_tokenizer.py
    fake_model_runner.py
```

## Contracts And Concrete Classes

Each replaceable layer should expose a small high-level contract and one Phase 1
concrete implementation. Prefer `typing.Protocol` or a narrow ABC over deep
inheritance trees. Future implementations should plug into the same contract;
they should not inherit from the HF baseline unless they are truly a small HF
variant.

Use `HF` names only for components that directly depend on Hugging Face APIs.
Do not name every layer `HF...` just because Phase 1 uses an HF model runner.

Recommended naming:

```text
InputProcessor protocol
  -> HFTokenizerInputProcessor
  -> CustomTokenizerInputProcessor later

Scheduler protocol
  -> SimpleFifoScheduler
  -> ContinuousBatchingScheduler later

KVCacheManager protocol
  -> PlaceholderKVCacheManager
  -> PagedKVCacheManager later

Executor protocol
  -> LocalExecutor
  -> MultiprocessExecutor later

ModelRunner protocol
  -> HFModelRunner
  -> PagedAttentionModelRunner later

OutputProcessor protocol
  -> SyncOutputProcessor
  -> StreamingOutputProcessor later
```

Avoid these names unless the component really owns Hugging Face-specific logic:

```text
HFEngineCore
HFScheduler
HFKVCacheManager
HFExecutor
HFOutputProcessor
```

Phase 1 can keep concrete classes simple. The important part is that public
collaboration happens through the parent contract at every replaceable boundary.

## Public API Target

```python
from simple_infer import LLMEngine
from simple_infer.sampling import SamplingParams

engine = LLMEngine(
    model="distilgpt2",
    dtype="float32",
    device="cpu",
)

outputs = engine.generate(
    prompts=["Write one sentence."],
    sampling_params=SamplingParams(max_tokens=8, temperature=0.0),
)
```

`LLMEngine` should accept ergonomic user arguments, normalize them into typed
config objects, wire the internal components, submit requests, step the core,
and return final `RequestOutput` values.

## Config Contracts

All config should be explicit dataclasses. Components should receive only the
config objects they need.

```python
@dataclass(frozen=True)
class ModelConfig:
    model: str
    tokenizer: str | None = None
    dtype: str = "auto"
    device: str = "auto"
    max_model_len: int | None = None
    trust_remote_code: bool = False
    revision: str | None = None


@dataclass(frozen=True)
class EngineConfig:
    max_active_requests: int = 256
    stream_interval: int = 1
    async_mode: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_scheduled_tokens: int = 8192
    max_num_scheduled_requests: int = 256
    policy: Literal["fifo"] = "fifo"
    decode_first: bool = True
    chunked_prefill_enabled: bool = True


@dataclass(frozen=True)
class CacheConfig:
    block_size: int = 16
    num_blocks: int | None = None
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = False


@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
```

Validation should reject invalid budgets, memory fractions, and parallel sizes
early.

Phase 1 may define future-facing flags such as `async_mode`,
`chunked_prefill_enabled`, `decode_first`, and `enable_prefix_caching`, but
contributors should treat them as configuration surface only. They do not need
to implement the underlying async, chunked-prefill, decode-first arbitration, or
prefix-cache behavior in Phase 1.

## Data Contracts

### InferenceRequest

Public, protocol-neutral request created by the engine/input layer.

```python
request_id: str
prompt: str | None
prompt_token_ids: list[int] | None
sampling_params: SamplingParams
arrival_time: float
priority: int = 0
metadata: dict[str, Any] = field(default_factory=dict)
```

Rules:

- No scheduler state.
- No HTTP or CLI payloads.
- Either `prompt` or `prompt_token_ids` must be present after input processing.

### RuntimeRequest

Mutable request state owned by the runtime after admission.

```python
request_id: str
prompt_token_ids: list[int]
output_token_ids: list[int]
num_computed_tokens: int
status: RequestStatus
kv_blocks: list[int]
arrival_time: float
priority: int
sampling_params: SamplingParams
finish_reason: FinishReason | None
eos_token_id: int | None
```

Rules:

- Scheduler and engine core can mutate this object.
- Workers and model runners should receive scheduled snapshots, not the live
  request object.

### ScheduledRequest

Serializable work description for one request in one model step.

```python
request_id: str
prompt_token_ids: list[int]
output_token_ids: list[int]
num_computed_tokens: int
num_new_tokens: int
is_prefill: bool
block_table: list[int]
sampling_params: SamplingParams
```

### SchedulerOutput

One complete model-step decision.

```python
scheduled_requests: list[ScheduledRequest]
prefill_request_ids: list[str]
decode_request_ids: list[str]
new_token_budget: int
block_tables: dict[str, list[int]]
finished_request_ids: list[str]
```

Rules:

- The model runner executes exactly this work.
- Scheduler queues do not leak to the worker.
- Output should remain serializable enough for future process boundaries.

### ModelRunnerOutput

Backend-neutral result from one model step.

```python
request_ids: list[str]
sampled_token_ids: list[list[int]]
logprobs: list[Any | None]
finish_candidates: dict[str, FinishReason | None]
timings: dict[str, float]
```

Rules:

- No detokenized text.
- No external response formatting.
- Do not return tensors unless a future executor boundary explicitly requires it.

### EngineCoreOutput

Runtime update consumed by the output layer.

```python
request_id: str
new_token_ids: list[int]
output_token_ids: list[int]
finished: bool
finish_reason: FinishReason | None
```

### RequestOutput

User-visible result.

```python
request_id: str
prompt: str | None
text: str
token_ids: list[int]
finished: bool
finish_reason: str | None
```

## HFModelRunner Design

The runner should implement a backend-neutral interface:

```python
class ModelRunner(Protocol):
    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        ...
```

Initial `HFModelRunner` responsibilities:

- Load `AutoModelForCausalLM.from_pretrained`.
- Move model to the configured device.
- Select dtype from `ModelConfig`.
- Call `model.eval()`.
- Use `torch.inference_mode()`.
- Build padded tensors from `ScheduledRequest` values.
- Run a model forward pass.
- Extract last-token logits per scheduled request.
- Call `Sampler`.
- Return sampled token ids in `ModelRunnerOutput`.

Phase 1 simplification:

```text
For every scheduled request step, recompute the full available context:

context = prompt_token_ids + output_token_ids
forward(context)
sample(logits at final position)
```

The first HF runner must not implement chunked-prefill internals or
`past_key_values`. It should ignore KV metadata for computation and treat each
decode step as a full-context forward pass. This is slower, but it creates a
solid accuracy baseline against native Hugging Face forward behavior.

The runner must accept `block_table` metadata but may ignore it for computation.
This preserves the future interface for paged attention.

## Sampler Design

`Sampler` owns token selection. Model forward code should not mutate request
state or scheduler queues.

Initial features:

- Greedy when `temperature == 0.0`.
- Temperature sampling when `temperature > 0.0`.
- Top-k filtering.
- Top-p filtering.
- Optional seeded generator for deterministic tests.

`SamplingParams`:

```python
max_tokens: int
temperature: float = 1.0
top_k: int | None = None
top_p: float | None = None
stop_token_ids: list[int] = field(default_factory=list)
seed: int | None = None
```

Finish conditions are evaluated by engine core or scheduler update code, not by
the runner.

## Logical KV Cache Baseline

Phase 1 should provide a placeholder logical cache contract, not a full
allocator. The goal is to let the scheduler, worker, and runner share the same
future-facing data shape without making contributors solve KV-cache management
yet.

```text
KVCacheManager
  returns empty or trivial block tables
  exposes allocate/free method names
  performs no real capacity planning
  stores no device tensors
```

The HF runner ignores block tables for tensor computation. Later phases can
replace the placeholder with a real `BlockPool`, `KVCacheBlock`, block-table
growth, capacity checks, and freeing behavior.

## Scheduler Baseline

Phase 1 should provide a minimal synchronous scheduler. Its purpose is to move
requests through the framework path, not to implement the production scheduling
strategy.

Phase 1 policy:

- FIFO waiting queue.
- Schedule one request step at a time, or a simple full batch if that is easier.
- Treat prefill as an initial full-context step.
- Treat each decode as one full-context step.
- Preserve fields for `prefill_request_ids`, `decode_request_ids`,
  `new_token_budget`, and `block_tables`.
- Do not enforce real token-budget packing.
- Do not implement chunked prefill.
- Do not implement decode-first arbitration across mixed prefill/decode work.
- Do not implement KV budget gating.
- No preemption.
- No priority ordering beyond storing the priority field.

The architecture document's two-request continuous batching example should be
used as a future scheduler test target, not a Phase 1 requirement.

## Component Responsibilities

### LLMEngine

- Accept prompts or token ids.
- Normalize config.
- Own `InputProcessor`, `EngineCore`, and `OutputProcessor`.
- Assign stable request ids.
- Submit requests.
- Step until all requests finish for sync `generate`.
- Return final `RequestOutput` list.
- Do not tokenize manually outside `InputProcessor`.
- Do not call workers or model runners directly.

### InputProcessor

- Own tokenizer loading.
- Convert prompt strings to token ids.
- Support already-tokenized inputs.
- Validate prompt length.
- Later: chat template rendering.
- Return `InferenceRequest` objects or tokenization results only.
- Do not know about scheduler queues, workers, or model forward execution.

### EngineCore

- Convert `InferenceRequest` into `RuntimeRequest`.
- Own active runtime requests.
- Call scheduler.
- Call executor.
- Apply model runner outputs.
- Mark finish reasons.
- Free KV blocks on finish.
- Return `EngineCoreOutput`.
- Depend on scheduler and executor contracts, not concrete HF classes.
- Do not tokenize, detokenize, or format public responses.

### Scheduler

- Own waiting/running queues.
- Own scheduling decisions.
- Ask KV cache manager for blocks.
- Produce `SchedulerOutput`.
- Update request state from model outputs.
- Depend on the KV-cache contract only.
- Do not import Hugging Face, torch, tokenizer, worker, or output processor code.

### LocalExecutor

- Hide execution topology from engine core.
- Forward `SchedulerOutput` to one local worker.
- Implement the executor contract.
- Do not inspect request internals beyond the scheduler output.

### Worker

- Own rank-local device state.
- Own one model runner.
- Execute scheduled batches.
- Depend on the model-runner contract.
- Do not schedule requests or detokenize outputs.

### HFModelRunner

- Own Hugging Face model forward execution.
- Build tensors from scheduled metadata.
- Call sampler.
- Return backend-neutral outputs.
- Depend on `SchedulerOutput`, `ModelRunnerOutput`, model config, and sampler.
- Do not mutate runtime requests or scheduler queues.
- Do not detokenize text.

### OutputProcessor

- Own detokenization.
- Own user-visible output state.
- Produce final sync outputs first.
- Later: streaming deltas.
- Consume `EngineCoreOutput`.
- Do not schedule, run model forward, or mutate runtime requests.

## Contributor Tasks

Each task should stay inside its owned package unless it is explicitly asked to
add exports in `__init__.py` files or tests. If a task needs another layer, use
that layer's public contract and fake implementations from `simple_infer.testing`.
Do not reach across packages to mutate another component's private state.

Recommended dependency order:

```text
config
protocol + sampling
runtime request + scheduler outputs
placeholder kv_cache
simple scheduler
model runner interface + fake runner
worker + executor
engine core
input processor + output processor
model loading (HF)
input tensor builder (HF)
forward + logit extraction (HF)
HF runner integration
LLMEngine component wiring
LLMEngine generate loop
metrics
```

Parallel contributors should agree on dataclass field names first, then work
against those contracts. If a contributor discovers a contract gap, update the
contract task before implementing cross-layer workarounds.

### 1. Config

- Create `simple_infer.config`.
- Add config dataclasses.
- Add validation.
- Export configs from package `__init__` files.
- Scope boundary:
  - Owns only config dataclasses and validation.
  - Does not construct engine, tokenizer, scheduler, model, or worker objects.
- Tests:
  - Defaults are usable.
  - Invalid token budgets fail.
  - Invalid memory fraction fails.
  - Invalid parallel sizes fail.

### 2. Protocol Contracts

- Create `InferenceRequest`.
- Create `RequestOutput`.
- Create `FinishReason`.
- Scope boundary:
  - Owns public request/output shapes only.
  - Does not include runtime mutable fields, scheduler fields, tensors, or HF objects.
- Tests:
  - Requests are protocol-neutral.
  - Output object is stable and serializable.

### 3. Sampling

- Create `SamplingParams`.
- Implement `Sampler`.
- Scope boundary:
  - Owns token selection from logits only.
  - Does not know about requests, queues, tokenization, detokenization, or model loading.
- Tests:
  - Greedy selects argmax.
  - Temperature sampling handles deterministic seeded case.
  - Top-k masks lower-ranked logits.
  - Top-p masks tail probability.

### 4. Input Processor

- Define `InputProcessor` contract.
- Implement `HFTokenizerInputProcessor`.
- Implement string prompt tokenization.
- Implement token-id input passthrough.
- Validate max model length.
- Scope boundary:
  - Owns prompt-to-token conversion only.
  - Does not schedule, run model forward, sample logits, or detokenize generated output.
- Tests:
  - Fake tokenizer path.
  - String prompt creates token ids.
  - Token ids pass through.
  - Too-long prompt is rejected.

### 5. Runtime Request

- Define `RuntimeRequest`.
- Define `RequestStatus`.
- Define `EngineCoreOutput`.
- Add helper methods for token append and finish marking if useful.
- Scope boundary:
  - Owns runtime request state types only.
  - Does not implement scheduler policy, executor calls, or model execution.
- Tests:
  - Prefill progress.
  - Decode append.
  - Length finish.
  - EOS or stop-token finish.

### 6. KV Cache

- Define `KVCacheManager` contract.
- Implement `PlaceholderKVCacheManager`.
- Return empty or trivial block tables.
- Expose future-facing `allocate_slots` and `free` methods.
- Do not implement real block pressure, eviction, or device tensors.
- Scope boundary:
  - Owns cache metadata contract only.
  - Does not allocate torch tensors, call the model runner, or decide scheduler policy.
- Tests:
  - Allocate returns a stable block-table shape.
  - Free is callable and idempotent for Phase 1.
  - No device tensors are exposed.

### 7. Scheduler

- Define `Scheduler` contract.
- Implement `SimpleFifoScheduler`.
- Schedule enough work for synchronous HF generation to progress.
- Preserve `SchedulerOutput` fields needed by future schedulers.
- Mark initial prompt work as prefill.
- Mark subsequent one-token generation steps as decode.
- Call placeholder KV cache methods only to preserve the boundary.
- Do not implement real chunked prefill.
- Do not implement continuous batching.
- Do not implement decode-first arbitration.
- Do not implement KV budget gating.
- Scope boundary:
  - Owns queue and scheduling decisions only.
  - Does not import torch, transformers, tokenizer, worker, executor, or output processor code.
- Tests:
  - Empty scheduler output.
  - Single request initial prefill-shaped output.
  - Single request decode-shaped output.
  - Multiple requests eventually complete in FIFO order.
  - Scheduler output remains serializable.

### 8. Model Runner Interface

- Define `ModelRunner` protocol.
- Define `ModelRunnerOutput`.
- Add fake deterministic runner for tests.
- Scope boundary:
  - Owns backend-neutral model execution contract only.
  - Does not implement HF loading in this task.
- Tests:
  - Interface returns output for scheduled requests.
  - Fake runner supports engine-core tests.

### 9a. Model Loading (HF)

- Load model via `AutoModelForCausalLM.from_pretrained`.
- Resolve dtype from `ModelConfig` (map string to `torch.dtype`).
- Resolve device from `ModelConfig` (map `"auto"` to best available device).
- Move model to resolved device.
- Call `model.eval()`.
- Return the loaded model and resolved device/dtype metadata.
- Scope boundary:
  - Owns model construction and device placement only.
  - Does not build input tensors, run forward, sample, or touch scheduler objects.
- Tests:
  - Model loads with explicit device string.
  - `"auto"` device resolves to a real device.
  - `"auto"` dtype maps to a concrete torch dtype.
  - Model is in eval mode after loading.

### 9b. Input Tensor Builder (HF)

- Accept `SchedulerOutput` (or list of `ScheduledRequest`).
- Concatenate `prompt_token_ids + output_token_ids` per request for full context.
- Pad to max sequence length in the batch.
- Build `attention_mask` (1 for real tokens, 0 for padding).
- Return `(input_ids, attention_mask)` tensors on the configured device.
- Scope boundary:
  - Owns tensor construction from scheduled metadata only.
  - Does not load models, call forward, or sample logits.
- Tests:
  - Single request produces correctly shaped tensors.
  - Multiple requests of different lengths are padded to same max length.
  - Attention mask is 1 for tokens and 0 for padding positions.
  - Tensors are on the expected device.

### 9c. Forward And Logit Extraction (HF)

- Accept loaded model and `(input_ids, attention_mask)`.
- Run `model(input_ids, attention_mask)` under `torch.inference_mode()`.
- Extract logits at the last non-padding position for each request in the batch.
- Return per-request logit tensors (shape `[vocab_size]` each).
- Scope boundary:
  - Owns forward pass and logit slicing only.
  - Does not build input tensors, call sampler, or mutate runtime state.
- Tests:
  - Forward produces logits with correct vocab dimension.
  - Last-position logits correspond to the final real token per request.
  - `inference_mode` context is used (verify no grad).

### 9d. HFModelRunner Integration

- Wire model loading (9a) + input builder (9b) + forward/logit extraction (9c) + `Sampler`.
- Implement `HFModelRunner` behind the `ModelRunner` protocol.
- `execute(scheduler_output)`:
  1. Build input tensors (9b).
  2. Run forward and extract logits (9c).
  3. Call `Sampler.sample(logits, sampling_params)` per request.
  4. Return `ModelRunnerOutput`.
- Recompute full context every step (no `past_key_values`).
- Accept and ignore `block_table` metadata (preserve future interface).
- Do NOT use `transformers.generate()`.
- Do NOT implement chunked prefill internals.
- Scope boundary:
  - Owns HF model runner composition only.
  - Does not tokenize, schedule, mutate runtime requests, or detokenize.
- Tests:
  - Mocked model forward path returns expected output shape.
  - Tiny-model smoke test when model is locally available.
  - Runner rejects `transformers.generate()` usage in the execute path.
  - Block table metadata is accepted without error.

### 10. Worker

- Implement single local worker.
- Own one model runner.
- Forward scheduler output to runner.
- Scope boundary:
  - Owns device/rank-local execution wrapper only.
  - Does not decide scheduling policy or inspect tokenizer/output formatting.
- Tests:
  - Pass-through with fake runner.

### 11. Executor

- Define executor protocol.
- Implement `LocalExecutor`.
- Scope boundary:
  - Owns dispatch topology only.
  - Does not run model forward directly except by delegating to worker.
- Tests:
  - Engine core depends on executor protocol only.
  - Local executor delegates to worker.

### 12. EngineCore

- Implement admission.
- Own active requests.
- Step scheduler and executor.
- Apply `ModelRunnerOutput`.
- Call placeholder cache free path on finish.
- Return core outputs.
- Scope boundary:
  - Owns orchestration and runtime lifecycle only.
  - Does not import Hugging Face, torch, tokenizer implementation, or output formatting code.
- Tests:
  - Add request.
  - Step with fake model output.
  - Max-token finish.
  - EOS finish.
  - Finished requests are removed or no longer scheduled.

### 13. OutputProcessor

- Detokenize token ids.
- Track per-request text.
- Produce final sync output.
- Scope boundary:
  - Owns generated-token-to-user-output conversion only.
  - Does not schedule requests, call executor, or load models.
- Tests:
  - Incremental decode appends text.
  - Finish reason is preserved.
  - Multiple requests return independent output.

### 14a. LLMEngine Component Wiring

- Accept user-facing config arguments (model path, dtype, device, etc.).
- Normalize into typed config dataclasses (`ModelConfig`, `EngineConfig`, etc.).
- Construct and wire the full component graph:
  - `InputProcessor`
  - `EngineCore` (with scheduler, executor, KV cache manager)
  - `OutputProcessor`
- Do NOT implement the generate loop in this task.
- Scope boundary:
  - Owns component construction and dependency injection only.
  - Does not step the engine, request tokens, or call model forward.
- Tests:
  - All components are constructed without error.
  - Config normalization produces valid typed configs.
  - Component graph is inspectable (correct types at each slot).

### 14b. LLMEngine Generate Loop

- Implement sync `generate(prompts, sampling_params)`.
- Call `InputProcessor.process()` to tokenize.
- Submit requests to `EngineCore`.
- Step `EngineCore` in a loop until all requests finish.
- Collect `EngineCoreOutput` and pass to `OutputProcessor`.
- Return `list[RequestOutput]`.
- Export `LLMEngine` from `simple_infer`.
- Keep the path synchronous for Phase 1.
- Do not add async driver or streaming API.
- Scope boundary:
  - Owns the public sync generate loop and composition orchestration.
  - Does not implement component internals inline.
- Tests:
  - End-to-end fake runner path produces output.
  - Multiple prompts return independent results.
  - Sampling params are respected.
  - Optional local tiny-HF smoke test.

### 15. Metrics

- Add lightweight stats object.
- Track admitted requests.
- Track finished requests.
- Track queue length.
- Track step latency.
- Track prefill and decode token counts.
- Track KV usage.
- Scope boundary:
  - Owns passive stats recording only.
  - Does not affect scheduling decisions or model execution.
- Tests:
  - Metrics updates do not affect control flow.

## Milestones

### Milestone 1: Phase 1 Contracts Compile

- Configs exist.
- Boundary dataclasses exist.
- Protocols exist.
- Placeholder scheduler and KV cache contracts exist.
- Fake model runner exists.
- No real HF dependency is exercised in tests.

### Milestone 2: Phase 1 Fake End-To-End Runtime

- Minimal FIFO scheduler progresses requests.
- Placeholder KV cache boundary is exercised.
- `LLMEngine.generate()` works with fake runner.
- Output processor returns final text.
- Engine core finish behavior is covered.

### Milestone 3: Phase 1 HF Accuracy Baseline

- Model loads and resolves dtype/device from config (9a).
- Input tensor builder handles multi-request batching with padding (9b).
- Forward pass extracts last-position logits per request (9c).
- `HFModelRunner.execute()` wires model, tensors, forward, and sampler (9d).
- Full framework path exercised: Engine → Core → Scheduler → Executor → Worker → Runner.
- No `transformers.generate()` in the main runtime path.
- No `past_key_values` requirement.
- No chunked-prefill requirement.

### Milestone 4: Phase 1 Hardening

- Improve errors.
- Add metrics.
- Add local smoke-test instructions.
- Document known placeholder contracts.

### Future Milestone: Real Scheduler And Cache

- Implement logical `BlockPool`.
- Implement KV block allocation and freeing.
- Implement request and token budget packing.
- Implement decode-first scheduling.
- Implement chunked prefill.
- Convert the architecture doc's two-request continuous batching example into
  required tests.
- Prepare for cached HF runner or custom paged-attention runner.

## Testing Strategy

Follow the canonical test-suite structure in
[`docs/testing.md`](../testing.md). Contributors should place tests under the
matching package path, for example `tests/unit/config/` for config-only tests
and `tests/integration/` for cross-layer workflows.

Phase 1 tests should not assert real chunked-prefill, KV-budget, streaming, or
async behavior. They should assert that placeholder contracts are present and
that the synchronous HF baseline produces tokens through the full framework
path.

Real HF tests must be opt-in, marked with `hf`, and use local model paths from
`test-models/` when available. They must not download models during the test
run.

## Design Guardrails

- Use typed dataclasses at layer boundaries.
- Keep protocol/rendering code above engine core.
- Keep tensors below worker/model-runner boundary.
- Keep scheduler outputs serializable.
- Keep detokenized text out of model runner outputs.
- Keep scheduler and model runner unaware of HTTP or CLI protocols.
- Treat KV cache as allocator-owned state.
- Avoid global mutable config.
- Avoid direct calls to `transformers.generate()` in the real runtime path.
- Add tests for state machines before adding performance features.
