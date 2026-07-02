# Simple Infer Testing Guide

This document defines the test-suite structure for `simple-infer`. Tests should
scale with the architecture: small layer-owned unit tests by default, explicit
integration tests for cross-layer behavior, and opt-in tests for local Hugging
Face models or slow paths.

## Goals

- Keep `uv run pytest` fast and reliable.
- Mirror the source architecture so contributors know where tests belong.
- Prevent local reference repositories and model checkpoints from being
  collected by pytest.
- Make HF/model-dependent tests opt-in.
- Keep unit tests isolated with fake dependencies.
- Keep cross-layer tests rare, intentional, and clearly marked.

## Required Pytest Configuration

The project should configure pytest in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
norecursedirs = [
    ".venv",
    "reference",
    "learning-docs",
    "test-models",
]
markers = [
    "unit: fast isolated tests for one package or component",
    "integration: tests that exercise multiple framework layers",
    "hf: tests requiring local Hugging Face model/tokenizer files",
    "slow: tests that are too slow for the default local loop",
]
```

Default command:

```bash
uv run pytest
```

The default command must not collect `reference/`, `learning-docs/`, or
`test-models/`.

## Directory Layout

Tests should mirror `src/simple_infer`:

```text
tests/
  unit/
    config/
      test_model_config.py
      test_engine_config.py
      test_scheduler_config.py
      test_cache_config.py
      test_parallel_config.py

    protocol/
    sampling/
    inputs/
    core/
    scheduler/
    kv_cache/
    executor/
    worker/
    model_runner/
    outputs/
    metrics/

  integration/
    test_fake_engine_e2e.py
    test_scheduler_core_contract.py
    test_worker_executor_contract.py

  smoke/
    test_imports.py

  conftest.py
```

Avoid large mixed files like `tests/test_config.py` once a package has more than
one component. Split tests by component so ownership remains clear.

## Test Categories

### Unit Tests

Unit tests are the default. They should:

- Cover one package or component.
- Use fake dependencies instead of real downstream layers.
- Avoid model downloads, GPU assumptions, network calls, and filesystem state
  outside pytest temporary directories.
- Assert component contracts and edge cases.
- Run quickly enough for every local edit loop.

Examples:

```text
tests/unit/config/test_cache_config.py
tests/unit/scheduler/test_simple_fifo_scheduler.py
tests/unit/model_runner/test_model_runner_contract.py
```

### Integration Tests

Integration tests verify that multiple layers collaborate through public
contracts. They should:

- Use fake model runners by default.
- Avoid local HF model requirements unless marked `hf`.
- Exercise one narrow workflow per test file.
- Avoid duplicating unit-test edge cases.

Examples:

```text
tests/integration/test_fake_engine_e2e.py
tests/integration/test_core_scheduler_executor_flow.py
```

### HF Tests

HF tests require local model/tokenizer files and must be opt-in.

Use the `hf` marker:

```python
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.hf]
```

HF tests should:

- Use local paths under `test-models/`.
- Skip cleanly if the required model path is absent.
- Never download models during the test run.
- Avoid asserting exact long text unless the model, seed, dtype, and sampling
  mode are controlled.

Run explicitly:

```bash
uv run pytest -m hf
```

### Slow Tests

Use the `slow` marker for tests that are valuable but unsuitable for the default
loop:

```bash
uv run pytest -m slow
```

Slow tests should usually also be integration or HF tests.

## Marker Usage

Recommended commands:

```bash
# Default local loop.
uv run pytest

# Unit-only loop.
uv run pytest tests/unit -m "not slow and not hf"

# Integration tests that do not require local HF models.
uv run pytest tests/integration -m "not hf and not slow"

# Local HF smoke tests.
uv run pytest -m hf

# Slow tests.
uv run pytest -m slow
```

The default suite should not require `-m` filtering to avoid HF or slow tests.
Place HF and slow tests in paths that are not accidentally run by contributors
unless they opt in, or mark and skip them cleanly.

## Fixture Design

Use `tests/conftest.py` for shared fixtures only when they are broadly useful.
Prefer local fixtures inside the test module when the fixture serves one
component.

Recommended shared fixtures:

- fake tokenizer
- fake model runner
- minimal sampling params
- minimal inference request
- temporary local model path helper

Avoid shared fixtures that:

- construct the whole engine for unit tests
- hide important assertions
- import optional HF dependencies for non-HF tests
- mutate global state across tests

Project-owned reusable fakes should live in `src/simple_infer/testing` when
runtime packages also need them, otherwise keep them under `tests/`.

## Naming Rules

- Test files: `test_<component_or_behavior>.py`.
- Test classes: optional, grouped by component.
- Test names: describe the behavior, not the implementation.

Good:

```python
def test_num_blocks_must_be_positive_when_provided() -> None:
    ...
```

Avoid:

```python
def test_config_7() -> None:
    ...
```

## Assertions

Prefer contract-level assertions:

- dataclass defaults
- validation failures
- serializable boundary objects
- public method outputs
- explicit finish reasons
- scheduler output shape

Avoid assertions on private implementation details unless the test is for that
specific internal component.

## Task 1 Config Test Requirements

Config tests should be split as:

```text
tests/unit/config/test_model_config.py
tests/unit/config/test_engine_config.py
tests/unit/config/test_scheduler_config.py
tests/unit/config/test_cache_config.py
tests/unit/config/test_parallel_config.py
```

Required coverage:

- Defaults are usable for every config dataclass.
- Invalid token/request/stream budgets fail.
- Invalid memory fractions fail.
- Invalid parallel sizes fail.
- Optional positive integer capacities fail when set to `0` or negative.
- Future-facing flags such as `async_mode`, `decode_first`,
  `chunked_prefill_enabled`, and `enable_prefix_caching` are accepted as config
  surface only; tests should not expect their runtime behavior yet.

Config unit tests must not:

- import torch or transformers
- construct engine, scheduler, worker, or model runner objects
- touch local model paths
- depend on network access

## Phase 1 Test Expectations

Phase 1 tests should prove the HF baseline framework path without requiring
advanced runtime behavior.

Required by default:

- config unit tests
- protocol/dataclass unit tests
- sampler unit tests
- placeholder KV-cache contract tests
- simple scheduler contract tests
- fake model runner contract tests
- worker/executor pass-through tests
- engine-core tests with fake executor/model output
- output processor tests
- fake-runner end-to-end integration test

Opt-in:

- local tiny-HF smoke test
- slow or GPU-sensitive tests

Not required in Phase 1:

- real chunked prefill
- real continuous batching
- real KV block pressure
- `past_key_values`
- async engine
- streaming output

## Review Checklist

When reviewing tests, check:

- Does the file live under the correct test category and package path?
- Does it test one layer unless explicitly marked integration?
- Does it use fakes instead of downstream real components?
- Does `uv run pytest` avoid local reference and checkpoint directories?
- Are HF and slow tests opt-in?
- Are validation edge cases covered?
- Are assertions made against public contracts?

