# Cathedral Plan

## Vision

Cathedral is a Pythonic probabilistic programming library inspired by Church and WebPPL.

The goal is to support expressive probabilistic world models with plain Python syntax that modern LLMs can generate and humans can read.

## Design Principles

- Pure Python DSL built from decorated model functions
- Trace-based execution with explicit random choices
- Stochastic control flow, recursion, and memoization as first-class features
- Simple public API with diagnostics and analysis built in
- NumPy / SciPy runtime today, with room for stronger numerics later

## Current State

Cathedral currently has:

- Core primitives: `sample`, `flip`, `condition`, `observe`, `factor`
- Stochastic memoization via `mem()` and `DPmem()`
- Four inference engines: rejection, importance, MH, enumeration
- Posterior analysis, checks, diagnostics, and visualization
- Example models spanning generative modeling, conditioning, regression, mixtures, social cognition, and grammars

## Current Architecture

```text
cathedral/
  __init__.py
  distributions.py
  primitives.py
  trace.py
  model.py
  checks.py
  viz.py
  plots.py
  inference/
    rejection.py
    importance.py
    mh.py
    enumeration.py
```

Core execution model:

1. A user writes a `@model` function using Cathedral primitives.
2. `infer()` selects an inference engine.
3. The engine repeatedly runs `run_with_trace()`.
4. Execution records `Choice` objects into a `Trace`.
5. Results are returned as a `Posterior` with diagnostics and analysis helpers.

## Milestone 0.3.0

Focus the next release on deterministic seeded inference, correctness for real importance-sampling workflows, and small modeling conveniences that make Cathedral easier to use on real projects soon.

### Release work

#### In scope

- Deterministic RNG plumbing built on per-run RNG objects rather than global `numpy.random`
- Seeded rejection sampling
- Seeded importance sampling
- Seeded MH and predictive utilities where feasible
- Correct weighted posterior analysis for importance sampling without resampling
- A compact posterior summary helper for quick model iteration
- A small set of practical distributions used in common applied models
- Benchmarking on existing Cathedral examples with a short writeup
- Explicit evaluation of whether parallel inference is worth pursuing further

#### Implementation phases

1. Introduce an internal RNG abstraction backed by `numpy.random.Generator`.
2. Thread RNG usage through `Distribution.sample()`, inference resampling, MH internals, and `DPmem()`.
3. Add `seed` support to rejection, importance, MH, and predictive utilities while keeping existing unseeded calls working.
4. Make `infer(..., method="importance", resample=False)` return weighted posteriors.
5. Add a compact `Posterior.summary()` API for common numeric summaries.
6. Add practical distributions needed for applied examples.
7. Write reproducibility, weighted-analysis, and distribution regression tests.
8. Run benchmarks and summarize runtime, acceptance rate, and ESS behavior.
9. Decide whether parallel inference should remain out of scope for `v0.3`.

#### Reproducibility contract

- Same `seed` should produce identical results for the same inference call.
- Seeded runs should not depend on prior global NumPy RNG state.
- Unseeded runs should preserve existing user-facing behavior.

#### Benchmark targets

- Rejection: `sprinkler`, `bag_of_marbles`
- Importance: `coin_bias_estimation`, `bayesian_regression`
- Richer structure sanity check: `coordination` or `expressions_equal_10`

#### Benchmark outputs

- Wall-clock time
- Attempts per second for rejection
- ESS and ESS per second for importance
- Seeded reproducibility checks

#### Practical distribution targets

- `Exponential`
- `LogNormal`
- `Binomial`
- `StudentT`

#### Stretch goal

- Optional multi-chain MH, but only if the API remains small and the seeded-chain story is clean
- Revisit parallel rejection and importance only if heavier benchmarks show a compelling upside

#### Execution plan

##### Workstream 1: RNG foundation

Goal: move Cathedral's stochastic execution off global `numpy.random` and onto explicit per-run generators.

Architecture decision:

- Use an internal active-RNG context layered on top of Cathedral's existing tracing/contextvars machinery.
- Do not thread `rng` as an explicit argument through every public primitive or `Distribution.sample()` call in `v0.3`.
- Keep the public seeding surface small: `infer(..., seed=...)` and direct engine-level `seed=` kwargs where useful.

Why this shape:

- `sample()` is the only place where traced model execution calls `Distribution.sample()`, so a context-based RNG fits the existing execution model cleanly.
- Cathedral already relies on `ContextVar` for trace state, so RNG can follow the same execution boundary.
- Explicit `rng` threading would touch nearly every stochastic call signature and make model internals noisier without buying much for this release.
- The same active RNG helper can cover non-distribution randomness such as `DPmem()`, importance resampling, MH transitions, and posterior predictive resampling.

Likely file touchpoints:

- `cathedral/distributions.py`
- `cathedral/primitives.py`
- `cathedral/trace.py`
- `cathedral/model.py`
- `cathedral/inference/importance.py`
- `cathedral/inference/mh.py`
- `cathedral/checks.py`
- new internal RNG helper module

Tasks:

1. Add an internal RNG module that owns:
   - seed normalization
   - `Generator` creation
   - active-RNG context helpers
2. Extend tracing state so `run_with_trace()` can install a per-run RNG for the duration of model execution.
3. Update `Distribution.sample()` implementations to draw from the active generator rather than global `np.random`.
4. Update stochastic helpers outside distributions, especially `DPmem()`, importance resampling, MH proposal selection, MH accept/reject draws, and posterior predictive resampling.
5. Thread `seed` through `infer()` and the engine entry points in a way that leaves existing unseeded calls working.
6. Keep direct standalone model execution working by falling back to an internal default generator when no seeded run context is active.

Acceptance criteria:

- A fixed `seed` yields repeatable serial runs for rejection and importance sampling.
- Serial seeded runs do not depend on prior library calls to global NumPy RNG state.
- Existing unseeded API behavior remains valid.

##### Workstream 2: Seeded serial inference

Goal: expose reproducible serial execution on the main approximate inference paths.

Likely file touchpoints:

- `cathedral/inference/rejection.py`
- `cathedral/inference/importance.py`
- `cathedral/inference/mh.py`
- `cathedral/model.py`
- tests in `tests/test_inference.py`

Tasks:

1. Add `seed` support to rejection, importance, and MH.
2. Ensure serial rejection diagnostics remain correct under seeded execution.
3. Ensure serial importance resampling remains deterministic under a fixed seed.
4. Ensure seeded runs can be reproduced independent of global NumPy RNG state.

Acceptance criteria:

- Same `seed` gives identical results.
- `resample=False` remains a transparent weighted-trace mode.
- Diagnostics match the serial formulas.

##### Workstream 3: Weighted importance posteriors

Goal: make importance sampling correct and useful when users keep weighted traces instead of resampling.

Likely file touchpoints:

- `cathedral/model.py`
- `cathedral/inference/importance.py`
- tests in `tests/test_inference.py`

Tasks:

1. Normalize importance `log_weights` into posterior weights when `resample=False`.
2. Pass normalized weights into `Posterior`.
3. Preserve raw `log_weights`, ESS, and log marginal likelihood diagnostics.
4. Add tests proving posterior summaries use weights.
5. Raise a clear error if all retained traces have impossible weights.

Acceptance criteria:

- `infer(..., method="importance", resample=False)` returns a weighted `Posterior`.
- `Posterior.mean()`, `std()`, `probability()`, `histogram()`, and intervals respect importance weights.
- `resample=True` behavior remains unchanged.

##### Workstream 4: Posterior summary helper

Goal: make iterative model development easier by exposing common posterior statistics in one call.

Likely file touchpoints:

- `cathedral/model.py`
- tests in `tests/test_inference.py`
- `README.md`

Tasks:

1. Add `Posterior.summary(key=None, level=0.95)`.
2. Include `num_samples`, `mean`, `std`, credible interval, ESS, and fixed-structure status.
3. Keep failure behavior simple for nonnumeric results.
4. Document the helper in posterior analysis docs.

Acceptance criteria:

- Summary values match existing `Posterior` helpers.
- Weighted posteriors are summarized with weights.

##### Workstream 5: Practical distributions

Goal: cover common applied-model needs without introducing a large distribution framework.

Likely file touchpoints:

- `cathedral/distributions.py`
- `cathedral/__init__.py`
- tests in `tests/test_distributions.py`
- `README.md`

Tasks:

1. Add `Exponential(rate)`.
2. Add `LogNormal(mu, sigma)`.
3. Add `Binomial(n, p)` with finite support for enumeration.
4. Add `StudentT(df, loc=0, scale=1)`.
5. Add sampling, `log_prob`, validation, reprs, exports, and tests.

Acceptance criteria:

- New distributions sample from the active Cathedral RNG.
- New distributions have correct support and log-probability behavior.
- Public imports work from `cathedral`.

##### Workstream 6: API and docs pass

Goal: keep the public interface small and explicit.

Tasks:

1. Add `seed` to the documented inference API.
2. Update docstrings for `infer()`, `rejection_sample()`, and `importance_sample()`.
3. Document weighted importance posteriors and `Posterior.summary()`.
4. Add one example showing a seeded serial run.
5. Document the reproducibility contract precisely:
   - same `seed` should match
   - seeded runs should not depend on global NumPy RNG state

Acceptance criteria:

- A new user can discover seeded execution, weighted importance analysis, summaries, and practical distributions from public docs alone.
- The documented guarantees match the tested behavior.

##### Workstream 7: Benchmarks

Goal: produce evidence that `v0.3` delivers real systems cleanup and clarify whether further performance work is worth the complexity.

Likely benchmark targets:

- Rejection: `sprinkler`, `bag_of_marbles`
- Importance: `coin_bias_estimation`, `bayesian_regression`
- Structural sanity check: `coordination` or `expressions_equal_10`

Tasks:

1. Create a lightweight benchmark harness or scripts with repeat averaging.
2. Record wall time, attempts/sec, ESS, and ESS/sec as applicable.
3. Run seeded reproducibility checks alongside performance runs.
4. Summarize whether candidate parallel inference work looks promising enough to revisit later.

Acceptance criteria:

- Benchmarks run from the repo with a documented command.
- The benchmark writeup names the tested models, hardware assumptions, and observed runtime behavior.

##### Workstream 8: Test plan

Goal: make regressions in determinism obvious.

Target test additions:

- Seed reproducibility for serial rejection and importance
- Seed reproducibility for MH
- Weighted importance posterior summaries
- `Posterior.summary()`
- New practical distributions
- Serial rejection diagnostics and `max_attempts` behavior
- Serial importance diagnostics, ESS, and `resample=False`
- Smoke coverage for `DPmem()` under seeded execution

Definition of done for `v0.3`:

1. RNG plumbing is complete for the inference paths touched by rejection, importance, and MH internals.
2. Rejection, importance, and MH expose `seed` where appropriate.
3. Weighted importance posteriors are correct for `resample=False`.
4. Common posterior summaries and practical distributions are available.
5. Reproducibility guarantees are tested and documented.
6. Benchmarks are checked in with a short results summary.
7. Parallel inference is either explicitly deferred or moved to a separate follow-up milestone.

## Milestone Backlog

These are candidate milestones, not a fixed sequence. Priority should be driven by the research track and by example ports.

### Milestone: Better MH Proposals

- Add optional proposal kernels to distributions
- Improve continuous-variable mixing for MH
- Expose user-registered proposal hooks where useful

### Milestone: HMC / NUTS

- Add gradient-based inference for static-structure continuous models
- Decide backend only when implementation begins
- Treat this as a major capability milestone, not an automatic next step

### Milestone: Exchangeable Random Primitives

- Add XRP-style sufficient-statistics machinery
- Improve nonparametric and hierarchical model performance
- Extend memoization toward richer Bayesian nonparametrics

### Milestone: Advanced MCMC

- Tempered transitions
- Parallel tempering
- Multiple-try Metropolis
- Annealed importance sampling

### Milestone: Constraint-Guided Initialization

- Backward constraint propagation
- Smarter MH initialization for tightly conditioned models

### Milestone: Conservative Trace Updates

- Partial re-execution
- Dependency-aware caching
- Reduced work for large traces with local changes

## Modeling Utilities Backlog

- Noisy logic primitives
- Standalone importance-sampling utilities
- More distributions
- Mixture distributions
- User-facing proposal APIs

## Near-Term Order

1. Land deterministic RNG plumbing and define the seeded reproducibility contract.
2. Fix weighted importance posterior analysis for `resample=False`.
3. Add posterior summaries and practical distributions for real-use readiness.
4. Benchmark the seeded engines on existing examples and record the results.
5. Run the applied-model survey and select the first external ports.
6. Port models and write down concrete gaps.
7. Choose the next milestone based on those gaps.

## Test Coverage

218 tests across 9 test files:

- `test_distributions.py`
- `test_trace.py`
- `test_primitives.py`
- `test_inference.py`
- `test_diagnostics.py`
- `test_checks.py`
- `test_viz.py`
- `test_plots.py`
- `test_arviz.py`

Run with:

```bash
uv run pytest tests/ -v
```

## Development Setup

```bash
git clone https://github.com/andrewgiessel/cathedral
cd cathedral
uv pip install -e ".[dev]"
uv pip install -e ".[viz]"
uv run pytest tests/ -v
uv run ruff format .
uv run ruff check .
```

## References

- [Church: A Language for Generative Models](https://arxiv.org/pdf/1206.3255v2.pdf)
- [Lightweight Implementations of Probabilistic Programming Languages](http://web.stanford.edu/~ngoodman/papers/lightweight-mcmc-aistats2011.pdf)
- [From Word Models to World Models](https://arxiv.org/abs/2306.12672)
- [WebPPL](http://webppl.org/)
- [Gen.jl](https://www.gen.dev/)
- [Probabilistic Models of Cognition (Church)](https://v1.probmods.org/)
- [Probabilistic Models of Cognition (WebPPL)](https://probmods.org/)
- [Pyro](https://pyro.ai/)
- [Large Language Bayes](https://arxiv.org/abs/2308.13111)
