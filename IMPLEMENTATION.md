# Cathedral Implementation Guide

This guide is for contributors who already understand how to use Cathedral and want a stronger picture of how the implementation fits together. The short version: Cathedral is a trace-based probabilistic programming library. User models are ordinary Python functions; Cathedral installs a tracing context while inference runs those functions, records random choices and scores, and then wraps completed traces in a `Posterior`.

For the user-facing API, start with `README.md`. For release intent and design notes, see `PLAN.md`.

## Start Here

The most useful reading path is:

1. `cathedral/model.py` for `@model`, `infer()`, `Posterior`, and inference dispatch.
2. `cathedral/trace.py` for trace state, random-choice records, interventions, and scoped execution.
3. `cathedral/primitives.py` for `sample`, `flip`, `condition`, `observe`, `factor`, `mem`, and `DPmem`.
4. One inference engine in `cathedral/inference/`, usually `rejection.py` or `enumeration.py`, to see how engines drive repeated traced executions.
5. `cathedral/distributions.py` for the distribution contract used by primitives and engines.

The reference material under `reference_material/` is useful background, but it is not part of the installable `cathedral` package.

## Package Map

- `cathedral/__init__.py`: public exports. If a new distribution or public helper should be imported as `from cathedral import ...`, it belongs here.
- `cathedral/model.py`: public model decorator, `infer()` entry point, `Posterior`, `InferenceInfo`, weighted posterior analysis, and method dispatch.
- `cathedral/trace.py`: `Trace`, `Choice`, `TraceContext`, active trace `ContextVar`, `run_with_trace()`, and control-flow exceptions.
- `cathedral/_rng.py`: internal NumPy `Generator` creation and active RNG context.
- `cathedral/primitives.py`: trace-aware modeling primitives.
- `cathedral/distributions.py`: `Distribution` base class and built-in probability distributions.
- `cathedral/inference/`: rejection sampling, importance sampling, single-site Metropolis-Hastings, and exact enumeration engines.
- `cathedral/checks.py`: prior predictive, posterior predictive, conditioning difficulty, and model comparison helpers.
- `cathedral/viz.py`: text and Graphviz trace visualization.
- `cathedral/plots.py`: optional Matplotlib diagnostics behind the `cathedral[viz]` extra.
- `examples/`: runnable tutorial-style examples.
- `benchmarks/`: lightweight benchmark scripts for inference behavior.
- `tests/`: pytest coverage for distributions, traces, primitives, inference, diagnostics, checks, visualization, plots, and ArviZ export.

## Execution Flow

A normal inference call follows this path:

1. A user writes a Python function and decorates it with `@model`.
2. The decorator in `cathedral/model.py` returns a wrapper and stores the undecorated callable on `_original_fn`.
3. `infer()` unwraps `_original_fn`, optionally wraps the model with an external return-value `condition=`, installs the requested scope-capture setting, and dispatches to `_run_inference()`.
4. The selected engine repeatedly calls `run_with_trace()` from `cathedral/trace.py`.
5. `run_with_trace()` creates a fresh `TraceContext`, installs it in a `ContextVar`, installs an active RNG, executes the model, stores the return value on the trace, and restores the previous context.
6. Calls to Cathedral primitives consult the active trace context. `sample()` records `Choice` objects, `observe()` and `factor()` add log-score, and failed `condition()` raises `Rejected`.
7. The engine returns completed `Trace` objects and diagnostic metadata.
8. `_run_inference()` builds an `InferenceInfo` and returns a `Posterior`.

The important design point is that user model code never receives a trace object. Trace state and RNG state are scoped around execution with `ContextVar`s.

## Core Data Model

`Trace` is the runtime record of one model execution. It contains:

- `choices`: address to `Choice` records for every random choice made during execution.
- `log_score`: accumulated score from `observe()`, `factor()`, and failed hard conditions.
- `result`: the model's return value.
- `log_joint`: sum of all choice log probabilities plus `log_score`.

`Choice` stores the address, distribution, realized value, log probability, and optional scope path for visualization.

`TraceContext` owns the mutable execution state while a model runs. It assigns addresses, stores interventions for replay, holds per-trace memoization caches, tracks optional scope paths, and knows whether enumeration mode is active.

`Posterior` is the analysis wrapper returned to users. It keeps the traces and their return values, optionally stores normalized weights, and provides summary helpers such as `mean()`, `std()`, `probability()`, `histogram()`, `credible_interval()`, `summary()`, `diagnostics()`, `save()`, `load()`, `extend()`, and `to_arviz()`.

## Addresses and Interventions

Every random choice needs a stable address. Users can provide one with `sample(dist, name="mu")` or `flip(0.5, name="rain")`; otherwise Cathedral generates addresses from the distribution type, such as `Bernoulli` and `Bernoulli__1`.

Interventions are replay values keyed by address. They are used by engines that need to re-run a model while pinning some choices:

- Enumeration forks execution by adding one intervention per support value at the next unexpanded discrete site.
- MH proposes a new value at one site while replaying the other current choices.
- Posterior predictive checks replay latent choices from existing traces.

Address stability matters for these features. Changes to auto-addressing, naming, or stochastic control flow can affect replay behavior and fixed-structure diagnostics.

## Primitives

`sample(dist, name=None)` is the central primitive. Outside inference, it calls `dist.sample()` directly. Inside inference, it chooses an address, applies any intervention, handles enumeration forking if needed, records the choice, and returns the value.

`flip(p, name=None)` is sugar for `sample(Bernoulli(p), name=name)`.

`condition(predicate)` is hard conditioning. If the predicate is false, it adds `-inf` to the active trace score and raises `Rejected`.

`observe(dist, value)` is soft conditioning. It adds `dist.log_prob(value)` to the active trace score.

`factor(score)` adds an arbitrary log-score.

`mem(fn)` gives per-trace stochastic memoization: the same arguments return the same result within a trace. `DPmem(alpha, fn)` adds Chinese Restaurant Process-style reuse and uses the active Cathedral RNG.

## RNG Model

Cathedral uses `numpy.random.Generator` through `cathedral/_rng.py`.

`infer(..., seed=...)` creates a per-run generator and passes it into engine execution. `run_with_trace()` installs that generator as the active Cathedral RNG while the model executes. Distribution implementations and stochastic helpers call `get_active_rng()` rather than global `np.random`.

The reproducibility contract covers random choices made through Cathedral primitives, distributions, inference resampling, MH transitions, and `DPmem()`. Raw `np.random` calls inside user model code are outside that contract.

## Inference Engines

All engines return `list[Trace]` and may populate an internal `_info` dictionary that `model.py` converts into public `InferenceInfo`.

`rejection.py` repeatedly runs the model and discards executions that raise `Rejected`. Accepted traces are equally weighted. It is best when hard conditions are not too rare.

`importance.py` forward-samples traces and uses each trace's `log_score` as the importance log weight. With `resample=True`, it returns an unweighted resampled posterior. With `resample=False`, `model.py` normalizes the log weights and returns a weighted `Posterior`. Use resampling when you want ordinary posterior samples; keep weights when you want diagnostics, ESS, marginal likelihood estimates, or lower-variance weighted summaries.

`mh.py` implements single-site Metropolis-Hastings. Each step selects one existing address, replays all other current choices via interventions, lets the selected site resample from its prior, and accepts or rejects using a structural correction for changing trace shapes. `Posterior.extend()` continues MH from the last retained trace.

`enumeration.py` exactly explores finite discrete execution paths. In enumeration mode, `sample()` raises `NeedsEnumeration` when it reaches an un-intervened finite-support distribution. The enumerator forks the worklist with one intervention per support value and re-executes until complete paths are found.

## Weights and Scores

Choice log probabilities represent the prior probability of sampled latent choices. Trace `log_score` represents conditioning evidence from `observe()`, `factor()`, and hard conditions.

Approximate engines use these values differently:

- Rejection returns only accepted traces, so posterior traces are unweighted.
- Importance uses `log_score` as the log weight because the proposal is the prior.
- Enumeration weights completed traces by normalized `log_joint`.
- MH accepts or rejects proposals using the old and new trace probabilities plus corrections for proposal symmetry and structural changes.

When changing scoring or weighting behavior, check `Posterior` methods too. Weighted posteriors should make `mean()`, `std()`, `probability()`, `histogram()`, intervals, and ESS agree with the stored weights.

## Adding a Distribution

Add the class in `cathedral/distributions.py` by subclassing `Distribution`.

Implement:

- `sample()` using `get_active_rng()`.
- `log_prob(value)` with `-math.inf` for impossible values.
- `support()` if the distribution has small finite support and should work with exact enumeration.
- `__repr__()` for readable traces and diagnostics.

Then export it from `cathedral/__init__.py`, add tests in `tests/test_distributions.py`, and update `README.md` if it is part of the public API.

## Adding an Inference Method

Add a new module under `cathedral/inference/` and expose it from `cathedral/inference/__init__.py` if it should be public. Match the existing engine shape:

- Accept `model_fn`, `args`, optional `kwargs`, `num_samples` where relevant, `seed` where stochastic, and optional `_info`.
- Use `make_rng(seed)` for engine-level randomness.
- Execute models through `run_with_trace()`.
- Return traces, not a `Posterior`.
- Populate `_info` with diagnostics that `InferenceInfo` can expose.

Wire the method into `_run_inference()` in `cathedral/model.py`, then cover public behavior in `tests/test_inference.py` and diagnostics in `tests/test_diagnostics.py` if new metadata is exposed.

## Debugging Implementation Issues

Useful tools while developing:

- `posterior.diagnostics()` for attempts, acceptance rate, ESS, log marginal likelihood, and fixed-structure status.
- `cathedral.viz.print_trace()` to inspect one execution.
- `cathedral.viz.structure_summary()` to understand variable-structure posteriors.
- `cathedral.viz.compare_traces()` to compare replay or MH proposal behavior.
- `infer(..., capture_scopes=True)` to include call-stack-derived scope paths on choices.
- `condition_acceptance_rate()` in `cathedral/checks.py` to estimate whether rejection sampling is viable.

If a model behaves unexpectedly, inspect one trace first, then inspect the engine loop. Most bugs come from address instability, unexpected `Rejected` exceptions, weight normalization, or a distribution that does not implement `log_prob()` and `support()` consistently.

## Tests and Tooling

Install development dependencies with:

```bash
uv sync --group dev --extra viz
```

Run the same checks as CI with:

```bash
uv run ruff check .
uv run ruff format --check .
uvx ty check cathedral/
uv run pytest tests/ -v
```

CI runs these checks on Python 3.10, 3.11, 3.12, and 3.13. Pre-commit runs file hygiene hooks, Ruff, Ruff format, `ty`, and Prettier.

Targeted test files:

- `tests/test_trace.py` for trace recording and execution context behavior.
- `tests/test_primitives.py` for primitive semantics and memoization.
- `tests/test_distributions.py` for distribution contracts.
- `tests/test_inference.py` for engine behavior.
- `tests/test_diagnostics.py` for `Posterior` and `InferenceInfo` behavior.
- `tests/test_checks.py` for predictive checks and model comparison.
- `tests/test_viz.py`, `tests/test_plots.py`, and `tests/test_arviz.py` for optional analysis integrations.

## Contributor Mental Model

When making implementation changes, keep these invariants in mind:

- Model functions should remain ordinary Python functions.
- Trace state and RNG state should be scoped to execution, not passed through user APIs.
- Engines should call `run_with_trace()` rather than invoking models directly.
- Public `infer()` should stay small and route complexity to engines or helpers.
- Distribution sampling must use Cathedral's active RNG.
- Enumeration only works for finite-support distributions.
- Weighted posteriors must keep analysis methods weight-aware.
- Posterior extension should be same-method only; cross-method reuse should create a separate posterior rather than silently appending incompatible traces.
- Pickle-backed posterior persistence is for trusted local checkpointing, not a portable interchange format.
- Fixed-structure assumptions should be explicit; variable-structure traces are a supported feature.
