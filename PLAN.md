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

Focus the next release on a concrete systems improvement plus stronger example-driven research.

### Release work

- Parallel rejection sampling
- Parallel importance sampling
- Reproducible parallel RNG streams
- Benchmarking on real examples
- Optional multi-chain MH if the API stays clean

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

1. Ship parallel rejection and importance sampling.
2. Run the applied-model survey and select the first ports.
3. Port models and write down concrete gaps.
4. Choose the next milestone based on those gaps.

## Test Coverage

188 tests across 9 test files:

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
