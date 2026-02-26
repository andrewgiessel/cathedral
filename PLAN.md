# Cathedral Development Plan

## Vision

Cathedral is a Pythonic probabilistic programming library inspired by Church and WebPPL. The core motivation is to provide a clean way for LLMs to generate probabilistic world models -- inspired by the paper ["From Word Models to World Models"](https://arxiv.org/abs/2306.12672) (Wong, Grand, Lew, Goodman et al.).

Cathedral aims for **Church/WebPPL-level expressiveness** (stochastic control flow, recursive models, stochastic memoization) with **Pythonic syntax** that modern LLMs can easily generate.

Key design decisions:
- Pure Python DSL (no Scheme/Hy syntax) -- models are decorated Python functions
- Trace-based execution (inspired by Pyro/Gen.jl) records every random choice
- `contextvars` for implicit trace threading (no explicit state passing)
- scipy/numpy backend for now, with future path to JAX/PyTorch for gradient-based inference

## Architecture

```
cathedral/
  __init__.py          # Public API: re-exports everything below
  distributions.py     # Distribution base class + 11 concrete distributions (with support() for discrete)
  primitives.py        # sample, flip, condition, observe, factor, mem, DPmem
  trace.py             # Choice, Trace, TraceContext, run_with_trace, NeedsEnumeration
  model.py             # @model decorator, InferenceInfo, Posterior class, infer() entry point
  checks.py            # prior_predictive, condition_acceptance_rate, posterior_predictive, compare_models
  viz.py               # format_trace, structure_summary, address_frequency, compare_traces, trace_to_dot
  plots.py             # plot_posterior, plot_weights, plot_trace_values, plot_ess (optional matplotlib)
  inference/
    __init__.py
    rejection.py       # Rejection sampling (for condition()-based models)
    importance.py      # Likelihood-weighted importance sampling (for observe())
    mh.py              # Single-site Metropolis-Hastings (Wingate et al. 2011)
    enumeration.py     # Exact enumeration (worklist with depth/breadth/likely-first strategies)
```

### Data Flow

1. User writes a `@model`-decorated function using primitives (`sample`, `flip`, `condition`, `observe`, `factor`, `mem`)
2. `infer(model_fn, method=...)` calls the chosen inference engine
3. The engine calls `run_with_trace(fn)` repeatedly, which:
   - Creates a `TraceContext` (stored in a `contextvar`)
   - Executes the model function; each `sample()`/`flip()` call records a `Choice` in the trace
   - `condition()` raises `Rejected` on failure; `observe()`/`factor()` add to `log_score`
   - In enumerate mode, `sample()` raises `NeedsEnumeration` for un-intervened discrete sites
   - Returns the completed `Trace` with all choices, scores, and the return value
4. The engine collects traces and populates an `InferenceInfo` with diagnostic metadata
5. Results are wrapped in a `Posterior` object for analysis (weight-aware for enumeration)

### Key Abstractions

- **Distribution**: ABC with `sample()`, `log_prob(value)`, and `support()` (returns finite support for discrete distributions, `None` for continuous). Implementations: Bernoulli, Categorical, Normal, HalfNormal, Beta, Gamma, Uniform, Poisson, UniformDraw, Geometric, Dirichlet
- **Choice**: Dataclass holding address, distribution, value, log_prob, and `scope_path` (for hierarchical visualization)
- **Trace**: Dataclass holding `choices` (dict of address -> Choice), `log_score`, and `result`. `log_joint` property sums all choice log-probs and the log-score.
- **TraceContext**: Manages auto-addressing, intervention support (for MH replay), per-trace memo caches, `enumerate_mode` flag, and scope capture via stack introspection
- **InferenceInfo**: Dataclass holding diagnostic metadata: method, num_samples, acceptance_rate, log_weights, log_marginal_likelihood, ESS
- **Posterior**: Wraps a list of traces with optional weights and InferenceInfo. Analysis methods (mean, std, probability, histogram, credible_interval) are all weight-aware. Diagnostic properties (ess, acceptance_rate, log_marginal_likelihood, has_fixed_structure) and `to_arviz()` bridge.

---

## What's Done

### Layer 1: Core DSL + Basic Inference
- All core primitives: `sample()`, `flip()`, `condition()`, `observe()`, `factor()`
- `@model` decorator and `infer()` entry point
- Rejection sampling and likelihood-weighted importance sampling
- 11 distributions with analytic `log_prob` for Normal/HalfNormal (performance-critical)
- Trace infrastructure with auto-addressing and intervention support
- `Posterior` class with full analysis API

### Layer 2: Stochastic Memoization + Examples
- `mem()`: per-trace memoization for persistent random properties
- `DPmem()`: Dirichlet Process stochastic memoizer (Chinese Restaurant Process)
- 7 ProbMods-inspired example files covering generative models, conditioning, Bayesian data analysis, mixture models, social cognition, and grammars/recursion
- Ruff formatting (120-char lines) and lint across entire codebase

### Layer 3: Single-Site Metropolis-Hastings
- `cathedral/inference/mh.py` with Wingate et al. (2011) lightweight MH
- Prior proposals: pick a random choice site, propose from its prior, re-execute with replay
- Full MH acceptance ratio with structural change corrections (appearing/disappearing sites)
- Supports `burn_in`, `lag` (thinning), and `max_init_attempts`

### Layer 4: Exact Enumeration
- `cathedral/inference/enumeration.py` with worklist-based enumerator
- Three traversal strategies: `depth_first`, `breadth_first`, `likely_first`
- `max_executions` cap for approximate enumeration
- `support()` on Bernoulli, Categorical, UniformDraw
- Weight-aware `Posterior` for exact probability computation
- `marginals_from_traces()` utility

### Layer 5: README + Examples Refresh
- Complete README with examples, distributions, primitives, and architecture docs
- Standalone examples (sprinkler, coin_flip, linear_regression)
- 108 tests passing

### Layer 6: Trace Visualization + Scope Capture
- `scope_path` field on `Choice` for hierarchical trace structure
- Automatic stack introspection in `TraceContext._resolve_scope()` using `sys._getframe()` — no manual annotation needed
- `mem()` and `DPmem()` automatically push/pop named scopes, with optional `name=` parameter for lambda identification
- `capture_scopes` flag on both `infer()` and `run_with_trace()`
- `viz.print_trace()` / `viz.format_trace()` text tree renderer with box-drawing connectors
- `viz.trace_to_dot()` for Graphviz DOT output with scope-based subgraph hierarchy

### Layer 7: Model Understanding Toolkit
- **Inference diagnostics**: `InferenceInfo` dataclass populated by all 4 engines. Rejection reports `num_attempts` + `acceptance_rate`. Importance reports `log_weights` + `log_marginal_likelihood` + `ess`. MH reports `acceptance_rate`. Enumeration reports exact `log_marginal_likelihood`.
- **Posterior properties**: `has_fixed_structure`, `ess`, `acceptance_rate`, `log_marginal_likelihood`, `diagnostics()` summary
- **Prior predictive checks** (`checks.py`): `prior_predictive()` forward-samples the model; `condition_acceptance_rate()` estimates conditioning difficulty
- **Posterior predictive checks** (`checks.py`): `posterior_predictive()` replays the model with posterior interventions
- **Model comparison** (`checks.py`): `compare_models()` reports log marginal likelihood and Bayes factors
- **Posterior-level analysis** (`viz.py`): `structure_summary()`, `address_frequency()`, `compare_traces()`
- **Diagnostic plots** (`plots.py`, optional matplotlib): `plot_posterior`, `plot_weights`, `plot_trace_values`, `plot_ess`
- **ArviZ bridge**: `Posterior.to_arviz()` converts fixed-structure posteriors to `InferenceData`
- **Optional deps**: `cathedral[viz]` installs matplotlib, graphviz, arviz
- README overhauled with 5 generated figures (consistent palette), real output examples
- 188 tests passing (80 new across 5 test files)

---

## What's Next

### Layer 8: Custom Proposals for MH (Priority: HIGH)

**The problem**: Cathedral's MH always proposes from the prior. For continuous variables with narrow posteriors, this means almost all proposals are rejected (we saw 9.5% acceptance on the Gaussian mean example). Church's ERPs come with built-in proposers.

**What to build**:
- Add optional `proposal` method to `Distribution` base class
- Gaussian drift proposal for `Normal` and `HalfNormal`: propose `value + Normal(0, step_size)` with tunable step size
- Positive drift proposal for `Gamma` and `Beta`: propose in log-space
- Swap proposal for `Categorical` / `UniformDraw`: propose a different value uniformly
- Modify `_mh_step()` to use the distribution's proposer when available, with correct forward/backward probabilities in the acceptance ratio
- Adaptive step size: tune step size during burn-in to target ~23% acceptance rate (optimal for continuous MH)

**Impact**: Dramatically better mixing for continuous models. Low implementation cost — mostly changes to `mh.py` and `distributions.py`.

### Layer 9: Hamiltonian Monte Carlo (Priority: HIGH)

**The problem**: Single-site MH scales poorly with dimensionality. HMC uses gradients to make large, coherent moves through continuous parameter space.

**What to build**:
- Automatic differentiation via JAX or a lightweight dual-number AD (Church uses its own `AD.ss`)
- Leapfrog integrator
- HMC kernel that separates continuous and discrete choices (Church's approach: only continuous ERPs get gradient proposals, discrete ones use standard MH)
- NUTS (No-U-Turn Sampler) adaptation for automatic tuning of step size and trajectory length
- New inference method: `infer(model, method="hmc")` for static-structure continuous models

**Design options**:
- **Option A**: Lightweight dual-number AD in pure Python/NumPy (like Church's `AD.ss`). No new dependencies, works everywhere, but slow.
- **Option B**: JAX backend. Fast, but requires JAX dependency and restricting models to JAX-compatible operations.
- **Option C**: PyTorch autograd. Similar to Pyro's approach.

Church's approach (Option A) is the most natural fit — it preserves Cathedral's "plain Python" philosophy. Can upgrade to JAX later for performance.

**Impact**: Unlocks efficient inference for continuous models (regression, hierarchical models, etc.). This is the biggest remaining gap vs. PyMC/Stan/Pyro.

### Layer 10: Exchangeable Random Primitives (Priority: MEDIUM)

**The problem**: Cathedral's CRP in `DPmem` recomputes scores from scratch. Church's XRP system maintains sufficient statistics for incremental updates, making MH on hierarchical/nonparametric models much faster.

**What to build**:
- `XRP` base class with `sample()`, `log_prob()`, `incorporate(value)`, `unincorporate(value)`, `propose()` methods
- Dirichlet-Discrete XRP: tracks counts, incremental score = `log(count + α) - log(total + Σα)`
- Beta-Binomial XRP: tracks (successes, failures), conjugate updates
- Normal-Normal-Gamma XRP: tracks (n, sum, sum_sq), posterior predictive is generalized t
- CRP XRP: tracks table counts with incremental scoring (upgrade existing `DPmem` internals)
- Modify `TraceContext` to maintain an XRP registry across MH steps, updating stats when choices change

**Additional nonparametric priors**:
- `PYmem(alpha, d, fn)`: Pitman-Yor Process memoization (two-parameter generalization of DP)
- `sticky_DPmem(alpha, fn)`: DP with stick-breaking representation
- `make_GEM(alpha)`: Stick-breaking process (GEM distribution)

**Impact**: Major speedup for hierarchical and nonparametric models. Medium implementation cost — requires changes to `TraceContext` and `mh.py` to support incremental score updates.

### Layer 11: Advanced MCMC Algorithms (Priority: MEDIUM)

Church implements several MCMC variants beyond single-site MH. These would improve mixing for multimodal or high-dimensional posteriors.

**Tempered Transitions**:
- Symmetric temperature schedule: heat up (flatten posterior), make MH moves, cool down
- Helps escape local modes
- Relatively simple to implement given existing MH infrastructure
- `infer(model, method="mh", tempering=True, num_temperatures=5)`

**Parallel Tempering**:
- Multiple chains at different temperatures, swap proposals between adjacent chains
- More aggressive than tempered transitions, better for strongly multimodal posteriors
- Embarrassingly parallel
- `infer(model, method="mh", parallel_tempering=True, num_chains=8)`

**Multiple-Try Metropolis (MTM)**:
- Generate K proposals per step, select one via importance weighting
- Better mixing than single proposals, especially in high dimensions
- `infer(model, method="mh", num_tries=5)`

**Annealed Importance Sampling (AIS)**:
- Bridge between importance sampling and MCMC
- Better marginal likelihood estimates than plain importance sampling
- Would strengthen `compare_models()` and the model comparison story
- `infer(model, method="ais", num_temperatures=20)`

### Layer 12: Constraint Propagation / Smart Initialization (Priority: MEDIUM-LOW)

**The problem**: MH initialization uses forward sampling, which can fail when conditions are very restrictive (`max_init_attempts` exceeded). Church solves this with backward constraint propagation.

**What to build**:
- Primitive inverse functions: `and_inverse`, `or_inverse`, `equal_inverse`, etc.
- Backward constraint propagation: given a desired output, work backward through the model to find compatible random choices
- Use for MH initialization when forward sampling fails
- Could also improve proposal quality by guiding proposals toward satisfying constraints

**Impact**: Fixes a real pain point (MH failing to initialize on tightly conditioned models), but niche. Lower priority than better proposals and HMC.

### Layer 13: Conservative Trace Updates (Priority: LOW)

**The problem**: Cathedral re-executes the entire model on every MH step. Church caches environment equality checks and only re-evaluates subexpressions that actually changed.

**What to build**:
- Environment fingerprinting: hash the values that each choice depends on
- Partial re-execution: only re-run the parts of the model affected by the proposed change
- Cached operator equality checks

**Impact**: Major speedup for large models with many independent choices. But architecturally complex — requires fundamentally rethinking how `run_with_trace` works. Church builds this into its evaluator; harder to retrofit.

### Layer 14: MCP Server (Priority: MEDIUM)

**Why**: The ultimate goal is for LLM agents to write and run Cathedral models as tools. An MCP server would expose tools like:
- `create_model(code)` — validate and register a model
- `run_inference(model_name, method, num_samples)` — run inference and return summary
- `analyze_posterior(query)` — answer questions about the posterior
- `compare_models(model_a, model_b)` — run model comparison

### Modeling Utilities (ongoing, no layer)

Small additions inspired by Church's standard library that could be added incrementally:

- **Noisy logic**: `noisy_and(p, *args)`, `noisy_or(p, *args)`, `noisify(p, value)` — soft boolean constraints with noise parameter
- **Importance sampling utilities**: `ess(weights)`, `importance_expectation(values, weights)` as standalone functions
- **More distributions**: Exponential, StudentT, Wishart, Multinomial, NegativeBinomial
- **Mixture distribution**: `Mixture(weights, components)` — a first-class distribution that marginalizes over components
- **Proposal distributions**: expose as a user-facing API so users can register custom proposals for their own distributions

---

## Performance & Parallelization (Parked)

Documented from an earlier analysis — intentionally deferred until the codebase is better understood and the algorithmic foundations (custom proposals, HMC, XRPs) are in place. Adding concurrency to code we're still actively changing is a recipe for pain.

### Parallelization

Three of the four engines are straightforwardly parallelizable:

**Rejection & importance sampling — embarrassingly parallel.** Every `run_with_trace()` call is independent. Split `num_samples` across `ProcessPoolExecutor` workers. `contextvars` are per-thread/per-process by design, so each worker gets isolated trace state with no races. Must use processes (not threads) because Cathedral models are interpreter-bound, not NumPy-bound — the GIL blocks true thread-level parallelism.

**MH — multiple independent chains.** Sequential within a chain, but K chains can run in parallel and merge samples. This also unlocks R-hat / Gelman-Rubin convergence diagnostics for free. Standard approach in Stan/PyMC.

**Enumeration — parallel branch exploration.** After the first fork, branches are independent. A work-stealing pattern over a shared worklist could parallelize this, though coordination is more complex than the other two.

### Non-parallelization speed wins

| Idea | Effort | Impact |
|------|--------|--------|
| `slots=True` on `Choice` and `Trace` dataclasses | Trivial | 10-20% less allocation overhead |
| Analytical `log_prob` for Beta, Gamma, Poisson, Geometric, Dirichlet (currently delegate to scipy which has per-call validation overhead) | Low | 2-5x per `log_prob` call |
| Modern RNG: `numpy.random.Generator` + `.spawn()` for reproducible parallel streams | Low | Modest + deterministic parallel runs |
| Vectorized batch sampling for fixed-structure models: detect no stochastic control flow, batch-sample all random values as NumPy arrays, skip per-sample Python loop | Medium | 10-100x for applicable models |
| Trace allocation reuse: pre-allocate and recycle `TraceContext`/`Trace`/`Choice` objects in hot loops | Medium | Reduces GC pressure on long runs |

### Why we're waiting

1. Layers 8-11 will significantly change the inference engines (new proposal mechanisms, AD, XRP registry, temperature schedules). Parallelizing code that's about to be rewritten wastes effort.
2. We need profiling data on real models to know where time actually goes — it might be `log_prob`, or trace allocation, or model execution. Optimizing the wrong thing is worse than not optimizing.
3. The `slots=True` and analytical `log_prob` changes are safe to do anytime as isolated PRs. They don't interact with the algorithmic work.

---

## Recommended Build Order

```
Layer 8: Custom proposals       ← Highest ROI, lowest effort
Layer 9: HMC                   ← Highest impact, medium effort
Layer 10: XRPs                 ← Important for nonparametric models
Layer 11: Advanced MCMC        ← Tempered transitions first, then parallel tempering
Layer 14: MCP server           ← Can start in parallel with above
Layer 12: Constraint prop      ← Nice to have
Layer 13: Conservative updates ← Major refactor, do last
```

Layers 8 and 9 are the clear priorities — they address the two biggest practical limitations of the current system (poor MH mixing on continuous models, and no gradient-based inference).

---

## Test Coverage

188 tests across 9 test files:
- `test_distributions.py` — 22 tests for all distribution types
- `test_trace.py` — 13 tests for Choice, Trace, TraceContext, run_with_trace
- `test_primitives.py` — 18 tests for sample, flip, condition, observe, factor, mem, DPmem
- `test_inference.py` — 35 tests for rejection, importance, MH, enumeration, and Posterior
- `test_diagnostics.py` — 14 tests for InferenceInfo and Posterior diagnostic properties
- `test_checks.py` — 20 tests for prior_predictive, condition_acceptance_rate, posterior_predictive, compare_models
- `test_viz.py` — 18 tests for format_trace, structure_summary, address_frequency, compare_traces, trace_to_dot
- `test_plots.py` — 13 tests for all 4 plot functions
- `test_arviz.py` — 6 tests for Posterior.to_arviz()

All running under `uv run pytest tests/ -v`.

## Development Setup

```bash
# Clone and install
git clone https://github.com/andrewgiessel/cathedral
cd cathedral
uv pip install -e ".[dev]"

# With visualization extras
uv pip install -e ".[viz]"

# Run tests
uv run pytest tests/ -v

# Run all examples
for f in examples/0*.py; do uv run python "$f"; done

# Format and lint
uv run ruff format .
uv run ruff check .
```

## References

- [Church: A Language for Generative Models](https://arxiv.org/pdf/1206.3255v2.pdf) - Goodman et al. 2008
- [Lightweight Implementations of Probabilistic Programming Languages](http://web.stanford.edu/~ngoodman/papers/lightweight-mcmc-aistats2011.pdf) - Wingate, Stuhlmuller, Goodman 2011
- [From Word Models to World Models](https://arxiv.org/abs/2306.12672) - Wong, Grand, Lew, Goodman et al.
- [WebPPL](http://webppl.org/) - Goodman & Stuhlmuller
- [Gen.jl](https://www.gen.dev/) - Cusumano-Towner et al.
- [Probabilistic Models of Cognition (Church)](https://v1.probmods.org/) - Goodman & Tenenbaum
- [Probabilistic Models of Cognition (WebPPL)](https://probmods.org/) - Goodman & Tenenbaum
- [Pyro](https://pyro.ai/) - Uber AI Labs
- [Large Language Bayes](https://arxiv.org/abs/2308.13111) - Goodman et al. (LLMs writing PyMC)
- [MIT Church source](https://github.com/LFY/bher) - Reference implementation studied for Layers 8-13
