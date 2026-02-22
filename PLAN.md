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
  model.py             # @model decorator, Posterior class (weight-aware), infer() entry point
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
4. The engine collects traces (rejecting failed ones, weighting by score, or accepting via MH ratio)
5. Results are wrapped in a `Posterior` object for analysis (weight-aware for enumeration)

### Key Abstractions

- **Distribution**: ABC with `sample()`, `log_prob(value)`, and `support()` (returns finite support for discrete distributions, `None` for continuous). Implementations: Bernoulli, Categorical, Normal, HalfNormal, Beta, Gamma, Uniform, Poisson, UniformDraw, Geometric, Dirichlet
- **Trace**: Dataclass holding `choices` (dict of address -> Choice), `log_score`, and `result`. `log_joint` property sums all choice log-probs and the log-score.
- **TraceContext**: Manages auto-addressing, intervention support (for MH replay), per-trace memo caches, and `enumerate_mode` flag
- **Posterior**: Wraps a list of traces with optional weights. Analysis methods (mean, std, probability, histogram, credible_interval) are all weight-aware for exact enumeration results.

## What's Done (Layers 1-5)

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
- 7 ProbMods-inspired example files with dual Church/WebPPL chapter links:
  - `01_generative_models.py` -- coins, mem, stochastic recursion, tug of war
  - `02_conditioning.py` -- Bayesian reasoning, explaining away, soft conditioning
  - `03_patterns_of_inference.py` -- Bayesian updating, disease tests, Monty Hall, Occam's razor
  - `04_bayesian_data_analysis.py` -- parameter estimation, model comparison, regression
  - `05_mixture_models.py` -- Gaussian mixtures, category learning, DPmem clustering, topic models
  - `06_social_cognition.py` -- goal inference, preferences, false belief, deception
  - `07_grammars_and_recursion.py` -- PCFGs, arithmetic expressions, conditioned generation
- Ruff formatting (120-char lines) and lint across entire codebase

### Layer 3: Single-Site Metropolis-Hastings
- `cathedral/inference/mh.py` with Wingate et al. (2011) lightweight MH
- Prior proposals: pick a random choice site, propose from its prior, re-execute with replay
- Full MH acceptance ratio with structural change corrections (appearing/disappearing sites)
- Supports `burn_in`, `lag` (thinning), and `max_init_attempts`
- Integrated via `infer(model, method="mh")`

### Layer 4: Exact Enumeration
- `cathedral/inference/enumeration.py` with worklist-based enumerator
- Re-execution approach: when `sample()` hits a discrete site in enumerate mode, raises `NeedsEnumeration`; the enumerator forks into one path per support value via interventions
- Three traversal strategies: `depth_first`, `breadth_first`, `likely_first` (priority queue on log-prob)
- `max_executions` cap for approximate enumeration of large models
- `support()` method on Bernoulli, Categorical, UniformDraw
- Weight-aware `Posterior` for exact probability computation
- `marginals_from_traces()` utility for computing exact marginal distributions
- Integrated via `infer(model, method="enumerate")`

### Layer 5: README + Examples Refresh
- Complete README rewrite documenting all four inference methods, distributions, primitives, and examples
- Standalone examples (sprinkler, coin_flip, linear_regression) updated to showcase enumeration and MH alongside sampling
- 108 tests passing across distributions, trace, primitives, and all four inference engines

## What's Next

### Layer 6: Trace Visualizer (Priority: MEDIUM)

**What to build**:
- Pretty-print or HTML visualization of a trace showing the tree of random choices
- Could use graphviz or a simple text-based tree
- Useful for debugging models and for teaching

### Layer 7: MCP Server (Priority: MEDIUM)

**Why**: The ultimate goal is for LLM agents to write and run Cathedral models as tools. An MCP server would expose tools like:
- `create_model(code)` -- validate and register a model
- `run_inference(model_name, method, num_samples)` -- run inference and return summary
- `analyze_posterior(query)` -- answer questions about the posterior

### Future: Gradient-Based Inference (Priority: EXPLORATORY)

**HMC/NUTS/VI** would enable efficient inference for continuous models but requires:
- Differentiable log-density (autograd compatibility)
- A "static model" subset where the program structure doesn't depend on random choices
- Possible backends: JAX (via NumPyro), PyTorch (via Pyro)

The architecture supports a "two-mode" approach:
- **Interpreted mode** (current): full generality, gradient-free inference
- **Compiled mode** (future): restricted to static models, compiled to a JAX/PyTorch computation graph for HMC/NUTS/VI

This is inspired by Gen.jl's approach of having both `DynamicDSL` (full generality) and `StaticDSL` (compiled for speed).

## Development Setup

```bash
# Clone and install
git clone https://github.com/andrewgiessel/cathedral
cd cathedral
uv pip install -e ".[dev]"

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
