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
  distributions.py     # Distribution base class + 12 concrete distributions
  primitives.py        # sample, flip, condition, observe, factor, mem, DPmem
  trace.py             # Choice, Trace, TraceContext, run_with_trace
  model.py             # @model decorator, Posterior class, infer() entry point
  inference/
    __init__.py
    rejection.py       # Rejection sampling (for condition()-based models)
    importance.py      # Likelihood-weighted importance sampling (for observe())
```

### Data Flow

1. User writes a `@model`-decorated function using primitives (`sample`, `flip`, `condition`, `observe`, `factor`, `mem`)
2. `infer(model_fn, method=...)` calls the chosen inference engine
3. The engine calls `run_with_trace(fn)` repeatedly, which:
   - Creates a `TraceContext` (stored in a `contextvar`)
   - Executes the model function; each `sample()`/`flip()` call records a `Choice` in the trace
   - `condition()` raises `Rejected` on failure; `observe()`/`factor()` add to `log_score`
   - Returns the completed `Trace` with all choices, scores, and the return value
4. The engine collects traces (rejecting failed ones, or weighting by score)
5. Results are wrapped in a `Posterior` object for analysis

### Key Abstractions

- **Distribution**: ABC with `sample()` and `log_prob(value)`. Current implementations: Bernoulli, Categorical, Normal, HalfNormal, Beta, Gamma, Uniform, Poisson, UniformDraw, Geometric, Dirichlet
- **Trace**: Dataclass holding `choices` (dict of address -> Choice), `log_score`, and `result`
- **TraceContext**: Manages auto-addressing, intervention support, and per-trace memo caches
- **Posterior**: Wraps a list of traces with analysis methods (mean, std, probability, histogram, credible_interval)

## What's Done (Layers 1-2)

### Layer 1: Core DSL + Basic Inference
- All core primitives: `sample()`, `flip()`, `condition()`, `observe()`, `factor()`
- `@model` decorator and `infer()` entry point
- Rejection sampling and likelihood-weighted importance sampling
- 12 distributions with analytic `log_prob` for Normal/HalfNormal (performance-critical)
- Trace infrastructure with auto-addressing and intervention support
- `Posterior` class with full analysis API
- 84 tests passing, 3 standalone examples (sprinkler, coin_flip, linear_regression)

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

## What's Next

### Layer 3: Single-Site Metropolis-Hastings (Priority: HIGH)

**Why**: Rejection sampling is exponentially slow for models with many conditions. Importance sampling works for `observe()` but has poor effective sample size in high dimensions. MH enables efficient inference for a much broader class of models.

**What to build**:
- `cathedral/inference/mh.py` with single-site MH
- Pick a random address in the trace, propose a new value from the prior, accept/reject based on the MH ratio
- Integrate with `infer(..., method="mh")` 
- The trace intervention mechanism already supports this (pass `interventions` dict to `run_with_trace`)

**Key considerations**:
- Need to handle `condition()` (hard constraints) -- traces that hit `Rejected` get score -inf
- `mem()` interactions: memoized values are in the trace implicitly through the choices they made; re-proposing those choices should work naturally
- Burn-in and thinning parameters

### Layer 4: Enumeration (Priority: MEDIUM)

**Why**: For small discrete models, exact inference is possible and gives ground truth. Useful for testing and for models where sampling is wasteful.

**What to build**:
- `cathedral/inference/enumerate.py` 
- Enumerate all possible executions of a model (only works for finite, discrete choice spaces)
- Return exact posterior probabilities
- Integrate with `infer(..., method="enumerate")`

### Layer 5: README + Documentation Update (Priority: MEDIUM)

**What to update**:
- README.md: add `mem`/`DPmem` to features and primitives table
- README.md: mention the 7 ProbMods example files
- README.md: add a "Quickstart" section showing `mem` usage
- Consider adding a `docs/` folder with a tutorial narrative

### Layer 6: Trace Visualizer (Priority: LOW)

**What to build**:
- Pretty-print or HTML visualization of a trace showing the tree of random choices
- Could use graphviz or a simple text-based tree
- Useful for debugging models and for teaching

### Layer 7: MCP Server (Priority: DEFERRED)

**Why**: The ultimate goal is for LLM agents to write and run Cathedral models as tools. An MCP server would expose tools like:
- `create_model(code)` -- validate and register a model
- `run_inference(model_name, method, num_samples)` -- run inference and return summary
- `analyze_posterior(query)` -- answer questions about the posterior

**Deferred until**: Core inference is solid and we have enough examples to validate the API surface.

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
- [From Word Models to World Models](https://arxiv.org/abs/2306.12672) - Wong, Grand, Lew, Goodman et al.
- [WebPPL](http://webppl.org/) - Goodman & Stuhlmuller
- [Gen.jl](https://www.gen.dev/) - Cusumano-Towner et al.
- [Probabilistic Models of Cognition (Church)](https://v1.probmods.org/) - Goodman & Tenenbaum
- [Probabilistic Models of Cognition (WebPPL)](https://probmods.org/) - Goodman & Tenenbaum
- [Pyro](https://pyro.ai/) - Uber AI Labs
- [Large Language Bayes](https://arxiv.org/abs/2308.13111) - Goodman et al. (LLMs writing PyMC)
