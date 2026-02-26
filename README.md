# Cathedral

<p align="center">
  <img src="assets/logo.jpeg" alt="Cathedral Logo" width="300"/>
</p>

[![CI](https://img.shields.io/github/actions/workflow/status/andrewgiessel/cathedral/python-package.yml?branch=main)](https://github.com/andrewgiessel/cathedral/actions/workflows/python-package.yml)

A Pythonic probabilistic programming library inspired by [Church](https://cocolab.stanford.edu/papers/GoodmanEtAl2008-UncertaintyInArtificialIntelligence.pdf) and [WebPPL](http://webppl.org/). Write probabilistic models as plain Python functions, then run inference to get posteriors.

Cathedral fills a gap in the Python ecosystem: **Church/WebPPL-level expressiveness** (stochastic control flow, recursive models, stochastic memoization) with **Pythonic syntax** and access to Python's scientific computing stack.

## Quick Start

```python
from cathedral import model, infer, flip, condition

@model
def sprinkler():
    rain = flip(0.3)
    sprinkler_on = flip(0.5)
    if rain:
        wet = flip(0.9)
    elif sprinkler_on:
        wet = flip(0.8)
    else:
        wet = flip(0.1)
    condition(wet)
    return {"rain": rain, "sprinkler": sprinkler_on}

# Exact answer via enumeration
posterior = infer(sprinkler, method="enumerate")
print(f"P(rain | wet grass) = {posterior.probability('rain'):.4f}")  # 0.4615...

# Or approximate via sampling
posterior = infer(sprinkler, method="rejection", num_samples=10000)
posterior = infer(sprinkler, method="mh", num_samples=5000, burn_in=1000)
```

## Installation

```bash
pip install cathedral

# With visualization extras (matplotlib, graphviz, arviz)
pip install cathedral[viz]
```

Requires Python >= 3.10, numpy, and scipy.

## Primitives

| Primitive | Description |
|-----------|-------------|
| `flip(p)` | Flip a coin with probability `p` of True |
| `sample(dist)` | Draw from any distribution (`Normal`, `Beta`, `Gamma`, ...) |
| `condition(pred)` | Hard conditioning: reject execution if `pred` is False |
| `observe(dist, val)` | Soft conditioning: score execution by `dist.log_prob(val)` |
| `factor(score)` | Add arbitrary log-probability to the trace |
| `mem(fn)` | Stochastic memoization: same args always return same random result |
| `DPmem(alpha, fn)` | Dirichlet Process memoization for nonparametric models |

## Inference Methods

| Method | Syntax | Best for |
|--------|--------|----------|
| **Rejection sampling** | `infer(m, method="rejection")` | Small discrete models with `condition()` |
| **Importance sampling** | `infer(m, method="importance")` | Continuous models with `observe()` |
| **Single-site MH** | `infer(m, method="mh")` | Complex models, rare conditions, many latent variables |
| **Exact enumeration** | `infer(m, method="enumerate")` | Small discrete models where you want exact answers |

### MH options

```python
infer(model, method="mh", num_samples=5000, burn_in=1000, lag=2)
```

### Enumeration options

```python
infer(model, method="enumerate", strategy="likely_first", max_executions=1000)
```

Strategies: `depth_first` (default), `breadth_first`, `likely_first`.

## Distributions

**Continuous:** `Normal`, `HalfNormal`, `Beta`, `Gamma`, `Uniform`

**Discrete:** `Bernoulli`, `Categorical`, `UniformDraw`, `Poisson`, `Geometric`

**Multivariate:** `Dirichlet`

All distributions support `.sample()`, `.log_prob(value)`, and `.prob(value)`. Discrete distributions also support `.support()` for enumeration.

## Posterior Analysis

```python
posterior = infer(my_model, method="rejection", num_samples=5000)

posterior.mean("param")                  # posterior mean
posterior.std("param")                   # posterior std
posterior.probability("flag")            # P(flag = True)
posterior.probability(lambda r: r > 0)   # P(predicate)
posterior.histogram("param")             # empirical distribution
posterior.credible_interval(0.95, "param")  # 95% credible interval
```

## Diagnostics & Model Understanding

Every inference run now returns diagnostic metadata automatically:

```python
posterior = infer(sprinkler, method="rejection", num_samples=1000)
print(posterior.diagnostics())
# Posterior: 1000 samples
#   method: rejection
#   attempts: 1694
#   acceptance rate: 0.5903
#   fixed structure: True

posterior.acceptance_rate     # fraction of proposals accepted (rejection/MH)
posterior.ess                 # effective sample size (importance/enumeration)
posterior.log_marginal_likelihood  # for model comparison (importance/enumerate)
posterior.has_fixed_structure # whether all traces share the same addresses
```

### Prior Predictive Checks

```python
from cathedral.checks import prior_predictive, condition_acceptance_rate

# Forward-sample to see what the model generates before conditioning
pp = prior_predictive(sprinkler, num_samples=5000)
print(f"Prior P(rain) = {pp.mean('rain'):.3f}")

# How hard is your inference problem?
rate = condition_acceptance_rate(sprinkler, num_samples=10000)
print(f"Condition satisfied {rate:.1%} of the time")
```

### Posterior Predictive Checks

```python
from cathedral.checks import posterior_predictive

# Replay the model with posterior latent values to generate new data
pp = posterior_predictive(posterior, my_model, num_samples=200)
```

### Model Comparison

```python
from cathedral.checks import compare_models

pa = infer(model_a, method="enumerate")
pb = infer(model_b, method="enumerate")
print(compare_models({"model_a": pa, "model_b": pb}))
# Reports log marginal likelihood and Bayes factors
```

### Trace Visualization

```python
from cathedral.viz import print_trace, structure_summary, address_frequency, trace_to_dot

# Text tree of a single trace (uses scope_path for hierarchy)
posterior = infer(my_model, method="rejection", num_samples=100, capture_scopes=True)
print_trace(posterior.traces[0])

# Posterior structure analysis (especially useful for variable-structure models)
print(structure_summary(posterior))
print(address_frequency(posterior))

# Graphviz DOT output
dot = trace_to_dot(posterior.traces[0])
```

### Diagnostic Plots (requires `cathedral[viz]`)

```python
from cathedral.plots import plot_posterior, plot_weights, plot_trace_values, plot_ess

plot_posterior(posterior, key="rain")        # histogram/KDE of return values
plot_weights(posterior)                      # importance weight distribution
plot_trace_values(posterior, "rain")         # mixing diagnostic for MH
plot_ess(posterior)                          # ESS per address
```

### ArviZ Integration (requires `cathedral[viz]`)

```python
# Convert fixed-structure posteriors to ArviZ InferenceData
idata = posterior.to_arviz()
# Then use the full ArviZ visualization/diagnostic suite
```

## Examples

The `examples/` directory contains runnable demonstrations inspired by [Probabilistic Models of Cognition](http://probmods.org/):

| File | Topics |
|------|--------|
| `01_generative_models.py` | Coin flips, composition, `mem`, stochastic recursion, causal models |
| `02_conditioning.py` | Bayesian reasoning, causal vs diagnostic inference, explaining away |
| `03_patterns_of_inference.py` | Bayesian updating, Monty Hall, Occam's razor |
| `04_bayesian_data_analysis.py` | Parameter estimation, model comparison, linear regression |
| `05_mixture_models.py` | Gaussian mixtures, `DPmem` for infinite components |
| `06_social_cognition.py` | Goal inference, preference learning, theory of mind |
| `07_grammars_and_recursion.py` | PCFGs, random arithmetic, conditioned generation |

Plus standalone examples: `sprinkler.py`, `coin_flip.py`, `linear_regression.py`.

## Architecture

Models are plain Python functions. A trace-based execution engine (via `contextvars`) records every random choice without passing trace objects through user code. Inference engines run models repeatedly, using interventions to replay or modify choices.

```
User code           Trace engine           Inference
─────────           ────────────           ─────────
@model fn    →    TraceContext      →    rejection / importance
flip/sample  →    Choice records    →    MH (propose + accept)
condition    →    log_score         →    enumeration (worklist)
observe      →    log_score         →    Posterior
```

## References

- [Church: A Language for Generative Models](https://arxiv.org/pdf/1206.3255v2.pdf) -- Goodman, Mansinghka, Roy, Bonawitz, Tenenbaum
- [Lightweight Implementations of Probabilistic Programming Languages](http://web.stanford.edu/~ngoodman/papers/lightweight-mcmc-aistats2011.pdf) -- Wingate, Stuhlmuller, Goodman
- [Probabilistic Models of Cognition](http://probmods.org/) -- Goodman & Tenenbaum
- [WebPPL](http://webppl.org/) -- Goodman & Stuhlmuller
- [Gen.jl](https://www.gen.dev/) -- Cusumano-Towner, Saad, Lew, Mansinghka
- [From Word Models to World Models](https://arxiv.org/abs/2306.12672) -- Wong, Grand, Lew, Goodman et al.

## License

MIT -- see [LICENSE](LICENSE).
