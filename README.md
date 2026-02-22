# Cathedral

<p align="center">
  <img src="assets/logo.jpeg" alt="Cathedral Logo" width="300"/>
</p>

[![Release](https://img.shields.io/github/v/release/andrewgiessel/cathedral)](https://img.shields.io/github/v/release/andrewgiessel/cathedral)
[![Build status](https://img.shields.io/github/actions/workflow/status/andrewgiessel/cathedral/main.yml?branch=main)](https://github.com/andrewgiessel/cathedral/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/andrewgiessel/cathedral)](https://img.shields.io/github/commit-activity/m/andrewgiessel/cathedral)

## About

Cathedral is a Pythonic probabilistic programming library inspired by [Church](https://cocolab.stanford.edu/papers/GoodmanEtAl2008-UncertaintyInArtificialIntelligence.pdf) and [WebPPL](http://webppl.org/). Write probabilistic models as plain Python functions, then run inference to get posteriors.

Cathedral aims to fill a gap in the Python ecosystem: **Church/WebPPL-level expressiveness** (stochastic control flow, recursive models, stochastic memoization) with **Pythonic syntax** and access to Python's scientific computing ecosystem.

- **Github repository**: <https://github.com/andrewgiessel/cathedral/>

## Quick Example

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

posterior = infer(sprinkler, method="rejection", num_samples=5000)
print(f"P(rain | wet grass) = {posterior.probability('rain'):.3f}")
```

## Features

- **Pure Python models**: No special syntax -- models are just decorated Python functions
- **Flexible conditioning**: `condition()` for hard constraints, `observe()` for fitting data
- **Multiple inference methods**: Rejection sampling, likelihood-weighted importance sampling (MCMC coming soon)
- **Rich posterior analysis**: `.mean()`, `.std()`, `.probability()`, `.histogram()`, `.credible_interval()`
- **Trace-based execution**: Every random choice is recorded for inspection and inference

## Installation

```bash
pip install cathedral
```

## Core Primitives

| Primitive | Description |
|-----------|-------------|
| `flip(p)` | Flip a coin with probability `p` of heads |
| `sample(dist)` | Draw from any distribution (`Normal`, `Beta`, `Gamma`, ...) |
| `condition(pred)` | Reject execution if `pred` is False |
| `observe(dist, value)` | Soft-condition: the value was drawn from this distribution |
| `factor(score)` | Add arbitrary log-probability score |

## References

- [Church: A Language for Generative Models](https://arxiv.org/pdf/1206.3255v2.pdf) - Goodman et al.
- [From Word Models to World Models](https://arxiv.org/abs/2306.12672) - Wong, Grand, Lew, Goodman et al.
- [WebPPL](http://webppl.org/) - Goodman & Stuhlmuller
- [Gen.jl](https://www.gen.dev/) - Cusumano-Towner et al.
- [Probabilistic Models of Cognition](http://probmods.org/) - Goodman & Tenenbaum

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
