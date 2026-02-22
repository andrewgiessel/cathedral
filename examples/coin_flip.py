"""Coin flip inference example.

Given that a coin landed heads, what's the probability it was a fair coin?
Demonstrates basic Bayesian reasoning with condition().
"""

from cathedral import condition, flip, infer, model


@model
def fair_coin():
    """Is the coin fair, given it landed heads?"""
    is_fair = flip(0.5)
    result = flip(0.5 if is_fair else 0.9)
    condition(result)
    return is_fair


@model
def coin_bias():
    """Infer coin bias from a sequence of observations."""
    from cathedral import observe, sample
    from cathedral.distributions import Bernoulli, Beta

    bias = sample(Beta(1, 1))
    observations = [True, True, False, True, True, True, False, True]
    for obs in observations:
        observe(Bernoulli(bias), obs)
    return bias


if __name__ == "__main__":
    print("=== Fair coin inference ===")
    posterior = infer(fair_coin, method="rejection", num_samples=10000)
    print(f"P(fair | heads) = {posterior.probability():.3f}")
    print(f"  (prior was 0.5, should decrease slightly since biased coin more likely to give heads)")

    print("\n=== Coin bias inference ===")
    posterior = infer(coin_bias, method="importance", num_samples=10000)
    print(f"Inferred bias: {posterior.mean():.3f} +/- {posterior.std():.3f}")
    lo, hi = posterior.credible_interval(level=0.95)
    print(f"95% CI: ({lo:.3f}, {hi:.3f})")
    print(f"  (observed 6/8 heads, expect bias ~0.75)")
