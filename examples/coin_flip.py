"""Coin flip inference example.

Given that a coin landed heads, what's the probability it was a fair coin?
Demonstrates Bayesian reasoning with condition() and observe(), and compares
exact enumeration with sampling-based inference.
"""

from cathedral import condition, flip, infer, model, observe, sample
from cathedral.distributions import Bernoulli, Beta


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
    bias = sample(Beta(1, 1), name="bias")
    observations = [True, True, False, True, True, True, False, True]
    for obs in observations:
        observe(Bernoulli(bias), obs)
    return bias


if __name__ == "__main__":
    # Exact answer for the discrete model
    print("=== Fair coin: exact enumeration ===")
    exact = infer(fair_coin, method="enumerate")
    p_fair = exact.probability()
    print(f"P(fair | heads) = {p_fair:.6f}")
    print(f"  = 0.25 / 0.70 = {0.25 / 0.70:.6f} (analytic)")

    # Compare with sampling methods
    print("\n=== Fair coin: rejection sampling (10,000 samples) ===")
    rej = infer(fair_coin, method="rejection", num_samples=10000)
    print(f"P(fair | heads) = {rej.probability():.4f}")

    print("\n=== Fair coin: single-site MH (10,000 samples) ===")
    mh = infer(fair_coin, method="mh", num_samples=10000, burn_in=2000)
    print(f"P(fair | heads) = {mh.probability():.4f}")

    # Continuous bias estimation (can't enumerate -- uses importance/MH)
    print("\n=== Coin bias: importance sampling (10,000 samples) ===")
    imp = infer(coin_bias, method="importance", num_samples=10000)
    print(f"Inferred bias: {imp.mean():.3f} +/- {imp.std():.3f}")
    lo, hi = imp.credible_interval(level=0.95)
    print(f"95% CI: ({lo:.3f}, {hi:.3f})")
    print("  (observed 6/8 heads, expect bias ~0.75)")

    print("\n=== Coin bias: single-site MH (10,000 samples) ===")
    mh_bias = infer(coin_bias, method="mh", num_samples=10000, burn_in=2000)
    print(f"Inferred bias: {mh_bias.mean():.3f} +/- {mh_bias.std():.3f}")
    lo, hi = mh_bias.credible_interval(level=0.95)
    print(f"95% CI: ({lo:.3f}, {hi:.3f})")
