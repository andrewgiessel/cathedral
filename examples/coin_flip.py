"""Coin flip inference example.

Given that a coin landed heads, what's the probability it was a fair coin?
Demonstrates Church-style query separation: the generative model describes
the coin-flipping process, and the condition is supplied externally.
"""

from cathedral import flip, infer, model, observe, sample
from cathedral.distributions import Bernoulli, Beta


@model
def fair_coin():
    """Generative model: is the coin fair, and what did it land?"""
    is_fair = flip(0.5)
    result = flip(0.5 if is_fair else 0.9)
    return {"fair": is_fair, "result": result}


@model
def coin_bias(observations):
    """Infer coin bias from a sequence of observations."""
    bias = sample(Beta(1, 1), name="bias")
    for obs in observations:
        observe(Bernoulli(bias), obs)
    return bias


if __name__ == "__main__":
    heads = lambda r: r["result"]

    # Exact answer for the discrete model
    print("=== Fair coin: exact enumeration ===")
    exact = infer(fair_coin, method="enumerate", condition=heads)
    p_fair = exact.probability("fair")
    print(f"P(fair | heads) = {p_fair:.6f}")
    print(f"  = 0.25 / 0.70 = {0.25 / 0.70:.6f} (analytic)")

    # Compare with sampling methods
    print("\n=== Fair coin: rejection sampling (10,000 samples) ===")
    rej = infer(fair_coin, method="rejection", num_samples=10000, condition=heads)
    print(f"P(fair | heads) = {rej.probability('fair'):.4f}")

    print("\n=== Fair coin: single-site MH (10,000 samples) ===")
    mh = infer(fair_coin, method="mh", num_samples=10000, burn_in=2000, condition=heads)
    print(f"P(fair | heads) = {mh.probability('fair'):.4f}")

    # Same model, different question: what if we saw tails?
    print("\n=== Fair coin: what if we saw tails? (exact) ===")
    tails = lambda r: not r["result"]
    exact_tails = infer(fair_coin, method="enumerate", condition=tails)
    print(f"P(fair | tails) = {exact_tails.probability('fair'):.6f}")

    # Continuous bias estimation — same model, different data
    flips = [True, True, False, True, True, True, False, True]

    print("\n=== Coin bias: importance sampling (10,000 samples) ===")
    imp = infer(coin_bias, flips, method="importance", num_samples=10000)
    print(f"Inferred bias: {imp.mean():.3f} +/- {imp.std():.3f}")
    lo, hi = imp.credible_interval(level=0.95)
    print(f"95% CI: ({lo:.3f}, {hi:.3f})")
    print("  (observed 6/8 heads, expect bias ~0.75)")

    print("\n=== Coin bias: single-site MH (10,000 samples) ===")
    mh_bias = infer(coin_bias, flips, method="mh", num_samples=10000, burn_in=2000)
    print(f"Inferred bias: {mh_bias.mean():.3f} +/- {mh_bias.std():.3f}")
    lo, hi = mh_bias.credible_interval(level=0.95)
    print(f"95% CI: ({lo:.3f}, {hi:.3f})")
