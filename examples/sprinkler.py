"""Sprinkler example from Goodman et al. 2008.

Classic Bayesian reasoning: given that the grass is wet, did it rain
or was the sprinkler on?  Demonstrates all four inference methods
and shows that enumeration gives the exact answer.
"""

from cathedral import condition, flip, infer, model


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


if __name__ == "__main__":
    # Exact answer via enumeration
    print("=== Exact enumeration ===")
    exact = infer(sprinkler, method="enumerate")
    print(f"P(rain | wet)      = {exact.probability('rain'):.6f}")
    print(f"P(sprinkler | wet) = {exact.probability('sprinkler'):.6f}")

    # Rejection sampling
    print("\n=== Rejection sampling (10,000 samples) ===")
    rej = infer(sprinkler, method="rejection", num_samples=10000)
    print(f"P(rain | wet)      = {rej.probability('rain'):.4f}")
    print(f"P(sprinkler | wet) = {rej.probability('sprinkler'):.4f}")

    # Single-site MH
    print("\n=== Single-site MH (10,000 samples, 2,000 burn-in) ===")
    mh = infer(sprinkler, method="mh", num_samples=10000, burn_in=2000)
    print(f"P(rain | wet)      = {mh.probability('rain'):.4f}")
    print(f"P(sprinkler | wet) = {mh.probability('sprinkler'):.4f}")

    # Joint distribution from enumeration
    print("\n=== Full joint posterior (exact) ===")
    for outcome, prob in exact.histogram().items():
        rain_str = "rain" if outcome["rain"] else "no rain"
        spr_str = "sprinkler" if outcome["sprinkler"] else "no sprinkler"
        print(f"  {rain_str:>12s}, {spr_str:>14s}: {prob:.6f}")
