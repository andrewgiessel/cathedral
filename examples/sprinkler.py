"""Sprinkler example from Goodman et al. 2008.

Classic Bayesian reasoning: given that the grass is wet, did it rain
or was the sprinkler on? Demonstrates condition() for hard conditioning
and rejection sampling for inference.
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
    posterior = infer(sprinkler, method="rejection", num_samples=10000)

    print("P(rain | grass is wet):")
    print(f"  mean = {posterior.probability('rain'):.3f}")
    print(f"  histogram = {posterior.histogram('rain')}")

    print("\nP(sprinkler | grass is wet):")
    print(f"  mean = {posterior.probability('sprinkler'):.3f}")
    print(f"  histogram = {posterior.histogram('sprinkler')}")
