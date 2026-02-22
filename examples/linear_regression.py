"""Bayesian linear regression example.

Infer the slope and intercept of a line from noisy observations.
Demonstrates observe() for soft conditioning, and compares importance
sampling with single-site MH.
"""

import numpy as np

from cathedral import HalfNormal, Normal, infer, model, observe, sample


@model
def line_model(xs, ys):
    slope = sample(Normal(0, 5), name="slope")
    intercept = sample(Normal(0, 5), name="intercept")
    noise = sample(HalfNormal(2), name="noise")

    for x, y in zip(xs, ys, strict=False):
        observe(Normal(slope * x + intercept, noise), y)

    return {"slope": slope, "intercept": intercept, "noise": noise}


def _print_results(posterior, label):
    print(f"\n=== {label} ===")
    for param in ["slope", "intercept", "noise"]:
        mean = posterior.mean(param)
        std = posterior.std(param)
        lo, hi = posterior.credible_interval(key=param)
        print(f"  {param:>12s}: {mean:.3f} +/- {std:.3f}  (95% CI: {lo:.3f} to {hi:.3f})")


if __name__ == "__main__":
    np.random.seed(42)

    true_slope = 2.0
    true_intercept = 1.0
    true_noise = 0.5
    xs = np.linspace(0, 5, 20)
    ys = true_slope * xs + true_intercept + np.random.normal(0, true_noise, len(xs))

    print(f"True parameters: slope={true_slope}, intercept={true_intercept}, noise={true_noise}")
    print(f"Data: {len(xs)} points")

    _print_results(
        infer(line_model, xs, ys, method="importance", num_samples=10000),
        "Importance sampling (10,000 samples)",
    )

    _print_results(
        infer(line_model, xs, ys, method="mh", num_samples=10000, burn_in=5000),
        "Single-site MH (10,000 samples, 5,000 burn-in)",
    )
