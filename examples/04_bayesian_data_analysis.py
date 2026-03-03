"""Bayesian Data Analysis -- Cathedral examples inspired by ProbMods Chapter 7.

Demonstrates parameter estimation, model comparison, and posterior predictive
checking using observe() and importance sampling.

Church (v1): https://v1.probmods.org/bayesian-data-analysis.html
WebPPL (v2): https://probmods.org/chapters/bayesian-data-analysis.html
"""

from cathedral import (
    Bernoulli,
    Beta,
    HalfNormal,
    Normal,
    flip,
    infer,
    model,
    observe,
    sample,
)

# ---------------------------------------------------------------------------
# 1. Estimating a proportion (beta-binomial)
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Estimating a coin's bias from data")
print("=" * 60)


@model
def coin_bias_estimation(observations):
    """Estimate a coin's bias from observed flips."""
    p = sample(Beta(1, 1), name="p")
    for outcome in observations:
        observe(Bernoulli(p), outcome)
    return p


data = [True, True, True, True, True, True, True, False, False, False]
posterior = infer(coin_bias_estimation, data, method="importance", num_samples=2000)
print(f"Posterior E[p] = {posterior.mean():.3f} (expected ~0.7)")
print(f"Posterior std[p] = {posterior.std():.3f}")
ci = posterior.credible_interval(0.95)
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

# ---------------------------------------------------------------------------
# 2. Estimating a Gaussian mean and variance
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Estimating Gaussian parameters from data")
print("=" * 60)

@model
def estimate_gaussian(data):
    """Estimate the mean and variance of a Gaussian from data."""
    mu = sample(Normal(0, 10), name="mu")
    sigma = sample(HalfNormal(5), name="sigma")
    for x in data:
        observe(Normal(mu, sigma), x)
    return {"mu": mu, "sigma": sigma}


data = [4.2, 3.8, 5.1, 4.6, 3.9, 4.4, 5.0, 4.7, 4.1, 4.3]
posterior = infer(estimate_gaussian, data, method="importance", num_samples=2000)
print(f"E[mu] = {posterior.mean('mu'):.3f} (true ≈ 4.41)")
print(f"E[sigma] = {posterior.mean('sigma'):.3f}")

# ---------------------------------------------------------------------------
# 3. Model comparison: one group vs two groups
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Model comparison: one vs two groups")
print("=" * 60)

@model
def one_vs_two_groups(group_a, group_b):
    """Model comparison: are two groups drawn from one distribution or two?"""
    two_groups = flip(0.5)

    if two_groups:
        mu_a = sample(Normal(0, 5), name="mu_a")
        mu_b = sample(Normal(0, 5), name="mu_b")
    else:
        mu_shared = sample(Normal(0, 5), name="mu_shared")
        mu_a = mu_shared
        mu_b = mu_shared

    for x in group_a:
        observe(Normal(mu_a, 0.5), x)
    for x in group_b:
        observe(Normal(mu_b, 0.5), x)

    return two_groups


# Clearly separated groups
posterior = infer(
    one_vs_two_groups,
    [2.1, 2.3, 1.9, 2.0, 2.2],
    [4.8, 5.1, 4.9, 5.0, 5.2],
    method="importance", num_samples=2000,
)
print(f"P(two groups) = {posterior.probability():.3f}")

# ---------------------------------------------------------------------------
# 4. Bayesian linear regression
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Bayesian linear regression")
print("=" * 60)

@model
def bayesian_regression(xs, ys):
    """Bayesian linear regression: infer slope, intercept, and noise from data."""
    slope = sample(Normal(0, 5), name="slope")
    intercept = sample(Normal(0, 5), name="intercept")
    noise = sample(HalfNormal(2), name="noise")

    for x, y in zip(xs, ys, strict=False):
        observe(Normal(slope * x + intercept, noise), y)

    return {"slope": slope, "intercept": intercept, "noise": noise}


xs = [1, 2, 3, 4, 5, 6, 7, 8]
ys = [2.1, 4.3, 5.8, 8.2, 9.9, 12.1, 14.0, 16.1]  # y ≈ 2x + 0
posterior = infer(bayesian_regression, xs, ys, method="importance", num_samples=2000)
print(f"Slope: {posterior.mean('slope'):.3f} (true ≈ 2.0)")
print(f"Intercept: {posterior.mean('intercept'):.3f} (true ≈ 0.0)")
print(f"Noise: {posterior.mean('noise'):.3f}")

# ---------------------------------------------------------------------------
# 5. Posterior predictive: what data would the model generate?
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. Posterior predictive")
print("=" * 60)


@model
def posterior_predictive(observed):
    """Learn the mean from data, then predict a new observation."""
    mu = sample(Normal(0, 10), name="mu")
    sigma = sample(HalfNormal(5), name="sigma")

    for x in observed:
        observe(Normal(mu, sigma), x)

    new_obs = sample(Normal(mu, sigma), name="prediction")
    return new_obs


posterior = infer(posterior_predictive, [10.1, 9.8, 10.3, 9.9, 10.0], method="importance", num_samples=2000)
print(f"Predicted new observation: {posterior.mean():.3f}")
print(f"Prediction uncertainty (std): {posterior.std():.3f}")
ci = posterior.credible_interval(0.95)
print(f"95% prediction interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
