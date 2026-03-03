"""Conditioning -- Cathedral examples inspired by ProbMods Chapter 3.

Demonstrates conditioning, Bayesian reasoning, and the interaction
between generative models and observations.

Church (v1): https://v1.probmods.org/conditioning.html
WebPPL (v2): https://probmods.org/chapters/conditioning.html
"""

from cathedral import (
    Normal,
    Uniform,
    UniformDraw,
    condition,
    flip,
    infer,
    model,
    observe,
    sample,
)

# ---------------------------------------------------------------------------
# 1. Simple conditioning: what's behind the curtain?
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Behind the curtain (simple conditioning)")
print("=" * 60)


@model
def behind_curtain():
    """There's either a blue or green marble behind the curtain.
    We observe it's not blue -- what is it?"""
    marble = sample(UniformDraw(["blue", "green"]))
    condition(marble != "blue")
    return marble


posterior = infer(behind_curtain, num_samples=1000)
print(f"P(green | not blue) = {posterior.probability(lambda x: x == 'green'):.3f}")

# ---------------------------------------------------------------------------
# 2. Fair vs biased coin (classic Bayesian reasoning)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Fair vs biased coin")
print("=" * 60)


@model
def fair_or_biased():
    """Is the coin fair? We see 3 heads in a row."""
    is_fair = flip(0.5)
    p = 0.5 if is_fair else 0.9

    condition(flip(p))  # heads
    condition(flip(p))  # heads
    condition(flip(p))  # heads

    return is_fair


posterior = infer(fair_or_biased, num_samples=2000)
print(f"P(fair | HHH) = {posterior.probability():.3f}")

# ---------------------------------------------------------------------------
# 3. Causal vs diagnostic reasoning
#    One generative model, multiple queries via condition= (Church-style).
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Causal vs diagnostic reasoning")
print("=" * 60)


@model
def lawn():
    """Pure generative model: rain, sprinkler, and wet lawn."""
    rain = flip(0.3)
    sprinkler = flip(0.5)
    wet = rain or sprinkler
    return {"rain": rain, "sprinkler": sprinkler, "wet": wet}


# Same model, different questions — no rewrite needed
posterior = infer(lawn, num_samples=1000, condition=lambda r: r["rain"])
print(f"P(wet lawn | rain) = {posterior.probability('wet'):.3f}")

posterior = infer(lawn, num_samples=2000, condition=lambda r: r["wet"])
print(f"P(rain | wet lawn) = {posterior.probability('rain'):.3f}")

# ---------------------------------------------------------------------------
# 4. Explaining away
#    Same lawn model, increasingly specific conditions.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Explaining away")
print("=" * 60)

posterior = infer(lawn, num_samples=2000, condition=lambda r: r["wet"])
print(f"P(rain | wet) = {posterior.probability('rain'):.3f}")

posterior = infer(lawn, num_samples=2000, condition=lambda r: r["wet"] and r["sprinkler"])
print(f"P(rain | wet, sprinkler on) = {posterior.probability('rain'):.3f}")

# ---------------------------------------------------------------------------
# 5. Learning a continuous parameter with observe
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. Learning a mean from data (importance sampling)")
print("=" * 60)


@model
def learn_mean(data):
    """Infer the mean of a Gaussian from observed data."""
    mu = sample(Normal(0, 5), name="mu")
    for x in data:
        observe(Normal(mu, 0.5), x)
    return mu


posterior = infer(learn_mean, [2.1, 1.8, 2.3, 1.9, 2.0], method="importance", num_samples=2000)
print(f"Posterior mean of mu: {posterior.mean():.3f}")
print(f"Posterior std of mu: {posterior.std():.3f}")
print(f"95% credible interval: {posterior.credible_interval(0.95)}")

# ---------------------------------------------------------------------------
# 6. Number game (hypothesis space)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. Simple number game")
print("=" * 60)


@model
def number_game():
    """A number between 1 and 10 was chosen from some concept.
    Concepts: 'small' (1-3), 'medium' (4-7), 'large' (8-10), 'even', 'all'.
    We observe the number 4."""
    concept = sample(UniformDraw(["small", "medium", "large", "even", "all"]))

    sets = {
        "small": [1, 2, 3],
        "medium": [4, 5, 6, 7],
        "large": [8, 9, 10],
        "even": [2, 4, 6, 8, 10],
        "all": list(range(1, 11)),
    }
    members = sets[concept]
    condition(4 in members)

    return concept


posterior = infer(number_game, num_samples=2000)
print("Concept distribution given observing 4:")
print(f"  {posterior.histogram()}")

# ---------------------------------------------------------------------------
# 7. Conditioning on an approximate constraint (soft conditioning)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("7. Soft conditioning (approximately equal)")
print("=" * 60)


@model
def soft_conditioning():
    """Infer x given that x + y is approximately 10."""
    x = sample(Uniform(0, 10), name="x")
    y = sample(Uniform(0, 10), name="y")
    observe(Normal(10, 0.5), x + y)
    return x


posterior = infer(soft_conditioning, method="importance", num_samples=2000)
print(f"E[x | x+y≈10] = {posterior.mean():.2f}")
print(f"Std[x | x+y≈10] = {posterior.std():.2f}")
