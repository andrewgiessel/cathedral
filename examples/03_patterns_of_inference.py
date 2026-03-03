"""Patterns of Inference -- Cathedral examples inspired by ProbMods Chapter 4.

Demonstrates classic inference patterns: Bayesian updating with more data,
learning from positive/negative evidence, and inference about inference.

Church (v1): https://v1.probmods.org/patterns-of-inference.html
WebPPL (v2): https://probmods.org/chapters/conditional-dependence.html
WebPPL (v2): https://probmods.org/chapters/learning-as-conditional-inference.html
"""

import math

from cathedral import (
    Bernoulli,
    Beta,
    Normal,
    UniformDraw,
    condition,
    factor,
    flip,
    infer,
    model,
    observe,
    sample,
)

# ---------------------------------------------------------------------------
# 1. Bayesian updating: learning a coin's bias with more evidence
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Bayesian updating: coin bias with increasing evidence")
print("=" * 60)


@model
def learn_bias(observations):
    """Infer coin bias from a sequence of boolean observations."""
    p = sample(Beta(1, 1), name="p")
    for obs in observations:
        observe(Bernoulli(p), obs)
    return p


for n_heads, n_total in [(1, 1), (3, 3), (5, 5), (7, 10), (15, 20)]:
    data = [True] * n_heads + [False] * (n_total - n_heads)
    posterior = infer(learn_bias, data, method="importance", num_samples=2000)
    print(f"  {n_heads}/{n_total} heads -> E[p] = {posterior.mean():.3f}, std = {posterior.std():.3f}")

# ---------------------------------------------------------------------------
# 2. Learning about bags of marbles
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Bags of marbles")
print("=" * 60)


@model
def bag_of_marbles(draws):
    """Which bag did the marble come from?
    Bag A: 80% blue, 20% red.  Bag B: 20% blue, 80% red.
    Draws is a list of observed colors."""
    bag = sample(UniformDraw(["A", "B"]))
    p_blue = 0.8 if bag == "A" else 0.2
    for draw in draws:
        condition(flip(p_blue) == (draw == "blue"))
    return bag


# Same model, different amounts of evidence
posterior = infer(bag_of_marbles, ["blue"], num_samples=2000)
print(f"P(bag A | blue) = {posterior.probability(lambda x: x == 'A'):.3f}")

posterior = infer(bag_of_marbles, ["blue", "blue"], num_samples=2000)
print(f"P(bag A | blue, blue) = {posterior.probability(lambda x: x == 'A'):.3f}")

posterior = infer(bag_of_marbles, ["blue", "red"], num_samples=2000)
print(f"P(bag A | blue, red) = {posterior.probability(lambda x: x == 'A'):.3f}")

# ---------------------------------------------------------------------------
# 3. Positive vs negative evidence
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Positive and negative evidence")
print("=" * 60)


@model
def disease_test(n_positive_tests):
    """A disease with 1% base rate. Test is 90% sensitive, 5% false positive.
    Uses factor() to score by the probability of each positive test result."""
    disease = flip(0.01)
    p_positive = 0.9 if disease else 0.05
    for _ in range(n_positive_tests):
        factor(math.log(p_positive))
    return disease


# Same model, different amounts of evidence
posterior = infer(disease_test, 1, method="importance", num_samples=2000)
print(f"P(disease | 1 positive test) = {posterior.probability():.3f}")

posterior = infer(disease_test, 2, method="importance", num_samples=2000)
print(f"P(disease | 2 positive tests) = {posterior.probability():.3f}")

# ---------------------------------------------------------------------------
# 4. The Monty Hall problem
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Monty Hall problem")
print("=" * 60)


@model
def monty_hall_stay():
    """P(win | staying with original choice)."""
    prize = sample(UniformDraw([1, 2, 3]))
    pick = 1  # player always picks door 1

    # Monty opens a door that's not the prize and not the pick
    open_options = [d for d in [1, 2, 3] if d != prize and d != pick]
    monty_opens = sample(UniformDraw(open_options))

    return prize == pick  # did we win by staying?


posterior = infer(monty_hall_stay, num_samples=2000)
print(f"P(win | stay) = {posterior.probability():.3f}")


@model
def monty_hall_switch():
    """P(win | switching)."""
    prize = sample(UniformDraw([1, 2, 3]))
    pick = 1

    open_options = [d for d in [1, 2, 3] if d != prize and d != pick]
    monty_opens = sample(UniformDraw(open_options))

    switch_to = [d for d in [1, 2, 3] if d != pick and d != monty_opens][0]
    return prize == switch_to  # did we win by switching?


posterior = infer(monty_hall_switch, num_samples=2000)
print(f"P(win | switch) = {posterior.probability():.3f}")

# ---------------------------------------------------------------------------
# 5. Learning with different amounts of data
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. Strength of evidence (more data -> less uncertainty)")
print("=" * 60)


@model
def learn_mean(data):
    """Infer the mean of a Gaussian from observed data."""
    mu = sample(Normal(0, 5), name="mu")
    for x in data:
        observe(Normal(mu, 1.0), x)
    return mu


# Same model, increasing data — shows how uncertainty shrinks
for n_obs in [1, 5, 20, 50]:
    data = [3.0] * n_obs
    posterior = infer(learn_mean, data, method="importance", num_samples=1000)
    ci = posterior.credible_interval(0.95)
    print(f"  {n_obs:2d} observations of 3.0 -> E[mu] = {posterior.mean():.3f}, 95% CI = [{ci[0]:.2f}, {ci[1]:.2f}]")

# ---------------------------------------------------------------------------
# 6. Occam's razor (simpler hypotheses preferred)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. Occam's razor")
print("=" * 60)


@model
def occam(observed_value):
    """Two hypotheses:
    - 'narrow': N(5, 0.5) -- strong prediction near 5
    - 'wide': N(5, 5.0)   -- weak prediction near 5
    Occam's razor: narrow is preferred when data is near 5."""
    hypothesis = sample(UniformDraw(["narrow", "wide"]))
    sigma = 0.5 if hypothesis == "narrow" else 5.0
    observe(Normal(5, sigma), observed_value)
    return hypothesis


posterior = infer(occam, 5.0, method="importance", num_samples=2000)
print(f"P(narrow | data=5.0) = {posterior.probability(lambda x: x == 'narrow'):.3f}")
print(f"P(wide | data=5.0) = {posterior.probability(lambda x: x == 'wide'):.3f}")

# Far from 5, the wide hypothesis is better
posterior = infer(occam, 15.0, method="importance", num_samples=2000)
print(f"P(narrow | data=15.0) = {posterior.probability(lambda x: x == 'narrow'):.3f}")
print(f"P(wide | data=15.0) = {posterior.probability(lambda x: x == 'wide'):.3f}")
