"""Generative Models -- Cathedral examples inspired by ProbMods Chapter 2.

Demonstrates the core building blocks: random primitives, composition,
persistent randomness via mem, and stochastic recursion.

Church (v1): https://v1.probmods.org/generative-models.html
WebPPL (v2): https://probmods.org/chapters/generative-models.html
"""

from cathedral import (
    Normal,
    UniformDraw,
    condition,
    flip,
    infer,
    mem,
    model,
    sample,
)

# ---------------------------------------------------------------------------
# 1. Flipping coins
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Fair and unfair coins")
print("=" * 60)


@model
def fair_coin():
    return flip(0.5)


posterior = infer(fair_coin, num_samples=1000)
print(f"P(heads) = {posterior.probability():.3f}")  # ~0.5


@model
def trick_coin():
    return flip(0.95)


posterior = infer(trick_coin, num_samples=1000)
print(f"P(trick heads) = {posterior.probability():.3f}")  # ~0.95

# ---------------------------------------------------------------------------
# 2. Composing random choices
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Random combinations (eye/hair color pairs)")
print("=" * 60)


@model
def random_appearance():
    eye = sample(UniformDraw(["blue", "green", "brown"]))
    hair = sample(UniformDraw(["blonde", "brown", "black", "red"]))
    return f"{eye} eyes, {hair} hair"


posterior = infer(random_appearance, num_samples=1000)
print(f"Most common: {max(posterior.histogram().items(), key=lambda x: x[1])}")

# ---------------------------------------------------------------------------
# 3. Persistent randomness with mem
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Persistent properties with mem")
print("=" * 60)


@model
def persistent_eye_color():
    """Each person has a fixed eye color, determined once."""
    eye_color = mem(lambda person: sample(UniformDraw(["blue", "green", "brown"])))

    bob1 = eye_color("bob")
    bob2 = eye_color("bob")
    alice = eye_color("alice")

    return {
        "bob_consistent": bob1 == bob2,
        "bob": bob1,
        "alice": alice,
    }


posterior = infer(persistent_eye_color, num_samples=1000)
print(f"Bob always consistent: {posterior.probability('bob_consistent'):.3f}")
print(f"Bob's eye color distribution: {posterior.histogram('bob')}")

# ---------------------------------------------------------------------------
# 4. Stochastic recursion: geometric distribution from scratch
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Stochastic recursion")
print("=" * 60)


@model
def geometric_from_flips():
    """Build a geometric distribution from recursive coin flips."""

    def geom(n):
        if flip(0.5):
            return n
        return geom(n + 1)

    return geom(0)


posterior = infer(geometric_from_flips, num_samples=2000)
hist = posterior.histogram()
sorted_vals = sorted(hist.items(), key=lambda x: x[0])[:8]
print("Geometric from flips:")
for val, prob in sorted_vals:
    bar = "#" * int(prob * 80)
    print(f"  {val}: {prob:.3f} {bar}")

# ---------------------------------------------------------------------------
# 5. Random lists with stochastic recursion
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. Random lists (variable length)")
print("=" * 60)


@model
def random_list():
    """Generate a random-length list of 'H' and 'T'."""

    def build():
        if flip(0.3):
            return []
        return ["H" if flip() else "T"] + build()

    result = build()
    return len(result)


posterior = infer(random_list, num_samples=2000)
print(f"Mean length: {posterior.mean():.2f}")
hist = posterior.histogram()
sorted_vals = sorted(hist.items(), key=lambda x: x[0])[:8]
for val, prob in sorted_vals:
    bar = "#" * int(prob * 80)
    print(f"  len={val}: {prob:.3f} {bar}")

# ---------------------------------------------------------------------------
# 6. Medical diagnosis (causal model)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. Medical diagnosis")
print("=" * 60)


@model
def medical_diagnosis():
    """Classic causal Bayes net: cold -> cough/fever, lung disease -> cough."""
    has_cold = flip(0.2)
    has_lung_disease = flip(0.001)

    cough = (has_cold and flip(0.5)) or (has_lung_disease and flip(0.3)) or flip(0.01)
    fever = (has_cold and flip(0.3)) or flip(0.01)
    chest_pain = (has_lung_disease and flip(0.5)) or flip(0.01)

    condition(cough)

    return {"cold": has_cold, "lung_disease": has_lung_disease}


posterior = infer(medical_diagnosis, num_samples=5000)
print(f"P(cold | cough) = {posterior.probability('cold'):.3f}")
print(f"P(lung disease | cough) = {posterior.probability('lung_disease'):.4f}")

# ---------------------------------------------------------------------------
# 7. Tug of war (with mem for persistent strength)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("7. Tug of war")
print("=" * 60)


@model
def tug_of_war():
    """Each person has a persistent strength; laziness varies per match."""
    strength = mem(lambda person: sample(Normal(0, 1)))
    lazy = lambda person: flip(0.25)

    def pulling(person):
        return strength(person) / 2 if lazy(person) else strength(person)

    def total_pulling(team):
        return sum(pulling(p) for p in team)

    def winner(team1, team2):
        return "team1" if total_pulling(team1) > total_pulling(team2) else "team2"

    condition(winner(["alice", "sue"], ["bob", "tom"]) == "team1")

    return {"alice_stronger_than_bob": strength("alice") > strength("bob")}


posterior = infer(tug_of_war, method="rejection", num_samples=1000)
print(f"P(Alice stronger than Bob | Alice's team wins) = {posterior.probability('alice_stronger_than_bob'):.3f}")

# ---------------------------------------------------------------------------
# 8. Mixture of Gaussians (using flip to choose component)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("8. Gaussian mixture model")
print("=" * 60)


@model
def gaussian_mixture():
    """Two-component Gaussian mixture."""
    if flip(0.3):
        return sample(Normal(-2, 0.5))
    else:
        return sample(Normal(3, 1.0))


posterior = infer(gaussian_mixture, num_samples=2000)
print(f"Mean: {posterior.mean():.2f}")
print(f"Std: {posterior.std():.2f}")
