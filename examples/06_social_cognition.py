"""Social Cognition -- Cathedral examples inspired by ProbMods Chapter 8.

Demonstrates theory of mind: reasoning about other agents' beliefs,
goals, and preferences using nested probabilistic models.

Church (v1): https://v1.probmods.org/social-cognition.html
WebPPL (v2): https://probmods.org/chapters/social-cognition.html
"""

from cathedral import (
    UniformDraw,
    condition,
    flip,
    infer,
    model,
    sample,
)

# ---------------------------------------------------------------------------
# 1. Goal inference: what does the agent want?
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Goal inference")
print("=" * 60)


@model
def goal_inference():
    """An agent chose action A. What was their goal?
    Actions: go-left leads to cookie, go-right leads to apple.
    Agent is rational: they go toward what they want."""
    goal = sample(UniformDraw(["cookie", "apple"]))

    # Rational agent picks the action leading to their goal
    if goal == "cookie":
        action = "go-left" if flip(0.9) else "go-right"
    else:
        action = "go-right" if flip(0.9) else "go-left"

    condition(action == "go-left")
    return goal


posterior = infer(goal_inference, num_samples=2000)
print(f"P(wants cookie | went left) = {posterior.probability(lambda x: x == 'cookie'):.3f}")
print(f"P(wants apple | went left) = {posterior.probability(lambda x: x == 'apple'):.3f}")

# ---------------------------------------------------------------------------
# 2. Preference inference from choices
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Food preferences")
print("=" * 60)


@model
def food_preference():
    """An agent consistently picks pizza. What do they prefer?"""
    preference = sample(UniformDraw(["pizza", "sushi", "salad"]))

    def choose(preference):
        if flip(0.8):
            return preference
        return sample(UniformDraw(["pizza", "sushi", "salad"]))

    condition(choose(preference) == "pizza")
    condition(choose(preference) == "pizza")
    condition(choose(preference) == "pizza")

    return preference


posterior = infer(food_preference, num_samples=5000)
print("P(preference | chose pizza 3 times):")
for food, prob in sorted(posterior.histogram().items(), key=lambda x: -x[1]):
    print(f"  {food}: {prob:.3f}")

# ---------------------------------------------------------------------------
# 3. Belief attribution (Sally-Anne test)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Sally-Anne false belief")
print("=" * 60)


@model
def sally_anne():
    """Sally puts marble in basket, leaves. Anne moves it to box.
    Where does Sally think the marble is?"""
    sally_belief = sample(UniformDraw(["basket", "box"]))

    # Sally saw it go in the basket (before she left)
    condition(sally_belief == "basket")

    # The marble is actually in the box (Anne moved it)
    actual_location = "box"

    return {
        "sally_belief": sally_belief,
        "actual": actual_location,
        "sally_correct": sally_belief == actual_location,
    }


posterior = infer(sally_anne, num_samples=1000)
print(f"Sally believes marble is in: {posterior.histogram('sally_belief')}")
print(f"P(Sally correct) = {posterior.probability('sally_correct'):.3f}")

# ---------------------------------------------------------------------------
# 4. Inferring helpfulness / deception
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Helpful vs deceptive informant")
print("=" * 60)


@model
def informant_trust():
    """An informant says 'the treasure is left'. Are they helpful or deceptive?"""
    informant_helpful = flip(0.7)  # prior: most people are helpful

    treasure = sample(UniformDraw(["left", "right"]))

    # Helpful informant tells truth, deceptive one lies
    if informant_helpful:
        utterance = treasure if flip(0.95) else ("right" if treasure == "left" else "left")
    else:
        utterance = ("right" if treasure == "left" else "left") if flip(0.9) else treasure

    condition(utterance == "left")

    return {
        "helpful": informant_helpful,
        "treasure": treasure,
    }


posterior = infer(informant_trust, num_samples=2000)
print(f"P(helpful | says left) = {posterior.probability('helpful'):.3f}")
print(f"P(treasure left | says left) = {posterior.probability(lambda x: x['treasure'] == 'left'):.3f}")

# ---------------------------------------------------------------------------
# 5. Strategic interaction: coordination game
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. Coordination game (choosing a meeting place)")
print("=" * 60)


@model
def coordination():
    """Two friends try to meet without communicating.
    There's a culturally salient 'focal point' (cafe)."""
    places = ["cafe", "park", "library"]
    salience = [0.5, 0.25, 0.25]

    # Each friend picks based on salience
    from cathedral import Categorical

    alice_choice = sample(Categorical(places, salience), name="alice")
    bob_choice = sample(Categorical(places, salience), name="bob")

    condition(alice_choice == bob_choice)

    return alice_choice


posterior = infer(coordination, num_samples=2000)
print("Meeting place distribution (given they coordinate):")
for place, prob in sorted(posterior.histogram().items(), key=lambda x: -x[1]):
    print(f"  {place}: {prob:.3f}")
