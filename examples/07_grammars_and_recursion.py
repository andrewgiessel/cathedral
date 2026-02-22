"""Grammars and Recursion -- Cathedral examples.

Demonstrates probabilistic grammars, stochastic recursion, and
programs generating programs -- core expressiveness of Church/WebPPL.

Church (v1): https://v1.probmods.org/generative-models.html
WebPPL (v2): https://probmods.org/chapters/generative-models.html
"""

from cathedral import (
    Categorical,
    Geometric,
    UniformDraw,
    condition,
    flip,
    infer,
    model,
    sample,
)

# ---------------------------------------------------------------------------
# 1. Probabilistic Context-Free Grammar
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Simple probabilistic grammar")
print("=" * 60)


@model
def pcfg():
    """Generate a sentence from a simple probabilistic CFG.
    S -> NP VP
    NP -> Det N
    VP -> V NP | V
    """

    def det():
        return sample(Categorical(["the", "a"], [0.6, 0.4]))

    def noun():
        return sample(Categorical(["dog", "cat", "mouse"], [0.4, 0.4, 0.2]))

    def verb():
        return sample(Categorical(["chased", "saw", "liked"], [0.5, 0.3, 0.2]))

    def np():
        return f"{det()} {noun()}"

    def vp():
        if flip(0.6):
            return f"{verb()} {np()}"
        return verb()

    def sentence():
        return f"{np()} {vp()}"

    return sentence()


posterior = infer(pcfg, num_samples=2000)
hist = posterior.histogram()
top5 = sorted(hist.items(), key=lambda x: -x[1])[:5]
print("Top 5 most probable sentences:")
for sent, prob in top5:
    print(f"  '{sent}': {prob:.3f}")

# ---------------------------------------------------------------------------
# 2. Arithmetic expressions
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Random arithmetic expressions")
print("=" * 60)


@model
def random_arithmetic():
    """Generate and evaluate a random arithmetic expression tree."""

    def expr(depth=0):
        if depth > 4 or flip(0.4):
            return sample(UniformDraw([1, 2, 3, 4, 5]))

        op = sample(UniformDraw(["+", "-", "*"]))
        left = expr(depth + 1)
        right = expr(depth + 1)

        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        else:
            return left * right

    return expr()


posterior = infer(random_arithmetic, num_samples=2000)
print(f"Mean value: {posterior.mean():.2f}")
print(f"Std: {posterior.std():.2f}")

# ---------------------------------------------------------------------------
# 3. Conditioned generation: find expressions that evaluate to 10
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Conditioned generation (expressions evaluating to 10)")
print("=" * 60)


@model
def expressions_equal_10():
    """Generate an arithmetic expression that evaluates to 10."""

    def expr_str(depth=0):
        if depth > 3 or flip(0.4):
            n = sample(UniformDraw([1, 2, 3, 4, 5]))
            return str(n), n

        op = sample(UniformDraw(["+", "*"]))
        left_s, left_v = expr_str(depth + 1)
        right_s, right_v = expr_str(depth + 1)

        if op == "+":
            return f"({left_s} + {right_s})", left_v + right_v
        else:
            return f"({left_s} * {right_s})", left_v * right_v

    s, v = expr_str()
    condition(v == 10)
    return s


posterior = infer(expressions_equal_10, num_samples=5000)
hist = posterior.histogram()
top10 = sorted(hist.items(), key=lambda x: -x[1])[:10]
print("Most common expressions that equal 10:")
for expr, prob in top10:
    print(f"  {expr} = 10  (prob {prob:.3f})")

# ---------------------------------------------------------------------------
# 4. Recursive list structures
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Random nested lists (S-expressions)")
print("=" * 60)


@model
def random_tree():
    """Generate a random tree structure."""

    def tree(depth=0):
        if depth > 3 or flip(0.4):
            return sample(UniformDraw(["a", "b", "c"]))
        n_children = sample(UniformDraw([1, 2, 3]))
        return [tree(depth + 1) for _ in range(n_children)]

    return str(tree())


posterior = infer(random_tree, num_samples=1000)
top5 = sorted(posterior.histogram().items(), key=lambda x: -x[1])[:5]
print("Most common trees:")
for tree, prob in top5:
    print(f"  {tree}: {prob:.3f}")

# ---------------------------------------------------------------------------
# 5. Geometric distribution via recursion (explicit construction)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. Geometric distribution: recursive vs primitive")
print("=" * 60)


@model
def recursive_geometric():
    def geom(n):
        if flip(0.6):
            return n
        return geom(n + 1)

    return geom(0)


@model
def primitive_geometric():
    return sample(Geometric(0.6))


posterior_rec = infer(recursive_geometric, num_samples=2000)
posterior_prim = infer(primitive_geometric, num_samples=2000)

print(f"Recursive: mean={posterior_rec.mean():.2f}, std={posterior_rec.std():.2f}")
print(f"Primitive: mean={posterior_prim.mean():.2f}, std={posterior_prim.std():.2f}")
print(f"Theoretical mean: {(1 - 0.6) / 0.6:.2f}")

# ---------------------------------------------------------------------------
# 6. Stochastic regex: pattern matching
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. Conditioned string generation")
print("=" * 60)


@model
def string_gen():
    """Generate a string that starts with 'a' and ends with 'b'."""
    chars = ["a", "b", "c"]
    length = sample(UniformDraw([2, 3, 4, 5]))
    s = "".join(sample(UniformDraw(chars)) for _ in range(length))
    condition(s.startswith("a") and s.endswith("b"))
    return s


posterior = infer(string_gen, num_samples=10000)
top10 = sorted(posterior.histogram().items(), key=lambda x: -x[1])[:10]
print("Strings starting with 'a' and ending with 'b':")
for s, prob in top10:
    print(f"  '{s}': {prob:.3f}")
