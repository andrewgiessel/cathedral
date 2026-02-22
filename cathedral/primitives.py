"""Probabilistic primitives for Cathedral.

These are the core building blocks for writing probabilistic models.
Each primitive is trace-aware: when called inside an inference context,
it records choices and scores. When called outside inference, it just
samples directly.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np

from cathedral.distributions import Bernoulli, Distribution
from cathedral.trace import NeedsEnumeration, Rejected, get_trace_context


def sample(dist: Distribution, *, name: str | None = None) -> Any:
    """Sample a value from a distribution.

    When running under inference, records the choice in the trace.
    When running standalone, just draws a sample.

    Args:
        dist: The distribution to sample from.
        name: Optional name for the sample site. Auto-generated if not provided.

    Returns:
        A sampled value.
    """
    ctx = get_trace_context()

    if ctx is None:
        return dist.sample()

    address = name if name is not None else ctx.fresh_address(type(dist).__name__)

    if ctx.has_intervention(address):
        value = ctx.get_intervention(address)
    elif ctx.enumerate_mode:
        if dist.support() is not None:
            raise NeedsEnumeration(address, dist)
        raise RuntimeError(
            f"Enumeration encountered a distribution with no finite support "
            f"at address '{address}': {dist!r}. "
            f"Enumeration only works with discrete distributions "
            f"(Bernoulli, Categorical, UniformDraw)."
        )
    else:
        value = dist.sample()

    ctx.record_choice(address, dist, value)
    return value


def flip(p: float = 0.5, *, name: str | None = None) -> bool:
    """Flip a (possibly weighted) coin. Sugar for sample(Bernoulli(p)).

    Args:
        p: Probability of True.
        name: Optional name for the sample site.

    Returns:
        True with probability p, False otherwise.
    """
    return sample(Bernoulli(p), name=name)


def condition(predicate: bool) -> None:
    """Hard conditioning: reject this execution if predicate is False.

    In rejection sampling, this causes the current execution to be
    discarded and retried. In MCMC, it contributes -inf to the log-score.

    Args:
        predicate: The condition that must be True.
    """
    if not predicate:
        ctx = get_trace_context()
        if ctx is not None:
            ctx.add_score(float("-inf"))
        raise Rejected()


def observe(dist: Distribution, value: Any) -> None:
    """Soft conditioning: observe that a value was drawn from a distribution.

    Adds the log-probability of the value under the distribution to the
    trace's score. This is the primary mechanism for fitting models to data.

    Args:
        dist: The distribution the value was (hypothetically) drawn from.
        value: The observed value.
    """
    ctx = get_trace_context()
    if ctx is not None:
        ctx.add_score(dist.log_prob(value))


def factor(score: float) -> None:
    """Add an arbitrary log-score to the trace.

    This is the most general form of conditioning. condition() and observe()
    are both implemented in terms of factor() conceptually.

    Args:
        score: Log-probability to add.
    """
    ctx = get_trace_context()
    if ctx is not None:
        ctx.add_score(score)


def _make_hashable(args: tuple) -> tuple:
    """Convert function arguments to a hashable key for memo caches."""
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.append(("__list__", _make_hashable(tuple(arg))))
        elif isinstance(arg, dict):
            result.append(("__dict__", tuple(sorted(arg.items()))))
        elif isinstance(arg, np.ndarray):
            result.append(("__ndarray__", arg.tobytes(), arg.shape))
        else:
            result.append(arg)
    return tuple(result)


def mem(fn: Callable) -> Callable:
    """Stochastic memoization.

    Returns a version of fn that caches its return value for each unique
    set of arguments. Within a single model execution (trace), calling
    the memoized function with the same arguments always returns the same
    value -- even if the original function is stochastic.

    This is the key primitive for "persistent randomness" in Church:
    random properties that are determined once and then fixed.

    Example:
        eye_color = mem(lambda person: sample(Categorical(
            ["blue", "green", "brown"], [1/3, 1/3, 1/3])))
        eye_color("bob")   # samples once
        eye_color("bob")   # returns same value
        eye_color("alice") # samples independently

    Args:
        fn: The function to memoize.

    Returns:
        A memoized version of fn with per-trace caching.
    """
    standalone_cache: dict = {}
    func_id = id(fn)

    @functools.wraps(fn)
    def memoized(*args):
        ctx = get_trace_context()
        key = _make_hashable(args)

        if ctx is not None:
            cache = ctx.get_memo_cache(func_id)
        else:
            cache = standalone_cache

        if key not in cache:
            cache[key] = fn(*args)
        return cache[key]

    memoized._is_memoized = True
    memoized._original_fn = fn
    return memoized


def DPmem(alpha: float, fn: Callable) -> Callable:
    """Dirichlet Process stochastic memoizer.

    Like mem, but instead of always returning the cached value, it
    stochastically decides whether to reuse a previous value or sample
    a new one, using the Chinese Restaurant Process.

    When alpha=0, behaves like mem (always reuse).
    When alpha=inf, behaves like the original function (always resample).

    Args:
        alpha: Concentration parameter. Higher = more likely to sample new values.
        fn: The function to memoize.

    Returns:
        A DP-memoized version of fn.
    """
    func_id = id(fn)
    standalone_cache: dict = {}

    def dp_memoized(*args):
        ctx = get_trace_context()
        key = _make_hashable(args)

        if ctx is not None:
            cache = ctx.get_memo_cache(func_id)
        else:
            cache = standalone_cache

        if key not in cache:
            cache[key] = {"values": [], "counts": []}

        table = cache[key]

        total_count = sum(table["counts"]) if table["counts"] else 0
        prob_new = alpha / (alpha + total_count)

        if not table["values"] or np.random.random() < prob_new:
            result = fn(*args)
            table["values"].append(result)
            table["counts"].append(1)
        else:
            probs = np.array(table["counts"], dtype=float)
            probs /= probs.sum()
            idx = np.random.choice(len(table["values"]), p=probs)
            result = table["values"][idx]
            table["counts"][idx] += 1

        return result

    dp_memoized._is_dp_memoized = True
    dp_memoized._alpha = alpha
    dp_memoized._original_fn = fn
    return dp_memoized
