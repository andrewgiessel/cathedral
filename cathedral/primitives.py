"""Probabilistic primitives for Cathedral.

These are the core building blocks for writing probabilistic models.
Each primitive is trace-aware: when called inside an inference context,
it records choices and scores. When called outside inference, it just
samples directly.
"""

from __future__ import annotations

from typing import Any

from cathedral.distributions import Bernoulli, Distribution
from cathedral.trace import Rejected, get_trace_context


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
