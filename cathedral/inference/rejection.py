"""Rejection sampling inference engine.

Repeatedly executes the model, discarding executions where condition()
fails (raises Rejected). Collected traces are equally weighted.

Best for: discrete models with condition() where the condition is
not too rare.
"""

from __future__ import annotations

from typing import Any, Callable

from cathedral.trace import Rejected, Trace, run_with_trace


def rejection_sample(
    model_fn: Callable,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    num_samples: int = 1000,
    max_attempts: int | None = None,
) -> list[Trace]:
    """Run rejection sampling on a model.

    Args:
        model_fn: The model function to sample from.
        args: Positional arguments to pass to the model.
        kwargs: Keyword arguments to pass to the model.
        num_samples: Number of accepted samples to collect.
        max_attempts: Maximum total attempts before giving up.
            Defaults to num_samples * 1000.

    Returns:
        A list of accepted Traces.

    Raises:
        RuntimeError: If max_attempts is exceeded without collecting enough samples.
    """
    if kwargs is None:
        kwargs = {}
    if max_attempts is None:
        max_attempts = num_samples * 1000

    samples: list[Trace] = []
    attempts = 0

    while len(samples) < num_samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Rejection sampling: collected only {len(samples)}/{num_samples} "
                f"samples after {max_attempts} attempts. "
                f"The condition may be too rare -- consider using method='importance' or method='mcmc'."
            )
        attempts += 1
        try:
            trace = run_with_trace(model_fn, args=args, kwargs=kwargs)
            samples.append(trace)
        except Rejected:
            continue

    return samples
