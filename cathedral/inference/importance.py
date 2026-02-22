"""Likelihood-weighted importance sampling inference engine.

Executes the model many times, collecting all traces (no rejection).
Each trace is weighted by exp(log_score) from observe() and factor() calls.
Optionally resamples to produce an unweighted set of traces.

Best for: models with continuous observations via observe() where
rejection sampling would be inefficient.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from cathedral.trace import Rejected, Trace, run_with_trace


def importance_sample(
    model_fn: Callable,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    num_samples: int = 1000,
    resample: bool = True,
) -> list[Trace]:
    """Run likelihood-weighted importance sampling on a model.

    Forward-samples from the prior, then weights each sample by the
    likelihood (from observe/factor calls). Optionally resamples
    proportional to weights to produce an unweighted sample set.

    Args:
        model_fn: The model function to sample from.
        args: Positional arguments to pass to the model.
        kwargs: Keyword arguments to pass to the model.
        num_samples: Number of samples to draw from the prior.
        resample: If True, resample proportional to weights to produce
            an unweighted sample set of the same size.

    Returns:
        A list of Traces (resampled if resample=True).
    """
    if kwargs is None:
        kwargs = {}

    traces: list[Trace] = []
    log_weights: list[float] = []

    for _ in range(num_samples):
        try:
            trace = run_with_trace(model_fn, args=args, kwargs=kwargs)
            traces.append(trace)
            log_weights.append(trace.log_score)
        except Rejected:
            continue

    if not traces:
        raise RuntimeError(
            "Importance sampling: all samples were rejected. "
            "Check that your model doesn't have unsatisfiable condition() calls."
        )

    if not resample:
        return traces

    log_weights_arr = np.array(log_weights)
    max_log_w = np.max(log_weights_arr)

    if math.isinf(max_log_w) and max_log_w < 0:
        raise RuntimeError(
            "Importance sampling: all samples have -inf log-weight. "
            "The observed data may be impossible under the prior."
        )

    weights = np.exp(log_weights_arr - max_log_w)
    weights /= weights.sum()

    indices = np.random.choice(len(traces), size=len(traces), replace=True, p=weights)
    return [traces[i] for i in indices]
