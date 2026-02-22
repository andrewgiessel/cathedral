"""Model decorator and inference entry point for Cathedral.

Provides the @model decorator for marking probabilistic model functions,
the infer() function for running inference, and the Posterior class for
analyzing results.
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import Any, Callable

import numpy as np

from cathedral.inference.importance import importance_sample
from cathedral.inference.rejection import rejection_sample
from cathedral.trace import Trace


def model(fn: Callable) -> Callable:
    """Decorator marking a function as a Cathedral probabilistic model.

    The decorated function can be passed to infer() for probabilistic inference,
    or called directly for forward sampling.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_cathedral_model = True
    wrapper._original_fn = fn
    return wrapper


class Posterior:
    """Results from probabilistic inference.

    Wraps a collection of traces and provides methods for analyzing
    the posterior distribution.
    """

    def __init__(self, traces: list[Trace]):
        if not traces:
            raise ValueError("Posterior requires at least one trace")
        self._traces = traces
        self._results = [t.result for t in traces]

    @property
    def traces(self) -> list[Trace]:
        """The raw traces from inference."""
        return self._traces

    @property
    def samples(self) -> list[Any]:
        """The return values from each trace."""
        return self._results

    @property
    def num_samples(self) -> int:
        """Number of samples in the posterior."""
        return len(self._traces)

    def mean(self, key: str | None = None) -> float:
        """Compute the posterior mean.

        Args:
            key: If results are dicts, compute mean of this key.
                If None, compute mean of the results directly (must be numeric).

        Returns:
            The posterior mean.
        """
        values = self._extract_values(key)
        return float(np.mean(values))

    def std(self, key: str | None = None) -> float:
        """Compute the posterior standard deviation.

        Args:
            key: If results are dicts, compute std of this key.

        Returns:
            The posterior standard deviation.
        """
        values = self._extract_values(key)
        return float(np.std(values))

    def probability(self, predicate: Callable[[Any], bool] | str | None = None) -> float:
        """Estimate the probability that a predicate holds.

        Args:
            predicate: Either a callable that takes a result and returns bool,
                or a string key (estimates P(key=True) for boolean results).

        Returns:
            Estimated probability.
        """
        if isinstance(predicate, str):
            key = predicate
            count = sum(1 for r in self._results if _get_value(r, key))
        elif callable(predicate):
            count = sum(1 for r in self._results if predicate(r))
        else:
            count = sum(1 for r in self._results if r)
        return count / len(self._results)

    def histogram(self, key: str | None = None) -> dict[Any, float]:
        """Compute a histogram (empirical distribution) over discrete values.

        Args:
            key: If results are dicts, compute histogram of this key.

        Returns:
            Dict mapping values to their estimated probabilities.
        """
        values = self._extract_values(key)
        counts = Counter(values)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def credible_interval(self, key: str | None = None, level: float = 0.95) -> tuple[float, float]:
        """Compute a credible interval.

        Args:
            key: If results are dicts, compute interval for this key.
            level: Credible level (default 0.95 for 95% CI).

        Returns:
            Tuple of (lower, upper) bounds.
        """
        values = np.array(self._extract_values(key), dtype=float)
        alpha = (1 - level) / 2
        lower = float(np.percentile(values, 100 * alpha))
        upper = float(np.percentile(values, 100 * (1 - alpha)))
        return (lower, upper)

    def _extract_values(self, key: str | None) -> list:
        """Extract numeric values from results, optionally by key."""
        if key is not None:
            return [_get_value(r, key) for r in self._results]
        return list(self._results)

    def __repr__(self) -> str:
        return f"Posterior(num_samples={self.num_samples})"


def _get_value(result: Any, key: str) -> Any:
    """Extract a value from a result by key (supports dicts and attribute access)."""
    if isinstance(result, dict):
        return result[key]
    return getattr(result, key)


def infer(
    model_fn: Callable,
    *args: Any,
    method: str = "rejection",
    num_samples: int = 1000,
    **kwargs: Any,
) -> Posterior:
    """Run probabilistic inference on a model.

    Args:
        model_fn: A function decorated with @model (or any callable).
        *args: Arguments to pass to the model function.
        method: Inference method. One of:
            - "rejection": Rejection sampling (for condition()-based models)
            - "importance": Likelihood-weighted importance sampling (for observe()-based models)
        num_samples: Number of posterior samples to collect.
        **kwargs: Additional keyword arguments passed to the inference engine.

    Returns:
        A Posterior object for analyzing the results.
    """
    fn = getattr(model_fn, "_original_fn", model_fn)

    if method == "rejection":
        max_attempts = kwargs.pop("max_attempts", None)
        traces = rejection_sample(
            fn,
            args=args,
            num_samples=num_samples,
            max_attempts=max_attempts,
        )
    elif method == "importance":
        resample = kwargs.pop("resample", True)
        traces = importance_sample(
            fn,
            args=args,
            num_samples=num_samples,
            resample=resample,
        )
    else:
        raise ValueError(f"Unknown inference method: {method!r}. Choose from: 'rejection', 'importance'")

    return Posterior(traces)
