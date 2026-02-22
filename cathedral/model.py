"""Model decorator and inference entry point for Cathedral.

Provides the @model decorator for marking probabilistic model functions,
the infer() function for running inference, and the Posterior class for
analyzing results.
"""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from cathedral.inference.enumeration import enumerate_executions
from cathedral.inference.importance import importance_sample
from cathedral.inference.mh import mh_sample
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
    the posterior distribution. Supports optional weights for engines
    like enumeration where traces have unequal probabilities.
    """

    def __init__(self, traces: list[Trace], weights: np.ndarray | None = None):
        if not traces:
            raise ValueError("Posterior requires at least one trace")
        self._traces = traces
        self._results = [t.result for t in traces]
        self._weights = weights

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
        values = np.array(self._extract_values(key), dtype=float)
        if self._weights is not None:
            return float(np.sum(values * self._weights))
        return float(np.mean(values))

    def std(self, key: str | None = None) -> float:
        """Compute the posterior standard deviation.

        Args:
            key: If results are dicts, compute std of this key.

        Returns:
            The posterior standard deviation.
        """
        values = np.array(self._extract_values(key), dtype=float)
        if self._weights is not None:
            mu = float(np.sum(values * self._weights))
            return float(np.sqrt(np.sum(self._weights * (values - mu) ** 2)))
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
            mask = np.array([bool(_get_value(r, key)) for r in self._results], dtype=float)
        elif callable(predicate):
            mask = np.array([bool(predicate(r)) for r in self._results], dtype=float)
        else:
            mask = np.array([bool(r) for r in self._results], dtype=float)

        if self._weights is not None:
            return float(np.sum(mask * self._weights))
        return float(np.mean(mask))

    def histogram(self, key: str | None = None) -> dict[Any, float]:
        """Compute a histogram (empirical distribution) over discrete values.

        Args:
            key: If results are dicts, compute histogram of this key.

        Returns:
            Dict mapping values to their estimated probabilities.
        """
        values = self._extract_values(key)
        if self._weights is not None:
            hist: dict[Any, float] = {}
            for v, w in zip(values, self._weights, strict=False):
                hist[v] = hist.get(v, 0.0) + w
            return hist
        counts = Counter(values)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def credible_interval(self, level: float = 0.95, key: str | None = None) -> tuple[float, float]:
        """Compute a credible interval.

        Args:
            level: Credible level (default 0.95 for 95% CI).
            key: If results are dicts, compute interval for this key.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        values = np.array(self._extract_values(key), dtype=float)
        if self._weights is not None:
            order = np.argsort(values)
            sorted_vals = values[order]
            cum_weights = np.cumsum(self._weights[order])
            alpha = (1 - level) / 2
            lower = float(sorted_vals[np.searchsorted(cum_weights, alpha)])
            upper = float(sorted_vals[np.searchsorted(cum_weights, 1 - alpha)])
            return (lower, upper)
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
            - "mh": Single-site Metropolis-Hastings (for complex models, supports burn_in and lag kwargs)
            - "enumerate": Exact enumeration (for small discrete models, supports strategy and max_executions)
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
    elif method == "mh":
        burn_in = kwargs.pop("burn_in", None)
        lag = kwargs.pop("lag", 1)
        traces = mh_sample(
            fn,
            args=args,
            num_samples=num_samples,
            burn_in=burn_in,
            lag=lag,
        )
    elif method == "enumerate":
        max_executions = kwargs.pop("max_executions", None)
        strategy = kwargs.pop("strategy", "depth_first")
        traces = enumerate_executions(
            fn,
            args=args,
            max_executions=max_executions,
            strategy=strategy,
        )
        log_joints = np.array([t.log_joint for t in traces])
        max_lj = np.max(log_joints)
        weights = np.exp(log_joints - max_lj)
        weights /= weights.sum()
        return Posterior(traces, weights=weights)
    else:
        raise ValueError(
            f"Unknown inference method: {method!r}. "
            f"Choose from: 'rejection', 'importance', 'mh', 'enumerate'"
        )

    return Posterior(traces)
