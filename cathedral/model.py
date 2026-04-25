"""Model decorator and inference entry point for Cathedral.

Provides the @model decorator for marking probabilistic model functions,
the infer() function for running inference, and the Posterior class for
analyzing results.
"""

from __future__ import annotations

import contextlib
import functools
import pickle
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cathedral._rng import SeedLike
from cathedral.inference.enumeration import enumerate_executions
from cathedral.inference.importance import importance_sample
from cathedral.inference.mh import mh_sample
from cathedral.inference.rejection import rejection_sample
from cathedral.trace import _CAPTURE_SCOPES, Trace


def _set_original_fn(wrapper: Any, fn: Callable) -> None:
    """Attach the original callable without triggering static attribute checks."""
    wrapper.__dict__["_original_fn"] = fn


@dataclass
class InferenceInfo:
    """Diagnostic metadata from an inference run."""

    method: str
    num_samples: int
    num_attempts: int | None = None
    acceptance_rate: float | None = None
    log_weights: np.ndarray | None = field(default=None, repr=False)
    log_marginal_likelihood: float | None = None
    ess: float | None = None
    extra: dict = field(default_factory=dict)


def model(fn: Callable) -> Callable:
    """Decorator marking a function as a Cathedral probabilistic model.

    The decorated function can be passed to infer() for probabilistic inference,
    or called directly for forward sampling.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    _set_original_fn(wrapper, fn)
    return wrapper


class Posterior:
    """Results from probabilistic inference.

    Wraps a collection of traces and provides methods for analyzing
    the posterior distribution. Supports optional weights for engines
    like enumeration where traces have unequal probabilities.
    """

    def __init__(
        self,
        traces: list[Trace],
        weights: np.ndarray | None = None,
        info: InferenceInfo | None = None,
    ):
        if not traces:
            raise ValueError("Posterior requires at least one trace")
        self._traces = traces
        self._results = [t.result for t in traces]
        self._weights = weights
        self.info = info

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

    @property
    def has_fixed_structure(self) -> bool:
        """Whether all traces share the same set of choice addresses."""
        if not self._traces:
            return True
        reference = set(self._traces[0].choices.keys())
        return all(set(t.choices.keys()) == reference for t in self._traces)

    @property
    def ess(self) -> float | None:
        """Effective sample size, if available from inference."""
        if self.info is not None and self.info.ess is not None:
            return self.info.ess
        if self._weights is not None:
            return 1.0 / float(np.sum(self._weights**2))
        return None

    @property
    def acceptance_rate(self) -> float | None:
        """Acceptance rate, if available from inference."""
        if self.info is not None:
            return self.info.acceptance_rate
        return None

    @property
    def log_marginal_likelihood(self) -> float | None:
        """Log marginal likelihood estimate, if available from inference."""
        if self.info is not None:
            return self.info.log_marginal_likelihood
        return None

    def save(self, path: str | Path) -> None:
        """Save this posterior to a pickle file.

        Pickle is intended for local checkpointing with trusted files. Do not
        load posterior pickle files from untrusted sources.

        Args:
            path: File path where the posterior should be written.
        """
        with Path(path).open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Posterior:
        """Load a posterior from a pickle file.

        Pickle can execute code while loading. Only load files you created or
        otherwise trust.

        Args:
            path: File path to a posterior pickle file.

        Returns:
            The loaded Posterior.

        Raises:
            TypeError: If the pickle file does not contain a Posterior.
        """
        with Path(path).open("rb") as f:
            posterior = pickle.load(f)  # noqa: S301
        if not isinstance(posterior, cls):
            raise TypeError(f"Expected a Posterior pickle, got {type(posterior).__name__}")
        return posterior

    def extend(
        self,
        model_fn: Callable,
        *args: Any,
        num_samples: int | None = None,
        capture_scopes: bool = False,
        condition: Callable[[Any], bool] | None = None,
        seed: SeedLike = None,
        **kwargs: Any,
    ) -> Posterior:
        """Continue inference with the same method and return a combined posterior.

        Rejection and importance sampling append new traces. MH continues from
        the last trace in this posterior and appends the new chain states.
        Enumeration recomputes the enumerated path set instead of duplicating
        exact paths.

        Args:
            model_fn: The same model function used to create this posterior.
            *args: Arguments to pass to the model function.
            num_samples: Number of additional samples to draw. Defaults to the
                current posterior size for sampling methods.
            capture_scopes: If True, record scope paths on newly generated traces.
            condition: Optional external condition predicate to apply while extending.
            seed: Optional seed for the extension run.
            **kwargs: Additional keyword arguments for the original inference method.

        Returns:
            A Posterior containing the original traces plus new traces for
            sampling methods, or a recomputed Posterior for enumeration.

        Raises:
            ValueError: If this posterior does not have inference method metadata.
        """
        if self.info is None:
            raise ValueError("Posterior.extend() requires inference metadata from infer().")

        method = self.info.method
        requested = self.num_samples if num_samples is None else num_samples

        if method == "mh":
            return self._extend_mh(
                model_fn,
                args,
                requested,
                capture_scopes=capture_scopes,
                condition=condition,
                seed=seed,
                **kwargs,
            )

        if method == "importance" and "resample" not in kwargs:
            kwargs["resample"] = self._weights is None

        new = infer(
            model_fn,
            *args,
            method=method,
            num_samples=requested,
            capture_scopes=capture_scopes,
            condition=condition,
            seed=seed,
            **kwargs,
        )

        if method == "rejection":
            return _combine_rejection_posteriors(self, new)
        if method == "importance":
            return _combine_importance_posteriors(self, new)
        if method == "enumerate":
            return new

        raise ValueError(f"Posterior.extend() does not support method {method!r}.")

    def _extend_mh(
        self,
        model_fn: Callable,
        args: tuple,
        num_samples: int,
        *,
        capture_scopes: bool,
        condition: Callable[[Any], bool] | None,
        seed: SeedLike,
        **kwargs: Any,
    ) -> Posterior:
        """Continue MH from the last retained trace."""
        if self.info is None:
            raise ValueError("Posterior.extend() requires inference metadata from infer().")

        fn = getattr(model_fn, "_original_fn", model_fn)
        if condition is not None:
            fn = _wrap_with_condition(fn, condition)

        burn_in = kwargs.pop("burn_in", 0)
        lag = kwargs.pop("lag", 1)
        engine_info: dict = {}
        token = _CAPTURE_SCOPES.set(capture_scopes)
        try:
            new_traces = mh_sample(
                fn,
                args=args,
                num_samples=num_samples,
                burn_in=burn_in,
                lag=lag,
                initial_trace=self.traces[-1],
                seed=seed,
                _info=engine_info,
            )
        finally:
            _CAPTURE_SCOPES.reset(token)

        combined_traces = [*self.traces, *new_traces]
        info = _combine_mh_info(self.info, engine_info, len(combined_traces))
        return Posterior(combined_traces, info=info)

    def to_arviz(self, chain_id: int = 0) -> Any:
        """Convert this Posterior to an ArviZ InferenceData object.

        Only works for fixed-structure posteriors where all traces share
        the same set of choice addresses with numeric values.

        Requires arviz to be installed: ``pip install cathedral[viz]``

        Args:
            chain_id: Chain identifier for the ArviZ dataset.

        Returns:
            An arviz.InferenceData object.

        Raises:
            ImportError: If arviz is not installed.
            ValueError: If the posterior has variable structure.
        """
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError("to_arviz() requires arviz. Install it with: pip install cathedral[viz]") from e

        if not self.has_fixed_structure:
            raise ValueError(
                "to_arviz() requires fixed-structure traces (all traces must "
                "have the same choice addresses). Use structure_summary() to "
                "inspect your posterior's structure."
            )

        if not self._traces:
            raise ValueError("No traces in posterior")

        addresses = list(self._traces[0].choices.keys())
        posterior_dict: dict[str, np.ndarray] = {}

        for addr in addresses:
            values = []
            for t in self._traces:
                v = t.choices[addr].value
                if isinstance(v, bool | np.bool_ | int | float | np.integer | np.floating):
                    values.append(float(v))
                else:
                    values.append(v)
            with contextlib.suppress(ValueError, TypeError):
                posterior_dict[addr] = np.array(values)[np.newaxis, :]

        return az.from_dict(posterior=posterior_dict)

    def diagnostics(self) -> str:
        """Return a human-readable summary of inference diagnostics."""
        lines: list[str] = []
        lines.append(f"Posterior: {self.num_samples} samples")
        if self.info is not None:
            lines.append(f"  method: {self.info.method}")
            if self.info.num_attempts is not None:
                lines.append(f"  attempts: {self.info.num_attempts}")
            if self.info.acceptance_rate is not None:
                lines.append(f"  acceptance rate: {self.info.acceptance_rate:.4f}")
            if self.info.ess is not None:
                lines.append(f"  ESS: {self.info.ess:.1f}")
            if self.info.log_marginal_likelihood is not None:
                lines.append(f"  log marginal likelihood: {self.info.log_marginal_likelihood:.4f}")
            if self.info.extra:
                for k, v in self.info.extra.items():
                    lines.append(f"  {k}: {v}")
        ess = self.ess
        if ess is not None and self.info is not None and self.info.ess is None:
            lines.append(f"  ESS (from weights): {ess:.1f}")
        lines.append(f"  fixed structure: {self.has_fixed_structure}")
        return "\n".join(lines)

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
            return _weighted_histogram(values, self._weights)
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

    def summary(self, key: str | None = None, level: float = 0.95) -> dict[str, Any]:
        """Return common numeric posterior summaries.

        Args:
            key: If results are dicts or objects, summarize this field.
                If None, summarize the results directly.
            level: Credible interval level.

        Returns:
            A dictionary with sample count, mean, standard deviation,
            credible interval, ESS, and fixed-structure status.
        """
        return {
            "num_samples": self.num_samples,
            "mean": self.mean(key),
            "std": self.std(key),
            "credible_interval": self.credible_interval(level, key),
            "level": level,
            "ess": self.ess,
            "has_fixed_structure": self.has_fixed_structure,
        }

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


class _UnhashableHistogram(dict):
    """Dict subclass that also yields unhashable (value, weight) pairs."""

    def __init__(self, hashable: dict, unhashable: list[tuple[Any, float]]):
        super().__init__(hashable)
        self._unhashable = unhashable

    def items(self):
        yield from super().items()
        yield from self._unhashable

    def values(self):
        yield from super().values()
        for _, w in self._unhashable:
            yield w

    def __iter__(self):
        yield from super().__iter__()
        for k, _ in self._unhashable:
            yield k

    def __len__(self):
        return super().__len__() + len(self._unhashable)


def _accumulate_unhashable(items: list[tuple[Any, float]], value: Any, weight: float) -> None:
    """Add *weight* for *value* into the unhashable linear-scan list in place."""
    for i, (uv, uw) in enumerate(items):
        if uv == value:
            items[i] = (uv, uw + weight)
            return
    items.append((value, weight))


def _weighted_histogram(values: list, weights: np.ndarray) -> dict[Any, float]:
    """Build a weighted histogram, handling unhashable values via linear scan fallback."""
    hashable_hist: dict[Any, float] = {}
    unhashable_items: list[tuple[Any, float]] = []

    for v, w in zip(values, weights, strict=False):
        try:
            hash(v)
            hashable_hist[v] = hashable_hist.get(v, 0.0) + w
        except TypeError:
            _accumulate_unhashable(unhashable_items, v, w)

    if not unhashable_items:
        return hashable_hist

    return _UnhashableHistogram(hashable_hist, unhashable_items)


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Normalize log weights into probability weights."""
    if len(log_weights) == 0:
        raise ValueError("Cannot normalize empty log weights")

    max_log_w = np.max(log_weights)
    if np.isneginf(max_log_w):
        raise RuntimeError(
            "Importance sampling: all samples have -inf log-weight. "
            "The observed data may be impossible under the prior."
        )

    weights = np.exp(log_weights - max_log_w)
    weights /= weights.sum()
    return weights


def _importance_diagnostics(log_weights: np.ndarray) -> tuple[float | None, float | None]:
    """Compute log marginal likelihood and ESS from log weights."""
    if len(log_weights) == 0:
        return None, None

    max_lw = np.max(log_weights)
    if np.isneginf(max_lw):
        return None, None

    weights = np.exp(log_weights - max_lw)
    log_marginal_likelihood = float(max_lw + np.log(np.mean(weights)))
    normalized = weights / weights.sum()
    ess = 1.0 / float(np.sum(normalized**2))
    return log_marginal_likelihood, ess


def _sum_optional_ints(a: int | None, b: int | None) -> int | None:
    """Add optional integer diagnostics, preserving None when either side is unknown."""
    if a is None or b is None:
        return None
    return a + b


def _combine_rejection_posteriors(old: Posterior, new: Posterior) -> Posterior:
    """Append rejection samples and combine acceptance diagnostics."""
    combined_traces = [*old.traces, *new.traces]
    num_attempts = _sum_optional_ints(old.info.num_attempts, new.info.num_attempts) if old.info and new.info else None
    acceptance_rate = len(combined_traces) / num_attempts if num_attempts else None
    info = InferenceInfo(
        method="rejection",
        num_samples=len(combined_traces),
        num_attempts=num_attempts,
        acceptance_rate=acceptance_rate,
    )
    return Posterior(combined_traces, info=info)


def _combine_importance_posteriors(old: Posterior, new: Posterior) -> Posterior:
    """Append importance samples, preserving weighted or resampled semantics."""
    old_weighted = old._weights is not None
    new_weighted = new._weights is not None
    if old_weighted != new_weighted:
        raise ValueError("Posterior.extend() requires the same importance resample setting.")

    combined_traces = [*old.traces, *new.traces]
    log_weights = None
    log_marginal_likelihood = None
    ess = None
    weights = None

    if (
        old.info is not None
        and new.info is not None
        and old.info.log_weights is not None
        and new.info.log_weights is not None
    ):
        log_weights = np.concatenate([old.info.log_weights, new.info.log_weights])
        log_marginal_likelihood, ess = _importance_diagnostics(log_weights)
        if old_weighted:
            weights = _normalize_log_weights(log_weights)

    num_attempts = _sum_optional_ints(old.info.num_attempts, new.info.num_attempts) if old.info and new.info else None
    info = InferenceInfo(
        method="importance",
        num_samples=len(combined_traces),
        num_attempts=num_attempts,
        log_weights=log_weights,
        log_marginal_likelihood=log_marginal_likelihood,
        ess=ess,
    )
    return Posterior(combined_traces, weights=weights, info=info)


def _combine_mh_info(old_info: InferenceInfo, new_engine_info: dict, num_samples: int) -> InferenceInfo:
    """Combine MH diagnostics from an existing posterior and an extension run."""
    old_total_steps = old_info.extra.get("total_steps") if old_info.extra else None
    new_total_steps = new_engine_info.get("total_steps")
    total_steps = _sum_optional_ints(old_total_steps, new_total_steps)

    old_acceptance_rate = old_info.acceptance_rate
    new_acceptance_rate = new_engine_info.get("acceptance_rate")
    if total_steps and old_total_steps is not None and new_total_steps is not None:
        old_accepted = (old_acceptance_rate or 0.0) * old_total_steps
        new_accepted = (new_acceptance_rate or 0.0) * new_total_steps
        acceptance_rate = (old_accepted + new_accepted) / total_steps
    else:
        acceptance_rate = new_acceptance_rate

    return InferenceInfo(
        method="mh",
        num_samples=num_samples,
        acceptance_rate=acceptance_rate,
        extra={
            "total_steps": total_steps,
            "burn_in": new_engine_info.get("burn_in"),
            "lag": new_engine_info.get("lag"),
        },
    )


def infer(
    model_fn: Callable,
    *args: Any,
    method: str = "rejection",
    num_samples: int = 1000,
    capture_scopes: bool = False,
    condition: Callable[[Any], bool] | None = None,
    seed: SeedLike = None,
    **kwargs: Any,
) -> Posterior:
    """Run probabilistic inference on a model.

    Supports Church-style separation of model and query: write a pure
    generative model and pass the condition externally via ``condition``.
    This lets you ask many different questions of the same model without
    rewriting it::

        @model
        def lawn():
            rain = flip(0.3)
            sprinkler = flip(0.5)
            wet = rain or sprinkler
            return {"rain": rain, "sprinkler": sprinkler, "wet": wet}

        infer(lawn, condition=lambda r: r["wet"])            # P(rain | wet)
        infer(lawn, condition=lambda r: r["wet"] and not r["sprinkler"])

    The ``condition`` predicate composes with any ``condition()`` calls
    already inside the model — both must be satisfied.

    Args:
        model_fn: A function decorated with @model (or any callable).
        *args: Arguments to pass to the model function.
        method: Inference method. One of:
            - "rejection": Rejection sampling (for condition()-based models)
            - "importance": Likelihood-weighted importance sampling (for observe()-based models)
            - "mh": Single-site Metropolis-Hastings (for complex models, supports burn_in and lag kwargs)
            - "enumerate": Exact enumeration (for small discrete models, supports strategy and max_executions)
        num_samples: Number of posterior samples to collect.
        capture_scopes: If True, record scope paths on each choice via
            Python stack introspection. Useful for trace visualization.
        condition: Optional predicate applied to the model's return value.
            Executions where ``condition(result)`` is False are rejected,
            equivalent to calling ``condition(predicate(result))`` at the
            end of the model body. Enables Church-style query separation.
        seed: Optional seed for reproducible stochastic inference.
        **kwargs: Additional keyword arguments passed to the inference engine.
            Common options include:
            - rejection: ``max_attempts``
            - importance: ``resample``
            - mh: ``burn_in``, ``lag``

    Returns:
        A Posterior object for analyzing the results.
    """
    fn = getattr(model_fn, "_original_fn", model_fn)

    if condition is not None:
        fn = _wrap_with_condition(fn, condition)

    token = _CAPTURE_SCOPES.set(capture_scopes)
    try:
        return _run_inference(fn, args, method, num_samples, seed=seed, **kwargs)
    finally:
        _CAPTURE_SCOPES.reset(token)


def _wrap_with_condition(fn: Callable, predicate: Callable[[Any], bool]) -> Callable:
    """Wrap a model function so its return value is conditioned on a predicate."""
    from cathedral.primitives import condition as _condition

    @functools.wraps(fn)
    def conditioned(*args, **kwargs):
        result = fn(*args, **kwargs)
        _condition(predicate(result))
        return result

    return conditioned


def _run_inference(
    fn: Callable,
    args: tuple,
    method: str,
    num_samples: int,
    seed: SeedLike = None,
    **kwargs: Any,
) -> Posterior:
    engine_info: dict = {}

    if method == "rejection":
        max_attempts = kwargs.pop("max_attempts", None)
        traces = rejection_sample(
            fn,
            args=args,
            num_samples=num_samples,
            max_attempts=max_attempts,
            seed=seed,
            _info=engine_info,
        )
        info = InferenceInfo(
            method="rejection",
            num_samples=len(traces),
            num_attempts=engine_info.get("num_attempts"),
            acceptance_rate=engine_info.get("acceptance_rate"),
        )
        return Posterior(traces, info=info)

    elif method == "importance":
        resample = kwargs.pop("resample", True)
        traces = importance_sample(
            fn,
            args=args,
            num_samples=num_samples,
            resample=resample,
            seed=seed,
            _info=engine_info,
        )
        weights = None
        if not resample:
            weights = _normalize_log_weights(engine_info["log_weights"])
        info = InferenceInfo(
            method="importance",
            num_samples=len(traces),
            num_attempts=engine_info.get("num_attempts"),
            log_weights=engine_info.get("log_weights"),
            log_marginal_likelihood=engine_info.get("log_marginal_likelihood"),
            ess=engine_info.get("ess"),
        )
        return Posterior(traces, weights=weights, info=info)

    elif method == "mh":
        burn_in = kwargs.pop("burn_in", None)
        lag = kwargs.pop("lag", 1)
        traces = mh_sample(
            fn,
            args=args,
            num_samples=num_samples,
            burn_in=burn_in,
            lag=lag,
            seed=seed,
            _info=engine_info,
        )
        info = InferenceInfo(
            method="mh",
            num_samples=len(traces),
            acceptance_rate=engine_info.get("acceptance_rate"),
            extra={
                "total_steps": engine_info.get("total_steps"),
                "burn_in": engine_info.get("burn_in"),
                "lag": engine_info.get("lag"),
            },
        )
        return Posterior(traces, info=info)

    elif method == "enumerate":
        max_executions = kwargs.pop("max_executions", None)
        strategy = kwargs.pop("strategy", "depth_first")
        traces = enumerate_executions(
            fn,
            args=args,
            max_executions=max_executions,
            strategy=strategy,
            _info=engine_info,
        )
        log_joints = np.array([t.log_joint for t in traces])
        max_lj = np.max(log_joints)
        weights = np.exp(log_joints - max_lj)
        weights /= weights.sum()
        info = InferenceInfo(
            method="enumerate",
            num_samples=len(traces),
            log_marginal_likelihood=engine_info.get("log_marginal_likelihood"),
            extra={
                "num_paths": engine_info.get("num_paths"),
                "exhaustive": engine_info.get("exhaustive"),
            },
        )
        return Posterior(traces, weights=weights, info=info)

    else:
        raise ValueError(
            f"Unknown inference method: {method!r}. Choose from: 'rejection', 'importance', 'mh', 'enumerate'"
        )
