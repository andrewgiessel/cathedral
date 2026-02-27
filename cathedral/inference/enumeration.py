"""Exact enumeration inference engine.

Systematically explores every possible execution path of a discrete
probabilistic model, computing exact posterior probabilities. Uses a
worklist of partially-resolved interventions: when a sample site with
finite support is reached without an intervention, the enumerator forks
into one path per support value and re-executes.

Supports depth-first, breadth-first, and likely-first traversal strategies,
plus a maxExecutions cap for approximate enumeration of large models.

Best for: small discrete models where exact answers are needed, or as
ground truth for validating sampling-based inference engines.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cathedral.trace import NeedsEnumeration, Rejected, Trace, run_with_trace


@dataclass(order=True)
class _WorkItem:
    """A pending execution path in the enumeration worklist."""

    priority: float
    interventions: dict[str, Any] = field(compare=False)
    log_prob: float = field(compare=False)


def enumerate_executions(
    model_fn: Callable,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    max_executions: int | None = None,
    strategy: str = "depth_first",
    _info: dict | None = None,
) -> list[Trace]:
    """Enumerate all executions of a discrete probabilistic model.

    Explores every possible combination of discrete random choices,
    computing exact probabilities for each execution path.

    Args:
        model_fn: The model function to enumerate.
        args: Positional arguments to pass to the model.
        kwargs: Keyword arguments to pass to the model.
        max_executions: Maximum number of complete executions to enumerate.
            None means exhaustive (all paths). Use for approximate enumeration
            of large models.
        strategy: Traversal order for the worklist.
            - "depth_first": LIFO stack order (default, low memory)
            - "breadth_first": FIFO queue order
            - "likely_first": Priority queue, most probable paths first

    Returns:
        A list of Traces. Each trace appears once per distinct execution path,
        with its log_joint reflecting the exact path probability. Paths that
        produce the same return value are NOT merged -- the caller (or Posterior)
        handles aggregation.

    Raises:
        RuntimeError: If a continuous distribution is encountered (no finite support).
    """
    if kwargs is None:
        kwargs = {}

    worklist: list[_WorkItem] | list = []
    completed: list[Trace] = []
    num_completed = 0

    _push(worklist, _WorkItem(priority=0.0, interventions={}, log_prob=0.0), strategy)

    while worklist:
        if max_executions is not None and num_completed >= max_executions:
            break

        item = _pop(worklist, strategy)

        try:
            trace = run_with_trace(
                model_fn,
                args=args,
                kwargs=kwargs,
                interventions=item.interventions,
                enumerate_mode=True,
            )
            completed.append(trace)
            num_completed += 1

        except NeedsEnumeration as e:
            support = e.distribution.support()
            if support is None:
                raise RuntimeError(
                    f"Enumeration encountered a distribution with no finite support "
                    f"at address '{e.address}': {e.distribution!r}. "
                    f"Enumeration only works with discrete distributions "
                    f"(Bernoulli, Categorical, UniformDraw)."
                ) from None

            for value in support:
                child_interventions = {**item.interventions, e.address: value}
                child_log_prob = item.log_prob + e.distribution.log_prob(value)
                priority = -child_log_prob if strategy == "likely_first" else 0.0
                _push(
                    worklist,
                    _WorkItem(priority=priority, interventions=child_interventions, log_prob=child_log_prob),
                    strategy,
                )

        except Rejected:
            num_completed += 1
            continue

    if not completed:
        raise RuntimeError(
            "Enumeration: all execution paths were rejected. Check that your model has at least one satisfiable path."
        )

    if _info is not None:
        _info["num_completed"] = num_completed
        _info["num_paths"] = len(completed)
        log_joints = np.array([t.log_joint for t in completed])
        max_lj = np.max(log_joints)
        _info["log_marginal_likelihood"] = max_lj + math.log(np.sum(np.exp(log_joints - max_lj)))
        _info["exhaustive"] = max_executions is None

    return completed


def _push(worklist: list, item: _WorkItem, strategy: str) -> None:
    if strategy == "likely_first":
        heapq.heappush(worklist, item)
    else:
        worklist.append(item)


def _pop(worklist: list, strategy: str) -> _WorkItem:
    if strategy == "likely_first":
        return heapq.heappop(worklist)
    elif strategy == "breadth_first":
        return worklist.pop(0)
    else:
        return worklist.pop()


def marginals_from_traces(traces: list[Trace]) -> dict[Any, float]:
    """Aggregate enumerated traces into a normalized marginal distribution.

    Merges traces with the same return value (using log-sum-exp for
    numerical stability) and normalizes to proper probabilities.

    Args:
        traces: Traces from enumerate_executions.

    Returns:
        Dict mapping return values to their exact posterior probabilities.
    """
    key_to_idx: dict[Any, int] = {}
    original_results: list[Any] = []
    log_probs: list[float] = []

    for trace in traces:
        key = _make_hashable(trace.result)
        if key in key_to_idx:
            idx = key_to_idx[key]
            log_probs[idx] = float(np.logaddexp(log_probs[idx], trace.log_joint))
        else:
            idx = len(original_results)
            key_to_idx[key] = idx
            original_results.append(trace.result)
            log_probs.append(trace.log_joint)

    max_lp = max(log_probs)
    if math.isinf(max_lp) and max_lp < 0:
        raise RuntimeError("All enumerated paths have -inf probability.")

    log_total = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))

    out: _MarginalDict = _MarginalDict()
    for idx, result in enumerate(original_results):
        out._insert(result, math.exp(log_probs[idx] - log_total))

    return out


class _MarginalDict(dict):
    """Dict-like container that supports unhashable keys (like dicts) via linear scan fallback."""

    def __init__(self):
        super().__init__()
        self._unhashable_items: list[tuple[Any, float]] = []

    def _insert(self, key: Any, value: float) -> None:
        try:
            hash(key)
            self[key] = value
        except TypeError:
            self._unhashable_items.append((key, value))

    def items(self):
        yield from super().items()
        yield from self._unhashable_items

    def values(self):
        yield from super().values()
        for _, v in self._unhashable_items:
            yield v

    def __iter__(self):
        yield from super().__iter__()
        for k, _ in self._unhashable_items:
            yield k

    def __len__(self):
        return super().__len__() + len(self._unhashable_items)


def _make_hashable(val: Any) -> Any:
    """Convert a result value to a hashable key for comparison."""
    if isinstance(val, dict):
        return ("__dict__", tuple(sorted((k, _make_hashable(v)) for k, v in val.items())))
    if isinstance(val, list):
        return ("__list__", tuple(_make_hashable(v) for v in val))
    if isinstance(val, np.ndarray):
        return ("__ndarray__", val.tobytes(), val.shape)
    return val
