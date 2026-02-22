"""Trace infrastructure for Cathedral.

A Trace records every random choice made during execution of a probabilistic
model, along with accumulated log-scores from conditioning and observations.
The tracing context is threaded via contextvars so model code doesn't need
to pass trace objects explicitly.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from cathedral.distributions import Distribution

_TRACE: ContextVar[TraceContext | None] = ContextVar("cathedral_trace", default=None)


class Rejected(Exception):
    """Raised when a condition() call fails during traced execution."""


@dataclass
class Choice:
    """A single random choice recorded in a trace."""

    address: str
    distribution: Distribution
    value: Any
    log_prob: float


@dataclass
class Trace:
    """A complete execution trace of a probabilistic model.

    Contains all random choices made, accumulated log-score from
    observe/condition/factor, and the model's return value.
    """

    choices: dict[str, Choice] = field(default_factory=dict)
    log_score: float = 0.0
    result: Any = None

    @property
    def log_joint(self) -> float:
        """Total log-probability: sum of choice log-probs plus log-score."""
        return sum(c.log_prob for c in self.choices.values()) + self.log_score

    @property
    def addresses(self) -> list[str]:
        """All choice addresses in insertion order."""
        return list(self.choices.keys())

    def __len__(self) -> int:
        return len(self.choices)

    def __str__(self) -> str:
        lines = ["Trace"]
        lines.append(f"  result: {self.result!r}")
        lines.append(f"  log_joint: {self.log_joint:.4f}")
        if self.log_score != 0.0:
            lines.append(f"  log_score: {self.log_score:.4f}")
        lines.append(f"  choices ({len(self.choices)}):")
        for addr, choice in self.choices.items():
            lines.append(f"    {addr}: {choice.value!r}  (log_prob={choice.log_prob:.4f}, {choice.distribution})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Trace(choices={len(self.choices)}, log_joint={self.log_joint:.4f}, result={self.result!r})"


class TraceContext:
    """Manages trace state during model execution.

    Handles auto-addressing of sample sites and optional interventions
    (for replay/MH proposals).
    """

    def __init__(self, interventions: dict[str, Any] | None = None):
        self.trace = Trace()
        self.interventions = interventions or {}
        self._counter: int = 0
        self._address_counts: dict[str, int] = {}
        self._memo_caches: dict[int, dict] = {}

    def fresh_address(self, prefix: str = "sample") -> str:
        """Generate a unique address for a sample site."""
        count = self._address_counts.get(prefix, 0)
        self._address_counts[prefix] = count + 1
        if count == 0:
            return prefix
        return f"{prefix}__{count}"

    def record_choice(self, address: str, distribution: Distribution, value: Any) -> None:
        """Record a random choice in the trace."""
        log_p = distribution.log_prob(value)
        self.trace.choices[address] = Choice(address, distribution, value, log_p)

    def add_score(self, score: float) -> None:
        """Add to the accumulated log-score (from observe/factor)."""
        self.trace.log_score += score

    def has_intervention(self, address: str) -> bool:
        """Check if an intervention exists for this address."""
        return address in self.interventions

    def get_intervention(self, address: str) -> Any:
        """Get the intervention value for this address."""
        return self.interventions[address]

    def get_memo_cache(self, func_id: int) -> dict:
        """Get or create the memo cache for a given function within this trace."""
        if func_id not in self._memo_caches:
            self._memo_caches[func_id] = {}
        return self._memo_caches[func_id]


def get_trace_context() -> TraceContext | None:
    """Get the currently active trace context, or None."""
    return _TRACE.get()


def run_with_trace(
    fn: callable,
    args: tuple = (),
    kwargs: dict | None = None,
    interventions: dict[str, Any] | None = None,
) -> Trace:
    """Execute a function within a fresh tracing context and return the trace.

    Args:
        fn: The model function to execute.
        args: Positional arguments to pass to fn.
        kwargs: Keyword arguments to pass to fn.
        interventions: Optional dict mapping addresses to values to intervene on.

    Returns:
        The completed Trace.

    Raises:
        Rejected: If a condition() call fails during execution.
    """
    if kwargs is None:
        kwargs = {}

    ctx = TraceContext(interventions=interventions)
    token = _TRACE.set(ctx)
    try:
        result = fn(*args, **kwargs)
        ctx.trace.result = result
        return ctx.trace
    finally:
        _TRACE.reset(token)
