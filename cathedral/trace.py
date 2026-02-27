"""Trace infrastructure for Cathedral.

A Trace records every random choice made during execution of a probabilistic
model, along with accumulated log-scores from conditioning and observations.
The tracing context is threaded via contextvars so model code doesn't need
to pass trace objects explicitly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from cathedral.distributions import Distribution

_TRACE: ContextVar[TraceContext | None] = ContextVar("cathedral_trace", default=None)
_CAPTURE_SCOPES: ContextVar[bool] = ContextVar("cathedral_capture_scopes", default=False)


class Rejected(Exception):
    """Raised when a condition() call fails during traced execution."""


class NeedsEnumeration(Exception):
    """Raised during enumeration when sample() hits an un-intervened discrete site."""

    def __init__(self, address: str, distribution: Distribution):
        self.address = address
        self.distribution = distribution
        super().__init__(f"Enumeration fork at {address}")


@dataclass
class Choice:
    """A single random choice recorded in a trace."""

    address: str
    distribution: Distribution
    value: Any
    log_prob: float
    scope_path: tuple[str, ...] = ()


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
            scope = f"  [{'/'.join(choice.scope_path)}]" if choice.scope_path else ""
            lines.append(
                f"    {addr}: {choice.value!r}  (log_prob={choice.log_prob:.4f}, {choice.distribution}){scope}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Trace(choices={len(self.choices)}, log_joint={self.log_joint:.4f}, result={self.result!r})"


class TraceContext:
    """Manages trace state during model execution.

    Handles auto-addressing of sample sites, optional interventions
    (for replay/MH proposals), and optional scope tracking for
    trace visualization.
    """

    def __init__(
        self,
        interventions: dict[str, Any] | None = None,
        enumerate_mode: bool = False,
        capture_scopes: bool = False,
    ):
        self.trace = Trace()
        self.interventions = interventions or {}
        self.enumerate_mode = enumerate_mode
        self._counter: int = 0
        self._address_counts: dict[str, int] = {}
        self._memo_caches: dict[int, dict] = {}
        self._scope_stack: list[str] = []
        self._capture_scopes = capture_scopes

    def fresh_address(self, prefix: str = "sample") -> str:
        """Generate a unique address for a sample site."""
        count = self._address_counts.get(prefix, 0)
        self._address_counts[prefix] = count + 1
        if count == 0:
            return prefix
        return f"{prefix}__{count}"

    def push_scope(self, name: str) -> None:
        """Push a named scope onto the scope stack."""
        self._scope_stack.append(name)

    def pop_scope(self) -> None:
        """Pop the innermost scope from the scope stack."""
        self._scope_stack.pop()

    def record_choice(self, address: str, distribution: Distribution, value: Any) -> None:
        """Record a random choice in the trace."""
        log_p = distribution.log_prob(value)
        scope_path = self._resolve_scope()
        self.trace.choices[address] = Choice(address, distribution, value, log_p, scope_path)

    def _resolve_scope(self) -> tuple[str, ...]:
        """Determine the current scope path for a choice being recorded.

        When capture_scopes is False, returns only explicit scopes from
        mem()/DPmem() push_scope calls.

        When capture_scopes is True, walks the Python call stack to extract
        user function names, combined with explicit scope names for
        mem()/DPmem() boundaries (replacing <lambda> frames).
        """
        if not self._capture_scopes:
            return tuple(self._scope_stack)

        path: list[str] = []
        scope_items = list(self._scope_stack)

        frame = sys._getframe()
        while frame is not None:
            func_name = frame.f_code.co_name
            module = frame.f_globals.get("__name__", "")

            if func_name == "run_with_trace":
                break

            if module.startswith("cathedral"):
                # Emit the explicit scope name for mem/DPmem wrappers
                if func_name in ("memoized", "dp_memoized") and scope_items:
                    path.append(scope_items.pop())
            elif func_name == "<lambda>":
                pass
            elif not func_name.startswith("<"):
                path.append(func_name)

            frame = frame.f_back

        path.reverse()
        return tuple(path)

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
    fn: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    interventions: dict[str, Any] | None = None,
    enumerate_mode: bool = False,
    capture_scopes: bool | None = None,
) -> Trace:
    """Execute a function within a fresh tracing context and return the trace.

    Args:
        fn: The model function to execute.
        args: Positional arguments to pass to fn.
        kwargs: Keyword arguments to pass to fn.
        interventions: Optional dict mapping addresses to values to intervene on.
        enumerate_mode: If True, sample() raises NeedsEnumeration for
            un-intervened discrete sites instead of sampling randomly.
        capture_scopes: If True, record scope paths on each choice via
            Python stack introspection. If None, uses the value set by
            infer(capture_scopes=...) or defaults to False.

    Returns:
        The completed Trace.

    Raises:
        Rejected: If a condition() call fails during execution.
        NeedsEnumeration: If enumerate_mode is True and a discrete site needs expansion.
    """
    if kwargs is None:
        kwargs = {}

    should_capture = capture_scopes if capture_scopes is not None else _CAPTURE_SCOPES.get(False)
    ctx = TraceContext(
        interventions=interventions,
        enumerate_mode=enumerate_mode,
        capture_scopes=should_capture,
    )
    token = _TRACE.set(ctx)
    try:
        result = fn(*args, **kwargs)
        ctx.trace.result = result
        return ctx.trace
    finally:
        _TRACE.reset(token)
