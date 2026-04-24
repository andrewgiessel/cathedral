"""Internal random number generation helpers for Cathedral.

Centralizes Generator creation and the active RNG context used during
traced execution so inference can be reproducible without relying on
global ``numpy.random`` state.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

Generator = np.random.Generator
SeedLike = int | np.integer | np.random.SeedSequence | None

_ACTIVE_RNG: ContextVar[Generator | None] = ContextVar("cathedral_active_rng", default=None)
_DEFAULT_RNG = np.random.default_rng()


def make_rng(seed: SeedLike = None) -> Generator:
    """Create a NumPy Generator from an optional seed-like input."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def get_active_rng() -> Generator:
    """Return the active RNG if present, otherwise the process-local default."""
    rng = _ACTIVE_RNG.get()
    if rng is not None:
        return rng
    return _DEFAULT_RNG


@contextmanager
def using_rng(rng: Generator) -> Iterator[Generator]:
    """Temporarily install *rng* as the active Cathedral RNG."""
    token = _ACTIVE_RNG.set(rng)
    try:
        yield rng
    finally:
        _ACTIVE_RNG.reset(token)
