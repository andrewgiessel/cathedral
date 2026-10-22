"""Sequential multi-chain inference and MCMC diagnostics.

This module deliberately depends only on the public :func:`cathedral.model.infer`
API.  Each chain is an ordinary ``Posterior``; the container retains the chain
and draw dimensions instead of concatenating chains before diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.special import ndtri  # type: ignore[import-untyped]
from scipy.stats import rankdata  # type: ignore[import-untyped]

from cathedral.model import Posterior, infer


@dataclass(frozen=True)
class ChainDiagnostics:
    """Diagnostics for one scalar estimand.

    A non-``"ok"`` status means numerical diagnostic values must not be used
    for convergence decisions.  ``constant`` is reported separately because
    empirical constancy cannot establish that a stochastic estimator has zero
    Monte Carlo error; all mixing and MCSE quantities are undefined.
    """

    rhat: float
    ess_bulk: float
    ess_tail: float
    ess_mean: float
    mcse_mean: float
    status: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EstimandSummary:
    """Summary of a named scalar query evaluated at every retained draw."""

    name: str
    mean: float
    diagnostics: ChainDiagnostics

    @property
    def mcse_mean(self) -> float:
        return self.diagnostics.mcse_mean


class ChainPosterior:
    """Independent posterior chains produced by :func:`infer_chains`.

    ``chains`` preserves individual ``Posterior`` objects.  ``values(query)``
    returns an array with leading ``(chain, draw)`` axes and any scalar/vector
    query shape after those axes.  MCMC diagnostics are defined only for a
    scalar query; summarize each component separately for vector estimands.
    """

    def __init__(self, chains: tuple[Posterior, ...], seeds: tuple[np.random.SeedSequence, ...]):
        if len(chains) < 2:
            raise ValueError("ChainPosterior requires at least two chains.")
        if len({chain.num_samples for chain in chains}) != 1:
            raise ValueError("All chains must retain the same number of draws.")
        self.chains = chains
        self.seeds = seeds

    @property
    def num_chains(self) -> int:
        return len(self.chains)

    @property
    def num_samples(self) -> int:
        return self.chains[0].num_samples

    def values(self, query: Callable[[Any], Any]) -> np.ndarray:
        """Evaluate *query* on model return values with ``(chain, draw)`` axes."""
        if not callable(query):
            raise TypeError("query must be callable")
        rows = [np.asarray([query(value) for value in chain.samples]) for chain in self.chains]
        try:
            values = np.stack(rows, axis=0)
        except ValueError as exc:
            raise ValueError("query must return a fixed shape for every draw and chain") from exc
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise TypeError("query must return real numeric values")
        return values

    def summarize(
        self, query: Callable[[Any], Any] | Mapping[str, Callable[[Any], Any]], name: str | None = None
    ) -> EstimandSummary | dict[str, EstimandSummary]:
        """Summarize one query or a mapping of named queries.

        A callable query requires ``name`` (or uses its ``__name__`` when it is
        informative).  Mapping keys are the reported estimand names.
        """
        if isinstance(query, Mapping):
            return {key: self._summarize_one(key, fn) for key, fn in query.items()}
        query_name = name or getattr(query, "__name__", None)
        if not query_name or query_name == "<lambda>":
            raise ValueError("Supply name= for a lambda or unnamed query.")
        return self._summarize_one(query_name, query)

    def _summarize_one(self, name: str, query: Callable[[Any], Any]) -> EstimandSummary:
        values = self.values(query)
        if values.ndim != 2:
            raise ValueError("MCMC diagnostics require a scalar query; summarize vector components separately.")
        diagnostics = diagnose_chains(values)
        return EstimandSummary(name=name, mean=float(np.mean(values)), diagnostics=diagnostics)


def infer_chains(
    model: Callable[..., Any],
    *args: Any,
    num_chains: int = 4,
    num_samples: int = 1000,
    warmup: int | None = None,
    blocks: Any = None,
    seed: int | np.integer | np.random.SeedSequence | None = None,
    **kwargs: Any,
) -> ChainPosterior:
    """Run independent sequential adaptive-MH chains using spawned RNG streams.

    This is intentionally sequential: it is safe for models and kernels that
    are not process-picklable.  ``seed`` is expanded through ``SeedSequence``;
    no chain receives an adjacent integer seed.  Extra keyword arguments are
    passed to public ``infer(..., method='adaptive_mh')`` (for example ``lag``).
    """
    if num_chains < 2:
        raise ValueError("num_chains must be at least 2 for between-chain diagnostics.")
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if isinstance(seed, np.random.SeedSequence):
        root = seed
    else:
        root = np.random.SeedSequence(seed)
    child_seeds = tuple(root.spawn(num_chains))
    chains = tuple(
        infer(
            model,
            *args,
            method="adaptive_mh",
            num_samples=num_samples,
            warmup=warmup,
            blocks=blocks,
            seed=child_seed,
            **kwargs,
        )
        for child_seed in child_seeds
    )
    return ChainPosterior(chains, child_seeds)


def diagnose_chains(samples: np.ndarray) -> ChainDiagnostics:
    """Compute rank-normalized split/folded R-hat and MCMC ESS diagnostics.

    Uses the split-chain, initial-positive-sequence estimator described by
    Vehtari et al. (2021), with FFT autocovariances.  This is MCMC ESS, not an
    importance-weight ESS.  Input must have shape ``(chains, draws)``.
    """
    if np.iscomplexobj(samples):
        raise TypeError("diagnostics require real-valued samples")
    x = np.asarray(samples, dtype=float)
    warnings: list[str] = []
    if x.ndim != 2:
        raise ValueError("samples must have shape (chains, draws)")
    chains, draws = x.shape
    if chains < 2 or draws < 4:
        warnings.append("rank-normalized split diagnostics require at least 2 chains and 4 draws per chain")
        return _invalid(warnings)
    if not np.all(np.isfinite(x)):
        warnings.append("samples contain NaN or infinity")
        return _invalid(warnings)
    if np.ptp(x) == 0:
        warnings.append("estimand is empirically constant; mixing diagnostics and mean MCSE are undefined")
        return ChainDiagnostics(np.nan, np.nan, np.nan, np.nan, np.nan, "constant", tuple(warnings))

    split = _split_chains(x)
    if split.shape[1] < 2:
        warnings.append("too few draws remain after chain splitting")
        return _invalid(warnings)
    ranked = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split)))
    rhat_bulk = _rhat(ranked)
    rhat_folded = _rhat(folded)
    # Do not use Python's max here: max(finite, nan) masks an invalid folded
    # diagnostic.  Rank R-hat is defined as the maximum only when both are.
    rhat = max(rhat_bulk, rhat_folded) if np.isfinite(rhat_bulk) and np.isfinite(rhat_folded) else np.nan
    ess_bulk = _ess(ranked)
    low, high = np.quantile(x, (0.05, 0.95))
    ess_tail = min(_ess(_split_chains((x <= low).astype(float))), _ess(_split_chains((x >= high).astype(float))))
    ess_mean = _ess(split)
    mcse = float(np.std(x, ddof=1) / np.sqrt(ess_mean)) if np.isfinite(ess_mean) and ess_mean > 0 else np.nan
    if not all(np.isfinite(v) for v in (rhat, ess_bulk, ess_tail, ess_mean, mcse)):
        warnings.append("autocorrelation estimate was undefined")
        return ChainDiagnostics(rhat, ess_bulk, ess_tail, ess_mean, mcse, "invalid", tuple(warnings))
    return ChainDiagnostics(rhat, ess_bulk, ess_tail, ess_mean, mcse, "ok", tuple(warnings))


def _invalid(warnings: list[str]) -> ChainDiagnostics:
    return ChainDiagnostics(np.nan, np.nan, np.nan, np.nan, np.nan, "invalid", tuple(warnings))


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Split chains into their first and last halves.

    For odd lengths the middle draw is intentionally omitted.  Taking the
    final half from ``-half:`` (rather than from the trimmed prefix) matches
    the split-chain definition used by Vehtari et al. and ArviZ.
    """
    half = x.shape[1] // 2
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x, method="average").reshape(x.shape)
    size = x.size
    return cast(np.ndarray, ndtri((ranks - 3.0 / 8.0) / (size + 1.0 / 4.0)))


def _rhat(x: np.ndarray) -> float:
    chains, draws = x.shape
    within = float(np.mean(np.var(x, axis=1, ddof=1)))
    if within <= 0 or not np.isfinite(within):
        return np.nan
    between = draws * float(np.var(np.mean(x, axis=1), ddof=1))
    return float(np.sqrt((((draws - 1) / draws) * within + between / draws) / within))


def _var_plus(x: np.ndarray) -> float:
    chains, draws = x.shape
    within = float(np.mean(np.var(x, axis=1, ddof=1)))
    between = draws * float(np.var(np.mean(x, axis=1), ddof=1))
    return float(((draws - 1) / draws) * within + between / draws)


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance for one chain, evaluated by FFT."""
    centered = x - np.mean(x)
    n = len(centered)
    fft = np.fft.rfft(centered, n=2 * n)
    return np.fft.irfft(fft * np.conjugate(fft), n=2 * n)[:n] / n


def _ess(x: np.ndarray) -> float:
    """Vehtari/Geyer bulk ESS, equivalent to ArviZ's internal estimator.

    The lower bound on tau is deliberate: negatively autocorrelated chains
    can have an ESS larger than the number of retained draws.
    """
    chains, draws = x.shape
    if draws < 2:
        return np.nan
    acov = np.stack([_autocovariance(chain) for chain in x])
    mean_var = float(np.mean(acov[:, 0]) * draws / (draws - 1.0))
    var_plus = mean_var * (draws - 1.0) / draws
    if chains > 1:
        var_plus += float(np.var(np.mean(x, axis=1), ddof=1))
    if var_plus <= 0 or not np.isfinite(var_plus):
        return np.nan

    # rho[0] is exactly one by definition, not an estimate subject to finite
    # sample variance or between-chain adjustment.
    rho = np.zeros(draws)
    rho[0] = 1.0
    rho_even = 1.0
    rho_odd = 1.0 - (mean_var - float(np.mean(acov[:, 1]))) / var_plus
    rho[1] = rho_odd
    lag = 1
    while lag < draws - 3 and rho_even + rho_odd > 0.0:
        rho_even = 1.0 - (mean_var - float(np.mean(acov[:, lag + 1]))) / var_plus
        rho_odd = 1.0 - (mean_var - float(np.mean(acov[:, lag + 2]))) / var_plus
        if rho_even + rho_odd >= 0.0:
            rho[lag + 1] = rho_even
            rho[lag + 2] = rho_odd
        lag += 2
    max_lag = lag - 2
    if rho_even > 0.0:
        rho[max_lag + 1] = rho_even

    # Geyer's initial monotone sequence on paired autocorrelations.
    lag = 1
    while lag <= max_lag - 2:
        if rho[lag + 1] + rho[lag + 2] > rho[lag - 1] + rho[lag]:
            rho[lag + 1] = (rho[lag - 1] + rho[lag]) / 2.0
            rho[lag + 2] = rho[lag + 1]
        lag += 2

    tau = -1.0 + 2.0 * float(np.sum(rho[: max_lag + 1])) + rho[max_lag + 1]
    tau = max(tau, 1.0 / np.log10(chains * draws))
    return float(chains * draws / tau) if np.isfinite(tau) else np.nan


__all__ = ["ChainDiagnostics", "ChainPosterior", "EstimandSummary", "diagnose_chains", "infer_chains"]
