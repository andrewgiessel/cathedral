"""Matplotlib-based diagnostic plots for Cathedral.

All functions in this module require matplotlib. If matplotlib is not
installed, importing this module will raise ImportError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "cathedral.plots requires matplotlib. Install it with: pip install cathedral[viz]"
    ) from e

if TYPE_CHECKING:
    from cathedral.model import Posterior


def plot_posterior(
    posterior: Posterior,
    key: str | None = None,
    *,
    bins: int = 30,
    kind: str = "auto",
    ax: Any = None,
) -> Figure:
    """Plot the posterior distribution of return values.

    Args:
        posterior: A Posterior from inference.
        key: If results are dicts, plot this key's values.
        bins: Number of histogram bins.
        kind: Plot type — "hist", "kde", or "auto" (chooses based on data).
        ax: Optional matplotlib Axes to plot on.

    Returns:
        The matplotlib Figure.
    """
    values = posterior._extract_values(key)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    is_numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in values)
    is_bool = all(isinstance(v, (bool, np.bool_)) for v in values)

    if is_bool or (not is_numeric):
        hist = posterior.histogram(key)
        labels = [str(k) for k in hist.keys()]
        probs = list(hist.values())
        ax.bar(labels, probs, color="steelblue", edgecolor="white")
        ax.set_ylabel("Probability")
    else:
        arr = np.array(values, dtype=float)
        if kind == "auto":
            kind = "hist"
        if kind == "hist":
            ax.hist(arr, bins=bins, density=True, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_ylabel("Density")
        elif kind == "kde":
            from scipy.stats import gaussian_kde

            xs = np.linspace(arr.min(), arr.max(), 200)
            kde = gaussian_kde(arr)
            ax.plot(xs, kde(xs), color="steelblue", linewidth=2)
            ax.fill_between(xs, kde(xs), alpha=0.3, color="steelblue")
            ax.set_ylabel("Density")

    title = "Posterior"
    if key:
        title += f" [{key}]"
    if posterior.info:
        title += f" ({posterior.info.method})"
    ax.set_title(title)
    ax.set_xlabel("Value")

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_weights(posterior: Posterior, *, ax: Any = None) -> Figure:
    """Plot the importance weight distribution from an importance sampling run.

    Args:
        posterior: A Posterior from importance sampling inference.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib Figure.
    """
    if posterior.info is None or posterior.info.log_weights is None:
        raise ValueError("plot_weights requires a Posterior with log_weights (use method='importance')")

    log_w = posterior.info.log_weights
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()

    ax.hist(weights, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Normalized Weight")
    ax.set_ylabel("Count")
    ax.set_title(f"Importance Weights (ESS={posterior.info.ess:.1f}/{len(log_w)})" if posterior.info.ess else "Importance Weights")
    ax.axvline(1 / len(log_w), color="red", linestyle="--", alpha=0.7, label="Uniform")
    ax.legend()

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_trace_values(
    posterior: Posterior,
    address: str,
    *,
    ax: Any = None,
) -> Figure:
    """Plot the value of a specific choice address across samples (trace plot).

    Useful for diagnosing mixing in MH sampling.

    Args:
        posterior: A Posterior from inference.
        address: The choice address to plot.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib Figure.
    """
    values = []
    for t in posterior.traces:
        if address in t.choices:
            values.append(t.choices[address].value)
        else:
            values.append(np.nan)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(values, linewidth=0.5, color="steelblue", alpha=0.8)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel(f"Value of '{address}'")
    ax.set_title(f"Trace Plot: {address}")

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_ess(posterior: Posterior, *, ax: Any = None) -> Figure:
    """Plot effective sample size per address for fixed-structure posteriors.

    Computes ESS using the autocorrelation-based method for each address.

    Args:
        posterior: A Posterior with fixed structure.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib Figure.

    Raises:
        ValueError: If the posterior has variable structure.
    """
    if not posterior.has_fixed_structure:
        raise ValueError("plot_ess requires a fixed-structure posterior")

    traces = posterior.traces
    if not traces:
        raise ValueError("No traces in posterior")

    addresses = list(traces[0].choices.keys())
    ess_values: dict[str, float] = {}

    for addr in addresses:
        vals = [t.choices[addr].value for t in traces]
        is_numeric = all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in vals)
        if not is_numeric:
            continue
        arr = np.array(vals, dtype=float)
        ess_values[addr] = _compute_ess(arr)

    if not ess_values:
        raise ValueError("No numeric addresses found for ESS computation")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(ess_values) * 0.8), 5))
    else:
        fig = ax.get_figure()

    names = list(ess_values.keys())
    ess_vals = [ess_values[n] for n in names]

    bars = ax.bar(range(len(names)), ess_vals, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("ESS")
    ax.set_title(f"Effective Sample Size (n={len(traces)})")
    ax.axhline(len(traces), color="red", linestyle="--", alpha=0.5, label=f"n={len(traces)}")
    ax.legend()

    if fig is not None:
        fig.tight_layout()
    return fig


def _compute_ess(x: np.ndarray) -> float:
    """Compute effective sample size via initial positive sequence estimator."""
    n = len(x)
    if n < 4:
        return float(n)

    x = x - np.mean(x)
    var = np.var(x, ddof=0)
    if var == 0:
        return float(n)

    max_lag = n // 2
    acf = np.correlate(x, x, mode="full")[n - 1 :] / (var * n)

    tau = 1.0
    for lag in range(1, max_lag):
        rho = acf[lag] if lag < len(acf) else 0.0
        if rho < 0.05:
            break
        tau += 2 * rho

    return n / tau
