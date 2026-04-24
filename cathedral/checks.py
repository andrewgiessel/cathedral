"""Model checking and comparison utilities for Cathedral.

Provides prior/posterior predictive checks, conditioning difficulty
analysis, and model comparison via marginal likelihood.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from cathedral._rng import SeedLike, make_rng
from cathedral.model import Posterior
from cathedral.trace import Rejected, Trace, run_with_trace


def prior_predictive(
    model_fn: Callable,
    *args: Any,
    num_samples: int = 1000,
    seed: SeedLike = None,
) -> Posterior:
    """Run a model forward without conditioning to examine the prior predictive.

    Executes the model, catching and discarding any Rejected exceptions
    from condition() calls. This shows what the model generates before
    any conditioning is applied.

    Args:
        model_fn: A @model-decorated function or plain callable.
        *args: Arguments to pass to the model function.
        num_samples: Number of forward samples to collect.
        seed: Optional seed for reproducible forward sampling.

    Returns:
        A Posterior containing unconditioned forward samples.
    """
    from cathedral.model import InferenceInfo

    fn = getattr(model_fn, "_original_fn", model_fn)
    rng = make_rng(seed)
    traces: list[Trace] = []
    rejected = 0

    for _ in range(num_samples):
        try:
            trace = run_with_trace(fn, args=args, rng=rng)
            traces.append(trace)
        except Rejected:
            rejected += 1

    if not traces:
        raise RuntimeError(
            "prior_predictive: all forward samples were rejected. This model may unconditionally call condition(False)."
        )

    info = InferenceInfo(
        method="prior_predictive",
        num_samples=len(traces),
        num_attempts=num_samples,
        acceptance_rate=len(traces) / num_samples,
    )
    return Posterior(traces, info=info)


def condition_acceptance_rate(
    model_fn: Callable,
    *args: Any,
    num_samples: int = 10000,
    seed: SeedLike = None,
) -> float:
    """Estimate what fraction of prior samples satisfy the model's conditions.

    A low acceptance rate means the conditioning is very restrictive and
    rejection sampling will be slow. Consider using importance sampling
    or MH instead.

    Args:
        model_fn: A @model-decorated function or plain callable.
        *args: Arguments to pass to the model function.
        num_samples: Number of forward samples to attempt.
        seed: Optional seed for reproducible acceptance-rate estimates.

    Returns:
        Fraction of samples that were not rejected (0.0 to 1.0).
    """
    fn = getattr(model_fn, "_original_fn", model_fn)
    rng = make_rng(seed)
    accepted = 0

    for _ in range(num_samples):
        try:
            run_with_trace(fn, args=args, rng=rng)
            accepted += 1
        except Rejected:
            pass

    return accepted / num_samples


def posterior_predictive(
    posterior: Posterior,
    model_fn: Callable,
    *args: Any,
    num_samples: int | None = None,
    seed: SeedLike = None,
) -> Posterior:
    """Generate posterior predictive samples by replaying the model.

    For each posterior trace, re-executes the model with the trace's
    latent variable values (via interventions), allowing new random
    choices to be drawn where the original trace had observed data.

    Args:
        posterior: A Posterior from a previous inference run.
        model_fn: The same model function used for inference.
        *args: Arguments to pass to the model function.
        num_samples: Number of predictive samples. Defaults to
            min(posterior.num_samples, 500).
        seed: Optional seed for reproducible posterior predictive draws.

    Returns:
        A Posterior containing posterior predictive samples.
    """
    from cathedral.model import InferenceInfo

    fn = getattr(model_fn, "_original_fn", model_fn)

    if num_samples is None:
        num_samples = min(posterior.num_samples, 500)

    rng = make_rng(seed)
    source_traces = posterior.traces
    indices = rng.choice(len(source_traces), size=num_samples, replace=True)

    traces: list[Trace] = []
    for idx in indices:
        source = source_traces[idx]
        interventions = {addr: choice.value for addr, choice in source.choices.items()}
        try:
            trace = run_with_trace(fn, args=args, interventions=interventions, rng=rng)
            traces.append(trace)
        except Rejected:
            traces.append(source)

    info = InferenceInfo(
        method="posterior_predictive",
        num_samples=len(traces),
    )
    return Posterior(traces, info=info)


def compare_models(posteriors: dict[str, Posterior]) -> str:
    """Compare multiple models via log marginal likelihood and Bayes factors.

    Args:
        posteriors: Dict mapping model names to Posterior objects.
            Each Posterior should have been obtained via infer() so that
            InferenceInfo with log_marginal_likelihood is available.

    Returns:
        A formatted string summarizing the comparison.
    """
    lines: list[str] = ["Model Comparison"]
    lines.append("=" * 50)

    lmls: dict[str, float] = {}
    for name, post in posteriors.items():
        lml = post.log_marginal_likelihood
        if lml is not None:
            lmls[name] = lml
            lines.append(f"  {name}: log ML = {lml:.4f}")
        else:
            lines.append(f"  {name}: log ML = not available (try importance or enumerate)")

    if len(lmls) >= 2:
        lines.append("")
        lines.append("Bayes Factors (log scale):")
        names = list(lmls.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                log_bf = lmls[a] - lmls[b]
                bf = math.exp(min(log_bf, 700))
                lines.append(f"  {a} vs {b}: log BF = {log_bf:.4f} (BF = {bf:.4f})")
                if log_bf > 1:
                    lines.append(f"    -> Strong evidence for {a}")
                elif log_bf > 0:
                    lines.append(f"    -> Evidence for {a}")
                elif log_bf > -1:
                    lines.append(f"    -> Evidence for {b}")
                else:
                    lines.append(f"    -> Strong evidence for {b}")

    return "\n".join(lines)
