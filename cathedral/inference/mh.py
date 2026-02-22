"""Single-site Metropolis-Hastings inference engine.

Explores the posterior by repeatedly proposing changes to one random
choice at a time. Each step picks a site uniformly at random, proposes
a new value from the prior, re-executes the model with the new value
(replaying all other sites), and accepts or rejects via the MH ratio.

Handles structural changes: when the proposed value alters control flow,
some choice sites may appear or disappear. The acceptance ratio accounts
for this via the Wingate et al. (2011) correction.

Best for: models too complex for rejection/importance -- rare conditions,
many latent variables, or models where you want correlated posterior samples.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from cathedral.trace import Rejected, Trace, run_with_trace


def mh_sample(
    model_fn: Callable,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    num_samples: int = 1000,
    burn_in: int | None = None,
    lag: int = 1,
    max_init_attempts: int = 10000,
) -> list[Trace]:
    """Run single-site Metropolis-Hastings on a model.

    Args:
        model_fn: The model function to sample from.
        args: Positional arguments to pass to the model.
        kwargs: Keyword arguments to pass to the model.
        num_samples: Number of posterior samples to return.
        burn_in: Steps to discard before collecting. Defaults to num_samples // 2.
        lag: Keep every nth sample after burn-in (thinning). Default 1 (keep all).
        max_init_attempts: Max forward-sampling tries to find a valid initial trace.

    Returns:
        A list of Traces from the posterior.

    Raises:
        RuntimeError: If no valid initial trace can be found.
    """
    if kwargs is None:
        kwargs = {}
    if burn_in is None:
        burn_in = num_samples // 2

    current = _get_initial_trace(model_fn, args, kwargs, max_init_attempts)

    total_steps = burn_in + num_samples * lag
    traces: list[Trace] = []

    for step in range(total_steps):
        current = _mh_step(model_fn, args, kwargs, current)
        if step >= burn_in and (step - burn_in) % lag == 0:
            traces.append(current)

    return traces


def _get_initial_trace(
    model_fn: Callable,
    args: tuple,
    kwargs: dict[str, Any],
    max_attempts: int,
) -> Trace:
    """Get an initial valid trace by forward sampling."""
    for _ in range(max_attempts):
        try:
            return run_with_trace(model_fn, args=args, kwargs=kwargs)
        except Rejected:
            continue
    raise RuntimeError(
        f"MH: could not find a valid initial trace after {max_attempts} attempts. "
        f"The model's conditions may be too restrictive."
    )


def _mh_step(
    model_fn: Callable,
    args: tuple,
    kwargs: dict[str, Any],
    current: Trace,
) -> Trace:
    """Perform one single-site MH step.

    Picks a random choice, proposes a new value from its prior,
    re-executes the model, and accepts/rejects.
    """
    addresses = current.addresses
    if not addresses:
        return current

    selected = addresses[np.random.randint(len(addresses))]

    interventions = {addr: choice.value for addr, choice in current.choices.items() if addr != selected}

    try:
        proposed = run_with_trace(model_fn, args=args, kwargs=kwargs, interventions=interventions)
    except Rejected:
        return current

    if selected not in proposed.choices:
        return current

    log_alpha = _log_acceptance_ratio(current, proposed, selected)

    if log_alpha >= 0 or math.log(np.random.random()) < log_alpha:
        return proposed
    return current


def _log_acceptance_ratio(old: Trace, new: Trace, selected: str) -> float:
    """Compute the log MH acceptance ratio for a single-site prior proposal.

    The full ratio for proposing from the prior at site k is:

        log a = (log_joint_new - log_joint_old)
              + log(|K_old| / |K_new|)
              + old_k.log_prob - new_k.log_prob
              + Σ_disappeared old_j.log_prob - Σ_new new_j.log_prob

    The prior-proposal terms (selected site + structural sites) cancel with
    corresponding terms in the joint, leaving only the effect of the change
    on shared sites and the log-score. But we compute via the explicit form
    since it avoids needing to identify shared sites.
    """
    if math.isinf(new.log_joint) and new.log_joint < 0:
        return float("-inf")

    old_addrs = set(old.choices.keys())
    new_addrs = set(new.choices.keys())

    k_old = len(old_addrs)
    k_new = len(new_addrs)

    if k_old == 0 or k_new == 0:
        return float("-inf")

    log_joint_ratio = new.log_joint - old.log_joint
    log_size_correction = math.log(k_old) - math.log(k_new)

    # Prior proposal cancellation for the selected site
    log_selected_correction = old.choices[selected].log_prob - new.choices[selected].log_prob

    # Structural change correction: new sites were sampled from prior in the
    # forward move; disappeared sites would need to be sampled in the reverse move.
    appeared = new_addrs - old_addrs
    disappeared = old_addrs - new_addrs

    log_structural = sum(old.choices[a].log_prob for a in disappeared) - sum(new.choices[a].log_prob for a in appeared)

    return log_joint_ratio + log_size_correction + log_selected_correction + log_structural
