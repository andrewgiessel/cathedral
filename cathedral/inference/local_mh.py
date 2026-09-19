"""Restricted, replay-safe adaptive local Metropolis-Hastings kernels."""

from __future__ import annotations

import copy
import math
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cathedral._rng import Generator, SeedLike, make_rng
from cathedral.distributions import (
    Beta,
    Binomial,
    Exponential,
    Gamma,
    Geometric,
    HalfNormal,
    LogNormal,
    Normal,
    Poisson,
    Uniform,
)
from cathedral.inference.mh import _get_initial_trace
from cathedral.trace import Choice, Rejected, Trace, run_with_trace


@dataclass(frozen=True)
class LocalKernel:
    """A scalar or user-declared block selected by the compound schedule."""

    addresses: tuple[str, ...]

    @property
    def name(self) -> str:
        """Unambiguous display name; state uses ``addresses`` directly as its key."""
        return repr(self.addresses)


@dataclass
class _KernelState:
    log_scale: float = 0.0
    visits: int = 0
    accepted: int = 0
    total_jump_distance: float = 0.0
    adaptation_visits: int = 0
    mean: np.ndarray | None = None
    scatter: np.ndarray | None = None


@dataclass
class LocalMHSamplerState:
    """Serializable state needed to resume an adaptive local-MH chain exactly."""

    trace: Trace
    rng_state: dict[str, Any]
    kernel_states: dict[tuple[str, ...], _KernelState]
    kernels: tuple[LocalKernel, ...]
    warmup: int
    step: int = 0


def local_mh_sample(  # noqa: C901
    model_fn: Callable, args: tuple = (), kwargs: dict[str, Any] | None = None,
    num_samples: int = 1000, *, warmup: int | None = None, lag: int = 1,
    blocks: Sequence[Sequence[str]] | None = None, initial_trace: Trace | None = None,
    sampler_state: LocalMHSamplerState | None = None, max_init_attempts: int = 10000,
    seed: SeedLike = None, _info: dict | None = None,
) -> tuple[list[Trace], LocalMHSamplerState]:
    """Sample on the explicitly named, scalar, fixed-structure subset.

    Adaptation uses every attempted transition during warmup and freezes both
    scale and covariance after exactly ``warmup`` transitions.
    """
    if kwargs is None:
        kwargs = {}
    if num_samples < 1 or lag < 1:
        raise ValueError("num_samples and lag must be positive")
    if sampler_state is not None and (initial_trace is not None or blocks is not None):
        raise ValueError("sampler_state cannot be combined with initial_trace or blocks")
    if sampler_state is not None and seed is not None:
        raise ValueError("seed cannot be supplied when resuming adaptive_mh; sampler state owns the RNG")

    if sampler_state is None:
        rng = make_rng(seed)
        current = initial_trace or _get_initial_trace(model_fn, args, kwargs, max_init_attempts, rng)
        # Validate with a clone: validation must not advance the chain RNG.
        validation_rng = make_rng()
        validation_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        current = _validate_initial_trace(model_fn, args, kwargs, current, validation_rng)
        kernels = _make_kernels(current, blocks)
        state = LocalMHSamplerState(
            trace=current, rng_state=copy.deepcopy(rng.bit_generator.state),
            kernel_states={kernel.addresses: _KernelState() for kernel in kernels},
            kernels=kernels, warmup=num_samples // 2 if warmup is None else warmup,
        )
    else:
        state = copy.deepcopy(sampler_state)
        rng = make_rng()
        rng.bit_generator.state = copy.deepcopy(state.rng_state)
        validation_rng = make_rng()
        validation_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        current = _validate_initial_trace(model_fn, args, kwargs, state.trace, validation_rng)
        state.trace = current

    if state.warmup < 0:
        raise ValueError("warmup must be non-negative")
    accepted = 0
    jump_distance = 0.0
    traces: list[Trace] = []
    initial_warmup = state.warmup if state.step == 0 else 0
    total_steps = initial_warmup + num_samples * lag
    for _ in range(total_steps):
        kernel = state.kernels[rng.integers(len(state.kernels))]
        current, did_accept, jump = _step(
            model_fn, args, kwargs, current, kernel, state.kernel_states[kernel.addresses], rng
        )
        accepted += int(did_accept)
        jump_distance += jump
        if state.step < state.warmup:
            _adapt(state.kernel_states[kernel.addresses], current, kernel, did_accept)
        state.step += 1
        if state.step > state.warmup and (state.step - state.warmup - 1) % lag == 0:
            traces.append(current)

    state.trace = current
    state.rng_state = copy.deepcopy(rng.bit_generator.state)
    if _info is not None:
        _info.update({
            "total_steps": total_steps, "warmup": state.warmup, "lag": lag,
            "acceptance_rate": accepted / total_steps,
            "mean_squared_jump_distance": jump_distance / total_steps,
            "kernel_diagnostics": _kernel_diagnostics(state),
            "adaptation_frozen": state.step >= state.warmup, "sampler_state": state,
        })
    return traces, state


def _same_value(left: Any, right: Any) -> bool:
    """Equality suitable for deterministic replay return values."""
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return left == right


def _metadata(choice: Choice) -> tuple[Any, ...]:
    """Replay-invariant distribution metadata relevant to a proposal transform.

    Location and scale of unbounded continuous distributions may depend on
    held choices and are deliberately rescored rather than frozen. Families,
    scalar shapes, and transform bounds must stay stable for this sampler.
    """
    dist = choice.distribution
    if isinstance(dist, Uniform):
        return (type(dist), dist.low, dist.high)
    if isinstance(dist, Beta):
        return (type(dist), 0.0, 1.0, dist.a, dist.b)
    if isinstance(dist, Normal):
        return (type(dist),)
    if isinstance(dist, HalfNormal | Exponential | LogNormal):
        return (type(dist),)
    if isinstance(dist, Gamma):
        return (type(dist), dist.shape)
    if isinstance(dist, Poisson | Geometric):
        return (type(dist),)
    if isinstance(dist, Binomial):
        return (type(dist), dist.n)
    return (type(dist),)


def _rng_state(rng: Generator) -> bytes:
    """Return a comparable RNG snapshot without advancing it."""
    return pickle.dumps(rng.bit_generator.state, protocol=pickle.HIGHEST_PROTOCOL)


def _replay_proposal(
    model_fn: Callable, args: tuple, kwargs: dict[str, Any], interventions: dict[str, Any], rng: Generator,
) -> Trace:
    """Replay one proposal and reject models whose replay consumes RNG."""
    before = _rng_state(rng)
    try:
        replayed = run_with_trace(model_fn, args=args, kwargs=kwargs, interventions=interventions, rng=rng)
    except Rejected:
        if _rng_state(rng) != before:
            raise ValueError("adaptive_mh does not support proposal replay that consumes RNG") from None
        raise
    if _rng_state(rng) != before:
        raise ValueError("adaptive_mh does not support proposal replay that consumes RNG")
    return replayed


def _validate_initial_trace(  # noqa: C901
    model_fn: Callable, args: tuple, kwargs: dict[str, Any], trace: Trace, rng: Generator
) -> Trace:
    if not trace.choices:
        raise ValueError("adaptive_mh requires at least one named random choice")
    unnamed = [address for address, choice in trace.choices.items() if not choice.explicit_name]
    if unnamed:
        raise ValueError(f"adaptive_mh requires explicit unique names; unnamed addresses: {unnamed!r}")
    interventions = {address: choice.value for address, choice in trace.choices.items()}
    try:
        replayed = _replay_proposal(model_fn, args, kwargs, interventions, rng)
    except Rejected as e:
        raise ValueError("initial_trace is not a valid execution of this model") from e
    if set(replayed.choices) != set(trace.choices):
        raise ValueError("adaptive_mh requires replay-safe fixed structure; replay changed choice addresses")
    if not _same_value(replayed.result, trace.result):
        raise ValueError("initial_trace result does not match deterministic replay")
    for address, replay_choice in replayed.choices.items():
        old_choice = trace.choices[address]
        if _metadata(replay_choice) != _metadata(old_choice):
            raise ValueError(f"initial_trace distribution metadata changed at {address!r}")
        if not math.isfinite(replay_choice.log_prob):
            raise ValueError(f"initial_trace value at {address!r} is outside replayed distribution support")
    if not math.isfinite(replayed.log_score) or not math.isfinite(replayed.log_joint):
        raise ValueError("initial_trace has non-finite replayed score")
    if not math.isclose(replayed.log_score, trace.log_score, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("initial_trace score does not match replayed model")
    if not math.isclose(replayed.log_joint, trace.log_joint, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("initial_trace joint score does not match replayed model")
    return replayed


def _make_kernels(trace: Trace, blocks: Sequence[Sequence[str]] | None) -> tuple[LocalKernel, ...]:
    seen: set[str] = set()
    kernels: list[LocalKernel] = []
    for requested in blocks or ():
        addresses = tuple(requested)
        if len(addresses) < 2 or len(set(addresses)) != len(addresses):
            raise ValueError("each adaptive_mh block must contain at least two distinct addresses")
        if set(addresses) - set(trace.choices) or seen.intersection(addresses):
            raise ValueError(f"invalid or overlapping adaptive_mh block addresses: {addresses!r}")
        for address in addresses:
            _transform(trace.choices[address])
        kernels.append(LocalKernel(addresses))
        seen.update(addresses)
    for address, choice in trace.choices.items():
        if address not in seen:
            _transform(choice)
            kernels.append(LocalKernel((address,)))
    return tuple(kernels)


def _expit(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _transform(choice: Choice) -> tuple[float, Callable[[float], Any], Callable[[Any], float], str]:
    """Return z, inverse z->x, log|dz/dx|, and proposal family."""
    value = choice.value
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"adaptive_mh supports scalar numeric choices only; {choice.address!r} is unsupported")
    dist = choice.distribution
    if isinstance(dist, Normal):
        return float(value), float, lambda _x: 0.0, "real"
    if isinstance(dist, HalfNormal | Gamma | Exponential | LogNormal):
        if value <= 0:
            raise ValueError(f"positive choice {choice.address!r} has invalid value")
        return math.log(float(value)), math.exp, lambda x: -math.log(float(x)), "positive"
    if isinstance(dist, Beta | Uniform):
        low, high = (0.0, 1.0) if isinstance(dist, Beta) else (dist.low, dist.high)
        if not low < value < high:
            raise ValueError(f"bounded choice {choice.address!r} must be strictly inside its bounds")
        width = high - low
        z = math.log((float(value) - low) / (high - float(value)))
        inverse = lambda zz: low + width * _expit(zz)
        log_jac = lambda x: math.log(width) - math.log(float(x) - low) - math.log(high - float(x))
        return z, inverse, log_jac, "bounded"
    if isinstance(dist, Poisson | Geometric | Binomial):
        return float(value), lambda z: round(z), lambda _x: 0.0, "integer"
    raise ValueError(f"adaptive_mh has no proposal transform for {choice.address!r}: {type(dist).__name__}")


def _step(
    model_fn: Callable, args: tuple, kwargs: dict[str, Any], current: Trace,
    kernel: LocalKernel, state: _KernelState, rng: Generator,
) -> tuple[Trace, bool, float]:
    state.visits += 1
    components = [_transform(current.choices[address]) for address in kernel.addresses]
    families = {component[3] for component in components}
    if len(kernel.addresses) > 1 and families - {"real", "positive", "bounded"}:
        raise ValueError("adaptive_mh blocks support continuous real, positive, and bounded choices only")
    old_z = np.array([component[0] for component in components])
    covariance = _covariance(state, len(old_z))
    proposed_z = old_z + rng.multivariate_normal(
        np.zeros(len(old_z)), math.exp(2 * state.log_scale) * covariance
    )
    try:
        values = {
            address: component[1](float(z))
            for address, z, component in zip(kernel.addresses, proposed_z, components, strict=True)
        }
    except (OverflowError, ValueError):
        return current, False, 0.0
    # Discrete support is checked before replay so model code never receives an
    # invalid count (e.g. range(-1) or invalid indexing).
    if any(not math.isfinite(current.choices[address].distribution.log_prob(value)) for address, value in values.items()):
        return current, False, 0.0
    interventions = {address: choice.value for address, choice in current.choices.items()}
    interventions.update(values)
    try:
        proposed = _replay_proposal(model_fn, args, kwargs, interventions, rng)
    except Rejected:
        return current, False, 0.0
    if set(proposed.choices) != set(current.choices):
        raise ValueError("adaptive_mh requires fixed structure; a local proposal changed choice addresses")
    for address, choice in proposed.choices.items():
        if _metadata(choice) != _metadata(current.choices[address]):
            raise ValueError(
                f"adaptive_mh does not support proposal replay with changed distribution metadata at {address!r}"
            )
        # Validate type and transform support before treating a non-finite
        # score as an ordinary MH rejection.
        _transform(choice)
    # A proposal outside model support is an ordinary MH rejection only after
    # its replay structure and proposal-family invariants have been checked.
    if not math.isfinite(proposed.log_score) or not math.isfinite(proposed.log_joint):
        return current, False, 0.0
    if any(not math.isfinite(choice.log_prob) for choice in proposed.choices.values()):
        return current, False, 0.0
    new_components = [_transform(proposed.choices[address]) for address in kernel.addresses]
    new_z = np.array([component[0] for component in new_components])
    log_q_reverse_minus_forward = sum(
        old[2](current.choices[address].value) - new[2](proposed.choices[address].value)
        for address, old, new in zip(kernel.addresses, components, new_components, strict=True)
    )
    log_alpha = proposed.log_joint - current.log_joint + log_q_reverse_minus_forward
    accepted = log_alpha >= 0 or math.log(max(float(rng.random()), np.finfo(float).tiny)) < log_alpha
    jump = float(np.sum((new_z - old_z) ** 2)) if accepted else 0.0
    state.accepted += int(accepted)
    state.total_jump_distance += jump
    return (proposed if accepted else current), accepted, jump


def _adapt(state: _KernelState, trace: Trace, kernel: LocalKernel, accepted: bool) -> None:
    z = np.array([_transform(trace.choices[address])[0] for address in kernel.addresses])
    state.adaptation_visits += 1
    n = state.adaptation_visits
    if state.mean is None:
        state.mean = z.copy()
        state.scatter = np.zeros((len(z), len(z)))
    else:
        delta = z - state.mean
        state.mean += delta / n
        state.scatter += np.outer(delta, z - state.mean)
    target = 0.44 if len(z) == 1 else 0.234
    gamma = min(0.05, 1.0 / math.sqrt(n))
    state.log_scale = float(np.clip(state.log_scale + gamma * (float(accepted) - target), math.log(1e-4), math.log(1e3)))


def _covariance(state: _KernelState, dimension: int) -> np.ndarray:
    if state.scatter is None or state.adaptation_visits < 2:
        return np.eye(dimension)
    return state.scatter / (state.adaptation_visits - 1) + np.eye(dimension) * 1e-6


def _kernel_diagnostics(state: LocalMHSamplerState) -> dict[str, dict[str, float | int]]:
    """Return display diagnostics without permitting scalar/block key clashes."""
    preferred = [kernel.addresses[0] if len(kernel.addresses) == 1 else kernel.name for kernel in state.kernels]
    duplicate_names = {name for name in preferred if preferred.count(name) > 1}
    diagnostics: dict[str, dict[str, float | int]] = {}
    for kernel, name in zip(state.kernels, preferred, strict=True):
        key = repr(kernel.addresses) if name in duplicate_names else name
        value = state.kernel_states[kernel.addresses]
        diagnostics[key] = {
            "visits": value.visits,
            "acceptance_rate": value.accepted / value.visits if value.visits else 0.0,
            "mean_squared_jump_distance": value.total_jump_distance / value.visits if value.visits else 0.0,
            "scale": math.exp(value.log_scale),
        }
    return diagnostics
