"""Restricted, replay-safe adaptive local Metropolis-Hastings kernels."""

from __future__ import annotations

import copy
import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from scipy.special import log_ndtr  # type: ignore[import-untyped]

from cathedral._rng import Generator, SeedLike, make_rng, using_rng
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
class ProposalAssessment:
    """All proposal terms owned by one target-aware MH transition.

    ``log_forward`` and ``log_reverse`` are complete densities/masses in
    model-value coordinates, including any transform Jacobians.  The minimal
    kernel interface has no separate deterministic-map correction.
    """

    proposed: Trace | None
    log_forward: float
    log_reverse: float

    def log_acceptance_ratio(self, current: Trace) -> float:
        """Return the MH ratio for the supplied target traces."""
        if self.proposed is None:
            return float("-inf")
        return (
            self.proposed.log_joint - current.log_joint
            + self.log_reverse - self.log_forward
        )


@dataclass(frozen=True)
class ReplayTarget:
    """The replayable model target passed to a scheduled kernel."""

    fn: Callable
    args: tuple
    kwargs: dict[str, Any]

    def replay(self, interventions: dict[str, Any], rng: Generator) -> Trace:
        return _replay_proposal(self.fn, self.args, self.kwargs, interventions, rng)


class Kernel(Protocol):
    """A callable target-aware MH proposal kernel."""

    @property
    def addresses(self) -> tuple[str, ...]: ...

    @property
    def name(self) -> str: ...

    def step(
        self, trace: Trace, target: ReplayTarget, state: _KernelState, rng: Generator,
    ) -> ProposalAssessment: ...


@dataclass(frozen=True)
class LocalKernel:
    """A scalar or user-declared random-walk block in a compound schedule."""

    addresses: tuple[str, ...]

    @property
    def name(self) -> str:
        """Unambiguous display name; state uses a typed internal key."""
        return repr(self.addresses)

    def step(self, trace: Trace, target: ReplayTarget, state: _KernelState, rng: Generator) -> ProposalAssessment:
        return _local_assessment(trace, target, self, state, rng)


@dataclass(frozen=True)
class PriorResimulationKernel:
    """A one-address prior-resimulation kernel for the fixed-structure subset."""

    addresses: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.addresses) != 1:
            raise ValueError("prior-resimulation kernels update exactly one address")

    @property
    def name(self) -> str:
        return f"prior:{self.addresses[0]}"

    def step(self, trace: Trace, target: ReplayTarget, state: _KernelState, rng: Generator) -> ProposalAssessment:
        return _prior_resimulation_assessment(trace, target, self, rng)


@dataclass
class _KernelState:
    log_scale: float = 0.0
    visits: int = 0
    accepted: int = 0
    total_jump_distance: float = 0.0
    adaptation_visits: int = 0
    # Covariance is estimated during the second half of warmup rather than
    # over all adaptation history.  This prevents the starting-region
    # transient from determining the geometry that is frozen for sampling.
    covariance_visits: int = 0
    mean: float | np.ndarray | None = None
    scatter: float | np.ndarray | None = None
    # Derived adaptation state is serializable so a resumed frozen chain keeps
    # exactly the geometry at which the original chain froze.
    covariance_cache: np.ndarray | None = None
    covariance_cache_visits: int = -1
    proposal_cholesky: float | np.ndarray | None = None
    proposal_log_determinant: float | None = None
    proposal_cache_visits: int = -1
    proposal_cache_log_scale: float | None = None


@dataclass
class LocalMHSamplerState:
    """Serializable state needed to resume an adaptive local-MH chain exactly."""

    trace: Trace
    rng_state: dict[str, Any]
    kernel_states: dict[tuple[str, tuple[str, ...]], _KernelState]
    kernels: tuple[Kernel, ...]
    kernel_weights: tuple[float, ...]
    warmup: int
    step: int = 0


def local_mh_sample(  # noqa: C901
    model_fn: Callable, args: tuple = (), kwargs: dict[str, Any] | None = None,
    num_samples: int = 1000, *, warmup: int | None = None, lag: int = 1,
    blocks: Sequence[Sequence[str]] | None = None, initial_trace: Trace | None = None,
    sampler_state: LocalMHSamplerState | None = None, max_init_attempts: int = 10000,
    seed: SeedLike = None, prior_resimulation: bool = False,
    kernel_weights: Sequence[float] | Mapping[tuple[str, ...], float] | None = None,
    _info: dict | None = None,
) -> tuple[list[Trace], LocalMHSamplerState]:
    """Sample on the explicitly named, scalar, fixed-structure subset.

    Adaptation uses every attempted transition during warmup and freezes both
    scale and covariance after exactly ``warmup`` transitions.
    """
    if kwargs is None:
        kwargs = {}
    if num_samples < 1 or lag < 1:
        raise ValueError("num_samples and lag must be positive")
    if sampler_state is not None and (initial_trace is not None or blocks is not None or prior_resimulation or kernel_weights is not None):
        raise ValueError("sampler_state cannot be combined with kernel configuration")
    if sampler_state is not None and seed is not None:
        raise ValueError("seed cannot be supplied when resuming adaptive_mh; sampler state owns the RNG")

    if sampler_state is None:
        rng = make_rng(seed)
        current = initial_trace or _get_initial_trace(model_fn, args, kwargs, max_init_attempts, rng)
        # Validate with a clone: validation must not advance the chain RNG.
        validation_rng = make_rng()
        validation_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        current = _validate_initial_trace(model_fn, args, kwargs, current, validation_rng)
        kernels = _make_kernels(current, blocks, prior_resimulation)
        weights = _validate_kernel_weights(kernels, kernel_weights)
        state = LocalMHSamplerState(
            trace=current, rng_state=copy.deepcopy(dict(rng.bit_generator.state)),
            kernel_states={_kernel_key(kernel): _KernelState() for kernel in kernels},
            kernels=kernels, kernel_weights=weights, warmup=num_samples // 2 if warmup is None else warmup,
        )
    else:
        state = copy.deepcopy(sampler_state)
        rng = make_rng()
        rng.bit_generator.state = copy.deepcopy(dict(state.rng_state))
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
    target = ReplayTarget(model_fn, args, kwargs)
    sole_kernel = state.kernels[0] if len(state.kernels) == 1 else None
    for _ in range(total_steps):
        # The second warmup phase estimates the covariance that will be frozen.
        # It excludes the initialization transient while retaining a long,
        # generic stabilization window for every scheduled local kernel.
        if state.step == state.warmup // 2:
            _restart_covariance_adaptation(state)
        kernel = sole_kernel if sole_kernel is not None else state.kernels[_draw_kernel_index(state.kernel_weights, rng)]
        current, did_accept, jump = _step(
            current, target, kernel, state.kernel_states[_kernel_key(kernel)], rng
        )
        accepted += int(did_accept)
        jump_distance += jump
        if state.step < state.warmup and isinstance(kernel, LocalKernel):
            _adapt(state.kernel_states[_kernel_key(kernel)], current, kernel, did_accept)
        state.step += 1
        if state.step > state.warmup and (state.step - state.warmup - 1) % lag == 0:
            traces.append(current)

    state.trace = current
    state.rng_state = copy.deepcopy(dict(rng.bit_generator.state))
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
        return bool(left == right)


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
        return (type(dist), 0.0, 1.0)
    if isinstance(dist, Normal):
        return (type(dist),)
    if isinstance(dist, HalfNormal | Exponential | LogNormal):
        return (type(dist),)
    if isinstance(dist, Gamma):
        return (type(dist),)
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
        if not replay_choice.explicit_name:
            raise ValueError(f"adaptive_mh requires explicit unique names; replayed address: {address!r}")
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


def _make_kernels(
    trace: Trace, blocks: Sequence[Sequence[str]] | None, prior_resimulation: bool,
) -> tuple[Kernel, ...]:
    seen: set[str] = set()
    kernels: list[Kernel] = []
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
    # Keep scalar moves for block members.  The fixed schedule therefore
    # composes coordinate and correlated moves rather than replacing one with
    # the other; this composition preserves the target even though a whole
    # cycle need not itself be reversible.
    for address, choice in trace.choices.items():
        _transform(choice)
        kernels.append(LocalKernel((address,)))
    if prior_resimulation:
        kernels.extend(PriorResimulationKernel((address,)) for address in trace.choices)
    return tuple(kernels)


def _validate_kernel_weights(
    kernels: Sequence[Kernel], weights: Sequence[float] | Mapping[tuple[str, ...], float] | None,
) -> tuple[float, ...]:
    """Validate immutable compound-schedule weights and normalize them."""
    if weights is None:
        raw = [1.0] * len(kernels)
    elif isinstance(weights, Mapping):
        # Address keys intentionally affect every matching component: a scalar
        # local and its optional prior-resimulation component are distinct
        # transition types and should use the sequence form to be separated.
        raw = [weights.get(kernel.addresses, 0.0) for kernel in kernels]
    else:
        raw = list(weights)
        if len(raw) != len(kernels):
            raise ValueError("kernel_weights must have one entry for each scheduled kernel")
    if len(raw) != len(kernels) or any(not math.isfinite(float(w)) or w < 0 for w in raw):
        raise ValueError("kernel_weights must be finite, non-negative weights")
    try:
        total = math.fsum(float(weight) for weight in raw)
    except OverflowError:
        total = float("inf")
    if not math.isfinite(total) or total <= 0:
        raise ValueError("kernel_weights must have a finite positive sum")
    normalized = tuple(float(weight / total) for weight in raw)
    active = {address for kernel, weight in zip(kernels, normalized, strict=True) if weight > 0 for address in kernel.addresses}
    missing = set().union(*(set(kernel.addresses) for kernel in kernels)) - active
    if missing:
        raise ValueError(f"kernel_weights leave active choices uncovered: {sorted(missing)!r}")
    return normalized


def _draw_kernel_index(weights: Sequence[float], rng: Generator) -> int:
    """Draw from a fixed compound schedule without mutable/adaptive weights."""
    return int(rng.choice(len(weights), p=np.asarray(weights, dtype=float)))


def _kernel_key(kernel: Kernel) -> tuple[str, tuple[str, ...]]:
    """Typed private state key, safe even for adversarial address strings."""
    return ("local" if isinstance(kernel, LocalKernel) else "prior", kernel.addresses)


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


def _local_assessment(  # noqa: C901
    current: Trace, target: ReplayTarget, kernel: LocalKernel, state: _KernelState, rng: Generator,
) -> ProposalAssessment:
    """Create a complete model-coordinate local random-walk assessment."""
    # Scalar moves dominate common models.  Keep their transform, proposal,
    # and density arithmetic as Python scalars; the block path below retains
    # the complete general-coordinate accounting.
    if len(kernel.addresses) == 1:
        address = kernel.addresses[0]
        component = _transform(current.choices[address])
        scalar_old_z, inverse, log_jacobian, family = component
        sigma, log_determinant = _scalar_proposal_geometry(state)
        scalar_proposed_z = scalar_old_z + sigma * float(rng.normal())
        try:
            value = inverse(scalar_proposed_z)
        except (OverflowError, ValueError):
            return ProposalAssessment(None, 0.0, 0.0)
        if family == "bounded":
            dist = current.choices[address].distribution
            if isinstance(dist, Beta):
                low, high = 0.0, 1.0
            else:
                uniform_dist = cast(Uniform, dist)
                low, high = uniform_dist.low, uniform_dist.high
            if not low < value < high:
                return ProposalAssessment(None, 0.0, 0.0)
        if not math.isfinite(current.choices[address].distribution.log_prob(value)):
            return ProposalAssessment(None, 0.0, 0.0)
        interventions = {key: choice.value for key, choice in current.choices.items()}
        interventions[address] = value
        try:
            proposed = target.replay(interventions, rng)
        except Rejected:
            return ProposalAssessment(None, 0.0, 0.0)
        if not _validate_proposed_trace(current, proposed, "local proposal"):
            return ProposalAssessment(None, 0.0, 0.0)
        new_component = _transform(proposed.choices[address])
        scalar_new_z = new_component[0]
        if family == "integer":
            return ProposalAssessment(
                proposed,
                _rounded_normal_log_mass(scalar_new_z, scalar_old_z, sigma),
                _rounded_normal_log_mass(scalar_old_z, scalar_new_z, sigma),
            )
        delta = (scalar_new_z - scalar_old_z) / sigma
        log_z = -0.5 * (math.log(2 * math.pi) + log_determinant + delta * delta)
        return ProposalAssessment(
            proposed,
            log_z + new_component[2](proposed.choices[address].value),
            log_z + log_jacobian(current.choices[address].value),
        )
    components = [_transform(current.choices[address]) for address in kernel.addresses]
    families = {component[3] for component in components}
    if len(kernel.addresses) > 1 and families - {"real", "positive", "bounded"}:
        raise ValueError("adaptive_mh blocks support continuous real, positive, and bounded choices only")
    vector_old_z = np.array([component[0] for component in components])
    cholesky, log_determinant = _proposal_geometry(state, len(vector_old_z))
    # Generator.multivariate_normal refactorizes on every call.  Sampling the
    # cached positive-definite factor is equivalent and avoids that overhead.
    if len(vector_old_z) == 1:
        vector_proposed_z = vector_old_z + cholesky[0, 0] * float(rng.normal())
    else:
        vector_proposed_z = vector_old_z + cholesky @ rng.standard_normal(len(vector_old_z))
    try:
        values = {
            address: component[1](float(z))
            for address, z, component in zip(kernel.addresses, vector_proposed_z, components, strict=True)
        }
    except (OverflowError, ValueError):
        return ProposalAssessment(None, 0.0, 0.0)
    # Floating point expit can round to an exact endpoint.  It has zero
    # transformed proposal density there, so reject before replay rather than
    # treating a boundary-valued Uniform/Beta draw as a valid transformed move.
    for address, value, component in zip(kernel.addresses, values.values(), components, strict=True):
        if component[3] == "bounded":
            dist = current.choices[address].distribution
            if isinstance(dist, Beta):
                low, high = 0.0, 1.0
            else:
                uniform_dist = cast(Uniform, dist)
                low, high = uniform_dist.low, uniform_dist.high
            if not low < value < high:
                return ProposalAssessment(None, 0.0, 0.0)
    # Discrete support is checked before replay so model code never receives an
    # invalid count (e.g. range(-1) or invalid indexing).
    if any(not math.isfinite(current.choices[address].distribution.log_prob(value)) for address, value in values.items()):
        return ProposalAssessment(None, 0.0, 0.0)
    interventions = {address: choice.value for address, choice in current.choices.items()}
    interventions.update(values)
    try:
        proposed = target.replay(interventions, rng)
    except Rejected:
        return ProposalAssessment(None, 0.0, 0.0)
    if not _validate_proposed_trace(current, proposed, "local proposal"):
        return ProposalAssessment(None, 0.0, 0.0)
    new_components = [_transform(proposed.choices[address]) for address in kernel.addresses]
    vector_new_z = np.array([component[0] for component in new_components])
    if families == {"integer"}:
        # Rounded-Normal masses, not fictitious continuous densities.
        sigma = float(cholesky[0, 0])
        log_forward = _rounded_normal_log_mass(float(vector_new_z[0]), float(vector_old_z[0]), sigma)
        log_reverse = _rounded_normal_log_mass(float(vector_old_z[0]), float(vector_new_z[0]), sigma)
    else:
        # Retain complete density accounting even though this random walk is
        # symmetric; the shared transformed-space term is computed once.
        log_z = _normal_log_density_from_cholesky(vector_new_z - vector_old_z, cholesky, log_determinant)
        log_forward = log_z + sum(component[2](proposed.choices[address].value) for address, component in zip(kernel.addresses, new_components, strict=True))
        log_reverse = log_z + sum(component[2](current.choices[address].value) for address, component in zip(kernel.addresses, components, strict=True))
    return ProposalAssessment(proposed, log_forward, log_reverse)


def _prior_resimulation_assessment(
    current: Trace, target: ReplayTarget, kernel: PriorResimulationKernel, rng: Generator,
) -> ProposalAssessment:
    """Propose from the current selected conditional without replay RNG use."""
    address = kernel.addresses[0]
    with using_rng(rng):
        value = current.choices[address].distribution.sample()
    interventions = {key: choice.value for key, choice in current.choices.items()}
    interventions[address] = value
    try:
        proposed = target.replay(interventions, rng)
    except Rejected:
        return ProposalAssessment(None, 0.0, 0.0)
    except ValueError as error:
        if "proposal replay that consumes RNG" in str(error):
            raise ValueError(
                "adaptive_mh prior resimulation requires fixed structure; proposal replay consumed RNG"
            ) from None
        raise
    if not _validate_proposed_trace(current, proposed, "prior resimulation"):
        return ProposalAssessment(None, 0.0, 0.0)
    # Sampling is from the current conditional; reverse is evaluated from the
    # candidate conditional, where held coordinates are the candidate values.
    return ProposalAssessment(
        proposed, current.choices[address].distribution.log_prob(value),
        proposed.choices[address].distribution.log_prob(current.choices[address].value),
    )


def _validate_proposed_trace(current: Trace, proposed: Trace, context: str) -> bool:
    if set(proposed.choices) != set(current.choices):
        raise ValueError(f"adaptive_mh requires fixed structure; a {context} changed choice addresses")
    for address, choice in proposed.choices.items():
        if _metadata(choice) != _metadata(current.choices[address]):
            raise ValueError(f"adaptive_mh does not support {context} with changed distribution metadata at {address!r}")
        _transform(choice)
    if math.isnan(proposed.log_score) or (math.isinf(proposed.log_score) and proposed.log_score > 0):
        raise ValueError(f"adaptive_mh {context} produced an invalid numeric target score")
    if math.isnan(proposed.log_joint) or (math.isinf(proposed.log_joint) and proposed.log_joint > 0):
        raise ValueError(f"adaptive_mh {context} produced an invalid numeric target score")
    return math.isfinite(proposed.log_joint) and all(math.isfinite(choice.log_prob) for choice in proposed.choices.values())


def _normal_log_density(value: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    delta = value - mean
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("adaptive_mh proposal covariance is not positive definite")
    return float(-0.5 * (len(delta) * math.log(2 * math.pi) + log_det + delta @ np.linalg.solve(covariance, delta)))


def _normal_log_density_from_cholesky(delta: np.ndarray, cholesky: np.ndarray, log_determinant: float) -> float:
    """Evaluate a normal density using its already validated factorization."""
    if len(delta) == 1:
        quadratic = float(delta[0] / cholesky[0, 0]) ** 2
    else:
        solved = np.linalg.solve(cholesky, delta)
        quadratic = float(solved @ solved)
    return -0.5 * (len(delta) * math.log(2 * math.pi) + log_determinant + quadratic)


def _rounded_normal_log_mass(value: float, mean: float, sigma: float) -> float:
    lower = (value - 0.5 - mean) / sigma
    upper = (value + 0.5 - mean) / sigma
    # Direct erf subtraction underflows in a tail.  Subtract the smaller
    # normal tail from the larger one in log space instead.
    if lower > 0:
        log_large = _normal_log_cdf(-lower)
        log_small = _normal_log_cdf(-upper)
    else:
        log_large = _normal_log_cdf(upper)
        log_small = _normal_log_cdf(lower)
    if log_small == float("-inf"):
        return log_large
    return log_large + math.log(-math.expm1(log_small - log_large))


def _normal_log_cdf(value: float) -> float:
    """Exact SciPy-backed log Phi(value), including narrow far-tail cells."""
    return float(log_ndtr(value))


def _step(
    current: Trace, target: ReplayTarget, kernel: Kernel, state: _KernelState, rng: Generator,
) -> tuple[Trace, bool, float]:
    """Invoke a kernel and apply the single shared MH accept/reject rule."""
    state.visits += 1
    assessment = kernel.step(current, target, state, rng)
    log_alpha = assessment.log_acceptance_ratio(current)
    if math.isnan(log_alpha) or log_alpha == math.inf:
        raise ValueError("adaptive_mh proposal produced an invalid numeric acceptance ratio")
    accepted = assessment.proposed is not None and (
        log_alpha >= 0 or math.log(max(float(rng.random()), np.finfo(float).tiny)) < log_alpha
    )
    if assessment.proposed is None:
        return current, False, 0.0
    if accepted and len(kernel.addresses) == 1:
        address = kernel.addresses[0]
        delta = _transform(assessment.proposed.choices[address])[0] - _transform(current.choices[address])[0]
        jump = delta * delta
    elif accepted:
        old_z = np.array([_transform(current.choices[address])[0] for address in kernel.addresses])
        new_z = np.array([_transform(assessment.proposed.choices[address])[0] for address in kernel.addresses])
        jump = float(np.sum((new_z - old_z) ** 2))
    else:
        jump = 0.0
    state.accepted += int(accepted)
    state.total_jump_distance += jump
    return (assessment.proposed if accepted else current), accepted, jump


def _adapt(state: _KernelState, trace: Trace, kernel: LocalKernel, accepted: bool) -> None:
    if len(kernel.addresses) == 1:
        scalar_z = _transform(trace.choices[kernel.addresses[0]])[0]
        state.adaptation_visits += 1
        n = state.adaptation_visits
        state.covariance_visits += 1
        covariance_n = state.covariance_visits
        if state.mean is None:
            state.mean = scalar_z
            state.scatter = 0.0
        else:
            scalar_mean = cast(float, state.mean)
            scalar_scatter = cast(float, state.scatter)
            delta = scalar_z - scalar_mean
            state.mean = scalar_mean + delta / covariance_n
            state.scatter = scalar_scatter + delta * (scalar_z - state.mean)
        state.covariance_cache = None
        state.proposal_cholesky = None
        target = 0.44
        gamma = min(0.05, 1.0 / math.sqrt(n))
        state.log_scale = max(math.log(1e-4), min(math.log(1e3), state.log_scale + gamma * (float(accepted) - target)))
        return
    vector_z = np.array([_transform(trace.choices[address])[0] for address in kernel.addresses])
    state.adaptation_visits += 1
    n = state.adaptation_visits
    state.covariance_visits += 1
    covariance_n = state.covariance_visits
    if state.mean is None:
        state.mean = vector_z.copy()
        state.scatter = np.zeros((len(vector_z), len(vector_z)))
    else:
        vector_mean = cast(np.ndarray, state.mean)
        vector_scatter = cast(np.ndarray, state.scatter)
        delta = vector_z - vector_mean
        state.mean = vector_mean + delta / covariance_n
        state.scatter = vector_scatter + np.outer(delta, vector_z - state.mean)
    state.covariance_cache = None
    state.proposal_cholesky = None
    target = 0.44 if len(vector_z) == 1 else 0.234
    gamma = min(0.05, 1.0 / math.sqrt(n))
    state.log_scale = float(np.clip(state.log_scale + gamma * (float(accepted) - target), math.log(1e-4), math.log(1e3)))


def _restart_covariance_adaptation(sampler_state: LocalMHSamplerState) -> None:
    """Begin the final warmup covariance window without changing scales."""
    for kernel in sampler_state.kernels:
        if isinstance(kernel, LocalKernel):
            state = sampler_state.kernel_states[_kernel_key(kernel)]
            state.covariance_visits = 0
            state.mean = None
            state.scatter = None
            state.covariance_cache = None
            state.proposal_cholesky = None


def _covariance(state: _KernelState, dimension: int) -> np.ndarray:
    if state.covariance_cache is not None and state.covariance_cache_visits == state.adaptation_visits:
        return state.covariance_cache
    # ``getattr`` keeps sampler states serialized before staged covariance
    # adaptation resumable; their accumulated statistic is one full window.
    covariance_visits = getattr(state, "covariance_visits", state.adaptation_visits)
    if state.scatter is None or covariance_visits < 2:
        covariance = np.eye(dimension)
    elif dimension == 1 and isinstance(state.scatter, float):
        covariance = np.array([[(state.scatter + 1.0) / covariance_visits + 1e-6]])
    else:
        # A unit-scale, one-pseudo-observation prior prevents two identical early
        # accepted states from collapsing a random walk to the numerical ridge.
        # The empirical covariance still dominates within the final window.
        vector_scatter = cast(np.ndarray, state.scatter)
        covariance = (vector_scatter + np.eye(dimension)) / covariance_visits + np.eye(dimension) * 1e-6
    state.covariance_cache = covariance
    state.covariance_cache_visits = state.adaptation_visits
    return covariance


def _scalar_proposal_geometry(state: _KernelState) -> tuple[float, float]:
    """Return scalar random-walk geometry without allocating NumPy arrays."""
    if (
        state.proposal_cholesky is not None
        and state.proposal_cache_visits == state.adaptation_visits
        and state.proposal_cache_log_scale == state.log_scale
    ):
        return cast(float, state.proposal_cholesky), cast(float, state.proposal_log_determinant)
    covariance_visits = getattr(state, "covariance_visits", state.adaptation_visits)
    if state.scatter is None or covariance_visits < 2:
        variance = 1.0
    else:
        scalar_scatter = cast(float, state.scatter)
        variance = (scalar_scatter + 1.0) / covariance_visits + 1e-6
    sigma = math.sqrt(variance) * math.exp(state.log_scale)
    log_determinant = 2.0 * math.log(sigma)
    state.proposal_cholesky = sigma
    state.proposal_log_determinant = log_determinant
    state.proposal_cache_visits = state.adaptation_visits
    state.proposal_cache_log_scale = state.log_scale
    return sigma, log_determinant


def _proposal_geometry(state: _KernelState, dimension: int) -> tuple[np.ndarray, float]:
    """Return cached Cholesky geometry for the state's current proposal."""
    if (
        state.proposal_cholesky is not None
        and state.proposal_cache_visits == state.adaptation_visits
        and state.proposal_cache_log_scale == state.log_scale
    ):
        return cast(np.ndarray, state.proposal_cholesky), cast(float, state.proposal_log_determinant)
    cholesky = np.linalg.cholesky(_covariance(state, dimension)) * math.exp(state.log_scale)
    log_determinant = 2.0 * math.fsum(math.log(float(value)) for value in np.diag(cholesky))
    state.proposal_cholesky = cholesky
    state.proposal_log_determinant = log_determinant
    state.proposal_cache_visits = state.adaptation_visits
    state.proposal_cache_log_scale = state.log_scale
    return cholesky, log_determinant


def _kernel_diagnostics(state: LocalMHSamplerState) -> dict[str, dict[str, float | int]]:
    """Return display diagnostics without permitting scalar/block key clashes."""
    preferred = [kernel.addresses[0] if len(kernel.addresses) == 1 else kernel.name for kernel in state.kernels]
    diagnostics: dict[str, dict[str, float | int]] = {}
    for ordinal, (kernel, name) in enumerate(zip(state.kernels, preferred, strict=True), start=1):
        # Ordinals make names injective even when arbitrary address strings
        # imitate a block repr or a prior prefix.
        key = f"{ordinal}:{'local' if isinstance(kernel, LocalKernel) else 'prior'}:{name}"
        value = state.kernel_states[_kernel_key(kernel)]
        diagnostics[key] = {
            "visits": value.visits,
            "acceptance_rate": value.accepted / value.visits if value.visits else 0.0,
            "mean_squared_jump_distance": value.total_jump_distance / value.visits if value.visits else 0.0,
            "scale": math.exp(value.log_scale),
        }
    return diagnostics
