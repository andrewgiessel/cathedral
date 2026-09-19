"""Source-invoking balance checks for the restricted local-MH kernels."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from cathedral.distributions import Binomial, HalfNormal, Normal, Uniform
from cathedral.inference.local_mh import (
    LocalKernel,
    PriorResimulationKernel,
    ReplayTarget,
    _KernelState,
    _proposal_geometry,
    _step,
)
from cathedral.primitives import condition, factor, sample
from cathedral.trace import run_with_trace


class _ForcedRNG:
    """Minimal deterministic source of the proposal and accept draws."""

    def __init__(self, *, normal=0.0, random=0.5, standard_normal=None, binomial=0):
        self.normal_value = normal
        self.random_value = random
        self.standard_normal_value = standard_normal
        self.binomial_value = binomial
        self.random_calls = 0
        # Proposal replays snapshot this public NumPy-generator interface.
        self.bit_generator = self
        self.state = {"forced_rng": True}

    def normal(self):
        return self.normal_value

    def standard_normal(self, size):
        assert self.standard_normal_value is not None
        assert size == len(self.standard_normal_value)
        return np.asarray(self.standard_normal_value, dtype=float)

    def random(self):
        self.random_calls += 1
        return self.random_value

    def binomial(self, n, p):
        assert n == 1
        return self.binomial_value


def _finite_model():
    """Four-state target with dependent scores, not a product distribution."""
    x = sample(Binomial(1, 0.35), name="x")
    y = sample(Binomial(1, 0.7), name="y")
    # Deliberately non-additive so x and y kernels do not commute.
    factor(((0.0, -1.1), (-0.4, 1.3))[x][y])
    return x, y


def _trace(state):
    return run_with_trace(_finite_model, interventions={"x": state[0], "y": state[1]})


def _target(states):
    weights = np.array([math.exp(_trace(state).log_joint) for state in states])
    return weights / weights.sum()


def _assert_balance_and_stationarity(pi, transition):
    assert np.allclose(pi[:, None] * transition, (pi[:, None] * transition).T, atol=2e-12)
    assert np.allclose(pi @ transition, pi, atol=2e-12)
    assert np.allclose(transition.sum(axis=1), 1.0, atol=2e-12)


def _local_matrix(states, kernel, sigma):
    """Enumerate actual assessments; use Normal CDF masses independently."""
    target = ReplayTarget(_finite_model, (), {})
    matrix = np.zeros((len(states), len(states)))
    address_index = 0 if kernel.addresses == ("x",) else 1
    for old_index, old in enumerate(states):
        current = _trace(old)
        old_value = old[address_index]
        for candidate_value in (0, 1):
            candidate = list(old)
            candidate[address_index] = candidate_value
            # This is only a representative draw in the rounding cell; its
            # probability is calculated independently from the exact cell mass.
            assessment = kernel.step(
                current, target, _KernelState(log_scale=math.log(sigma)),
                _ForcedRNG(normal=(candidate_value - old_value) / sigma),
            )
            assert assessment.proposed is not None
            assert tuple(assessment.proposed.result) == tuple(candidate)
            log_alpha = assessment.log_acceptance_ratio(current)
            proposal_mass = norm.cdf((candidate_value + 0.5 - old_value) / sigma) - norm.cdf(
                (candidate_value - 0.5 - old_value) / sigma
            )
            new_index = states.index(tuple(candidate))
            matrix[old_index, new_index] += proposal_mass * min(1.0, math.exp(log_alpha))
        # All rounded candidates outside Binomial(1, p)'s support are rejected.
        matrix[old_index, old_index] += 1.0 - matrix[old_index].sum()
    return matrix


def _prior_matrix(states, kernel):
    target = ReplayTarget(_finite_model, (), {})
    matrix = np.zeros((len(states), len(states)))
    address_index = 0 if kernel.addresses == ("x",) else 1
    p = 0.35 if address_index == 0 else 0.7
    for old_index, old in enumerate(states):
        current = _trace(old)
        for candidate_value, proposal_mass in ((0, 1.0 - p), (1, p)):
            candidate = list(old)
            candidate[address_index] = candidate_value
            assessment = kernel.step(
                current, target, _KernelState(), _ForcedRNG(binomial=candidate_value)
            )
            assert assessment.proposed is not None
            assert tuple(assessment.proposed.result) == tuple(candidate)
            new_index = states.index(tuple(candidate))
            matrix[old_index, new_index] += proposal_mass * min(
                1.0, math.exp(assessment.log_acceptance_ratio(current))
            )
        matrix[old_index, old_index] += 1.0 - matrix[old_index].sum()
    return matrix


def test_actual_finite_kernel_matrices_balance_and_compositions_are_stationary():
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pi = _target(states)
    local_x = _local_matrix(states, LocalKernel(("x",)), sigma=0.8)
    prior_y = _prior_matrix(states, PriorResimulationKernel(("y",)))

    _assert_balance_and_stationarity(pi, local_x)
    _assert_balance_and_stationarity(pi, prior_y)
    # Kernel composition preserves pi, while a deterministic sweep is not
    # generally reversible and must not be tested as if it were.
    forward = local_x @ prior_y
    reverse = prior_y @ local_x
    assert not np.allclose(forward, reverse)
    assert np.allclose(pi @ forward, pi, atol=2e-12)
    assert np.allclose(pi @ reverse, pi, atol=2e-12)


def test_step_uses_assessment_likelihood_ratio_for_accept_and_reject():
    current = _trace((0, 0))
    target = ReplayTarget(_finite_model, (), {})
    kernel = LocalKernel(("x",))
    state = _KernelState(log_scale=math.log(0.8))
    assessment = kernel.step(current, target, state, _ForcedRNG(normal=1 / 0.8))
    alpha = math.exp(assessment.log_acceptance_ratio(current))
    assert 0 < alpha < 1

    rejected_rng = _ForcedRNG(normal=1 / 0.8, random=(alpha + 1.0) / 2.0)
    rejected, accepted, jump = _step(current, target, kernel, _KernelState(log_scale=math.log(0.8)), rejected_rng)
    assert not accepted and jump == 0.0 and rejected is current and rejected_rng.random_calls == 1

    accepted_rng = _ForcedRNG(normal=1 / 0.8, random=alpha / 2.0)
    accepted_trace, accepted, jump = _step(current, target, kernel, _KernelState(log_scale=math.log(0.8)), accepted_rng)
    assert accepted and tuple(accepted_trace.result) == (1, 0) and jump == pytest.approx(1.0)
    assert accepted_rng.random_calls == 1


def test_transformed_and_block_assessments_include_full_model_coordinate_q():
    def positive_model():
        x = sample(HalfNormal(1.2), name="x")
        factor(-0.3 * x)
        return x

    current = run_with_trace(positive_model, interventions={"x": 1.5})
    sigma = 0.7
    proposed_z = math.log(1.5) + sigma * 0.4
    proposed_x = math.exp(proposed_z)
    assessment = LocalKernel(("x",)).step(
        current, ReplayTarget(positive_model, (), {}), _KernelState(log_scale=math.log(sigma)), _ForcedRNG(normal=0.4)
    )
    log_z_q = norm.logpdf(proposed_z - math.log(1.5), scale=sigma)
    assert assessment.log_forward == pytest.approx(log_z_q - math.log(proposed_x))
    assert assessment.log_reverse == pytest.approx(log_z_q - math.log(1.5))

    def bounded_model():
        x = sample(Uniform(2.0, 5.0), name="x")
        factor(-x)
        return x

    current = run_with_trace(bounded_model, interventions={"x": 3.0})
    old_z = math.log((3.0 - 2.0) / (5.0 - 3.0))
    proposed_z = old_z + sigma * -0.25
    proposed_x = 2.0 + 3.0 / (1.0 + math.exp(-proposed_z))
    assessment = LocalKernel(("x",)).step(
        current, ReplayTarget(bounded_model, (), {}), _KernelState(log_scale=math.log(sigma)), _ForcedRNG(normal=-0.25)
    )
    log_z_q = norm.logpdf(proposed_z - old_z, scale=sigma)
    jacobian = lambda value: math.log(3.0) - math.log(value - 2.0) - math.log(5.0 - value)
    assert assessment.log_forward == pytest.approx(log_z_q + jacobian(proposed_x))
    assert assessment.log_reverse == pytest.approx(log_z_q + jacobian(3.0))

    def block_model():
        x = sample(Normal(0, 1), name="x")
        y = sample(Normal(0, 1), name="y")
        factor(-0.2 * (x - y) ** 2)
        return x, y

    current = run_with_trace(block_model, interventions={"x": 0.2, "y": -0.4})
    state = _KernelState(adaptation_visits=3, scatter=np.array([[2.0, 0.6], [0.6, 1.0]]), log_scale=math.log(0.9))
    cholesky, log_det = _proposal_geometry(state, 2)
    draw = np.array([0.3, -0.2])
    assessment = LocalKernel(("x", "y")).step(
        current, ReplayTarget(block_model, (), {}), state, _ForcedRNG(standard_normal=draw)
    )
    delta = cholesky @ draw
    assert assessment.log_forward == pytest.approx(-0.5 * (2 * math.log(2 * math.pi) + log_det + draw @ draw))
    assert assessment.log_reverse == pytest.approx(assessment.log_forward)
    assert tuple(assessment.proposed.result) == pytest.approx(tuple(np.array([0.2, -0.4]) + delta))


def test_prior_replay_rng_guard_triggers_after_new_move_then_rejection(monkeypatch):
    def changing_rejected_model():
        x = sample(Binomial(1, 0.5), name="x")
        if x:
            sample(Normal(0, 1), name="introduced_after_move")
            condition(False)
        return x

    initial = run_with_trace(changing_rejected_model, interventions={"x": 0})
    monkeypatch.setattr(Binomial, "sample", lambda self: 1)
    with pytest.raises(ValueError, match="prior resimulation requires fixed structure; proposal replay consumed RNG"):
        PriorResimulationKernel(("x",)).step(
            initial, ReplayTarget(changing_rejected_model, (), {}), _KernelState(), np.random.default_rng(12)
        )
