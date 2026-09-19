"""Focused tests for the replay-safe adaptive local-MH subset."""

import math

import numpy as np
import pytest
from scipy.special import log_ndtr

from cathedral import model
from cathedral.distributions import Beta, Gamma, Normal, Poisson, Uniform
from cathedral.inference.local_mh import (
    LocalKernel,
    PriorResimulationKernel,
    _covariance,
    _draw_kernel_index,
    _kernel_key,
    _KernelState,
    _proposal_geometry,
    _rounded_normal_log_mass,
    _validate_kernel_weights,
    local_mh_sample,
)
from cathedral.model import infer
from cathedral.primitives import observe, sample
from cathedral.trace import run_with_trace


class TestAdaptiveMH:
    def test_scalar_normal_matches_analytic_posterior(self):
        @model
        def normal_model():
            x = sample(Normal(0, 1), name="x")
            observe(Normal(x, 1), 1.0)
            return x

        posterior = infer(normal_model, method="adaptive_mh", num_samples=3_000, warmup=500, seed=42)
        # N(0,1) prior and N(x,1) observation y=1 gives N(1/2, 1/2).
        assert abs(posterior.mean() - 0.5) < 0.12
        assert abs(float(np.var(posterior.samples)) - 0.5) < 0.12
        assert posterior.info is not None
        assert posterior.info.extra["adaptation_frozen"] is True
        assert posterior.info.extra["kernel_diagnostics"]["1:local:x"]["visits"] == 3_500

    def test_bounded_transform_matches_analytic_beta_mean(self):
        @model
        def beta_model():
            p = sample(Beta(1, 1), name="p")
            observe(Normal(p, 0.08), 0.75)
            return p

        posterior = infer(beta_model, method="adaptive_mh", num_samples=4_000, warmup=750, seed=15)
        assert 0.68 < posterior.mean() < 0.80
        assert all(0 < value < 1 for value in posterior.samples)

    def test_joint_conditionals_are_rescored(self):
        @model
        def conditional_model():
            x = sample(Normal(0, 1), name="x")
            y = sample(Normal(x, 1), name="y")
            observe(Normal(y, 0.5), 1.0)
            return {"x": x, "y": y}

        posterior = infer(conditional_model, method="adaptive_mh", num_samples=6_000, warmup=1_000, seed=31)
        xs = [value["x"] for value in posterior.samples]
        ys = [value["y"] for value in posterior.samples]
        # Exact posterior: E[x]=4/9, Var[x]=5/9 and E[y]=8/9.
        assert abs(sum(xs) / len(xs) - 4 / 9) < 0.12
        assert abs(sum(ys) / len(ys) - 8 / 9) < 0.12
        assert abs(sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs) - 5 / 9) < 0.14

    def test_dependent_bounds_raise_unsupported_model_error(self):
        @model
        def dependent_bound_model():
            x = sample(Normal(0, 1), name="x")
            y = sample(Uniform(0, math.exp(x)), name="y")
            return x, y

        with pytest.raises(ValueError, match="changed distribution metadata"):
            infer(dependent_bound_model, method="adaptive_mh", num_samples=20, warmup=0, seed=19)

    def test_rng_consumption_introduced_by_a_proposal_is_rejected(self):
        def model_fn():
            x = sample(Normal(0, 1), name="x")
            if x > 0:
                sample(Normal(0, 1), name="new_random_site")
            return x

        initial = run_with_trace(model_fn, interventions={"x": 0.0})
        with pytest.raises(ValueError, match="proposal replay that consumes RNG"):
            infer(model_fn, method="adaptive_mh", num_samples=20, warmup=0, initial_trace=initial, seed=3)

    def test_user_value_errors_during_replay_are_not_rejections(self):
        def model_fn():
            x = sample(Normal(0, 1), name="x")
            if x != 0:
                raise ValueError("user model failure")
            return x

        initial = run_with_trace(model_fn, interventions={"x": 0.0})
        with pytest.raises(ValueError, match="user model failure"):
            infer(model_fn, method="adaptive_mh", num_samples=2, warmup=0, initial_trace=initial, seed=3)

    def test_integer_kernel_and_named_block(self):
        @model
        def mixed_model():
            x = sample(Normal(0, 2), name="x")
            y = sample(Normal(0, 2), name="y")
            k = sample(Poisson(3), name="k")
            observe(Normal(x + y + k, 0.5), 5.0)
            return {"x": x, "y": y, "k": k}

        posterior = infer(
            mixed_model, method="adaptive_mh", num_samples=400, warmup=100, blocks=[("x", "y")], seed=12
        )
        assert all(float(result["k"]).is_integer() and result["k"] >= 0 for result in posterior.samples)
        assert posterior.info is not None
        assert set(posterior.info.extra["kernel_diagnostics"]) == {"1:local:('x', 'y')", "2:local:x", "3:local:y", "4:local:k"}

    def test_continuation_preserves_rng_and_adaptation_state(self):
        @model
        def normal_model():
            x = sample(Normal(0, 1), name="x")
            observe(Normal(x, 1), 0.5)
            return x

        whole = infer(normal_model, method="adaptive_mh", num_samples=300, warmup=80, seed=73)
        first = infer(normal_model, method="adaptive_mh", num_samples=150, warmup=80, seed=73)
        resumed = first.extend(normal_model, num_samples=150)
        assert resumed.samples == whole.samples
        assert resumed.info is not None
        assert resumed.info.extra["adaptation_frozen"] is True

    def test_rejects_unnamed_and_variable_structure(self):
        @model
        def unnamed():
            return sample(Normal(0, 1))

        with pytest.raises(ValueError, match="explicit unique names"):
            infer(unnamed, method="adaptive_mh", num_samples=10, seed=1)

        @model
        def variable_structure():
            branch = sample(Poisson(1), name="branch")
            if branch:
                sample(Normal(0, 1), name="child")
            return branch

        # Fixed structure is enforced on every replayed transition.  No
        # speculative probe is run because it can call arbitrary model code
        # with invalid values.
        with pytest.raises(ValueError, match="fixed structure|replay that consumes RNG"):
            infer(variable_structure, method="adaptive_mh", num_samples=10, seed=1)

    def test_duplicate_explicit_names_are_rejected(self):
        def duplicated():
            sample(Normal(0, 1), name="x")
            sample(Normal(0, 1), name="x")

        with pytest.raises(ValueError, match="Duplicate sample address"):
            run_with_trace(duplicated)

    def test_post_warmup_covariance_is_invariant_on_continuation(self):
        @model
        def normal_model():
            return sample(Normal(0, 1), name="x")

        first = infer(normal_model, method="adaptive_mh", num_samples=60, warmup=20, seed=11)
        state = first.info.extra["sampler_state"]
        before = _covariance(state.kernel_states[("local", ("x",))], 1).copy()
        resumed = first.extend(normal_model, num_samples=90, lag=3)
        after_state = resumed.info.extra["sampler_state"]
        after = _covariance(after_state.kernel_states[("local", ("x",))], 1)
        assert (before == after).all()
        assert after_state.kernel_states[("local", ("x",))].visits == state.kernel_states[("local", ("x",))].visits + 270

    def test_integer_proposals_preserve_type_and_never_leave_support(self):
        @model
        def count_model():
            k = sample(Poisson(1), name="k")
            return sum(range(k))

        posterior = infer(count_model, method="adaptive_mh", num_samples=200, warmup=50, seed=9)
        assert all(isinstance(trace.choices["k"].value, int) for trace in posterior.traces)

    def test_changed_bounds_and_initial_target_are_rejected(self):
        @model
        def bounded(high=1.0):
            return sample(Uniform(0, high), name="x")

        initial = infer(bounded, method="adaptive_mh", num_samples=5, warmup=0, seed=4).traces[-1]
        with pytest.raises(ValueError, match="(metadata changed|joint score)"):
            infer(bounded, 2.0, method="adaptive_mh", num_samples=5, initial_trace=initial, seed=4)

    def test_extreme_proposals_do_not_overflow(self):
        @model
        def bounded_model():
            return sample(Uniform(0, 1), name="x")

        posterior = infer(bounded_model, method="adaptive_mh", num_samples=20, warmup=5, seed=8)
        state = posterior.info.extra["sampler_state"]
        state.kernel_states[("local", ("x",))].log_scale = math.log(1e3)
        continued = posterior.extend(bounded_model, num_samples=10)
        assert all(0 < value < 1 for value in continued.samples)

    def test_kernel_key_collision_is_impossible(self):
        @model
        def collision_model():
            scalar = sample(Normal(0, 1), name="('x', 'y')")
            x = sample(Normal(0, 1), name="x")
            y = sample(Normal(0, 1), name="y")
            return scalar + x + y

        posterior = infer(collision_model, method="adaptive_mh", num_samples=30, warmup=10, blocks=[("x", "y")], seed=6)
        assert len(posterior.info.extra["sampler_state"].kernel_states) == 4
        diagnostics = posterior.info.extra["kernel_diagnostics"]
        assert len(diagnostics) == 4
        assert set(diagnostics) == {"1:local:('x', 'y')", "2:local:('x', 'y')", "3:local:x", "4:local:y"}

    def test_invalid_user_initial_trace_is_rejected(self):
        def model_fn():
            return sample(Normal(0, 1), name="x")

        trace = run_with_trace(model_fn, interventions={"x": math.inf})
        with pytest.raises(ValueError, match="outside replayed distribution support"):
            infer(model_fn, method="adaptive_mh", num_samples=10, seed=1, initial_trace=trace)

    def test_early_identical_samples_keep_defensible_covariance_scale(self):
        state = _KernelState(adaptation_visits=2, scatter=np.zeros((1, 1)))
        assert _covariance(state, 1)[0, 0] > 0.49

    def test_covariance_uses_final_warmup_window(self):
        @model
        def normal_model():
            return sample(Normal(0, 1), name="x")

        posterior = infer(normal_model, method="adaptive_mh", num_samples=1, warmup=20, seed=11)
        state = posterior.info.extra["sampler_state"].kernel_states[("local", ("x",))]
        # The reset occurs halfway through warmup; a single always-scheduled
        # scalar kernel consequently contributes its latter 10 states only.
        assert state.adaptation_visits == 20
        assert state.covariance_visits == 10

    def test_frozen_proposal_geometry_is_cached_and_resumable(self):
        state = _KernelState(adaptation_visits=4, scatter=np.array([[3.0]]), log_scale=math.log(2.0))
        first, first_log_det = _proposal_geometry(state, 1)
        second, second_log_det = _proposal_geometry(state, 1)
        assert first is second
        assert first_log_det == second_log_det
        restored = _KernelState(**state.__dict__)
        resumed, resumed_log_det = _proposal_geometry(restored, 1)
        assert np.array_equal(resumed, first)
        assert resumed_log_det == first_log_det

    @pytest.mark.parametrize(
        ("value", "mean", "sigma"),
        [
            (101.0, 100.0, 0.1),
            (100.0, 101.0, 0.1),
            (1_000_000.0, 1_000_001.0, 0.01),
            (-1_000_000.0, -1_000_001.0, 0.01),
        ],
    )
    def test_rounded_normal_tail_mass_matches_scipy_reference(self, value, mean, sigma):
        lower = (value - 0.5 - mean) / sigma
        upper = (value + 0.5 - mean) / sigma
        if lower > 0:
            log_large, log_small = float(log_ndtr(-lower)), float(log_ndtr(-upper))
        else:
            log_large, log_small = float(log_ndtr(upper)), float(log_ndtr(lower))
        expected = log_large if math.isinf(log_small) else log_large + math.log(-math.expm1(log_small - log_large))
        actual = _rounded_normal_log_mass(value, mean, sigma)
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)

    def test_rounded_normal_tail_mass_is_symmetric(self):
        forward = _rounded_normal_log_mass(101.0, 100.0, 0.1)
        reverse = _rounded_normal_log_mass(100.0, 101.0, 0.1)
        assert math.isfinite(forward)
        assert forward == reverse

    def test_fixed_weight_schedule_and_prior_resimulation_are_resumable(self):
        @model
        def conditional_model():
            x = sample(Normal(0, 1), name="x")
            y = sample(Normal(x, 1), name="y")
            observe(Normal(y, 1), 0.25)
            return x, y

        whole, _ = local_mh_sample(
            conditional_model, num_samples=40, warmup=10, prior_resimulation=True,
            kernel_weights=[1, 1, 2, 2], seed=91,
        )
        first, state = local_mh_sample(
            conditional_model, num_samples=20, warmup=10, prior_resimulation=True,
            kernel_weights=[1, 1, 2, 2], seed=91,
        )
        second, _ = local_mh_sample(conditional_model, num_samples=20, sampler_state=state)
        assert [trace.result for trace in [*first, *second]] == [trace.result for trace in whole]
        assert len(state.kernels) == 4

    def test_kernel_weights_are_fixed_and_validate_all_components(self):
        kernels = (LocalKernel(("x",)), PriorResimulationKernel(("x",)))
        assert _validate_kernel_weights(kernels, [1, 3]) == (0.25, 0.75)
        with pytest.raises(ValueError, match="one entry"):
            _validate_kernel_weights(kernels, [1])
        with pytest.raises(ValueError, match="uncovered"):
            _validate_kernel_weights((LocalKernel(("x",)), LocalKernel(("y",))), [1, 0])
        with pytest.raises(ValueError, match="finite positive sum"):
            _validate_kernel_weights(kernels, [float.fromhex("0x1.fffffffffffffp+1023")] * 2)
        rng = np.random.default_rng(4)
        draws = [_draw_kernel_index((0.0, 1.0), rng) for _ in range(20)]
        assert draws == [1] * 20

    def test_analytic_finite_transition_detailed_balance_and_compound_stationarity(self):
        # Target pi=(1/3, 2/3).  Each target-aware assessment contains the
        # complete forward/reverse proposal accounting for its transition.
        pi = np.array([1 / 3, 2 / 3])
        q = np.array([[0.7, 0.3], [0.4, 0.6]])
        transition = np.zeros((2, 2))
        for old in range(2):
            for new in range(2):
                log_alpha = math.log(pi[new] / pi[old]) + math.log(q[new, old] / q[old, new])
                transition[old, new] = q[old, new] * min(1.0, math.exp(log_alpha))
            transition[old, old] += 1 - transition[old].sum()
        assert np.allclose(pi[:, None] * transition, (pi[:, None] * transition).T)
        # A deterministic composition of valid kernels need not be reversible,
        # but it preserves stationarity.
        composed = transition @ transition
        assert np.allclose(pi @ composed, pi)

    def test_prior_resimulation_has_no_hidden_replay_rng(self):
        def model_fn():
            return sample(Normal(0, 1), name="x")

        initial = run_with_trace(model_fn, interventions={"x": 0.0})
        traces, _ = local_mh_sample(
            model_fn, num_samples=3, warmup=0, initial_trace=initial,
            prior_resimulation=True, kernel_weights=[0, 1], seed=7,
        )
        assert len(traces) == 3

    def test_hierarchical_beta_gamma_parameters_are_rescored(self):
        @model
        def hierarchical():
            x = sample(Normal(0, 1), name="x")
            p = sample(Beta(math.exp(x), 1), name="p")
            g = sample(Gamma(math.exp(x), 1), name="g")
            observe(Normal(p + g, 1), 1.0)
            return x, p, g

        posterior = infer(hierarchical, method="adaptive_mh", num_samples=50, warmup=10, seed=5)
        assert all(0 < value[1] < 1 and value[2] > 0 for value in posterior.samples)

    def test_prior_and_local_names_are_collision_proof(self):
        @model
        def collision_model():
            return sample(Normal(0, 1), name="x")

        info = {}
        _, state = local_mh_sample(
            collision_model, num_samples=4, warmup=0, prior_resimulation=True, seed=8, _info=info,
        )
        diagnostics = info["kernel_diagnostics"]
        assert set(diagnostics) == {"1:local:x", "2:prior:x"}
        assert _kernel_key(LocalKernel(("x",))) != _kernel_key(PriorResimulationKernel(("x",)))
        assert len(state.kernel_states) == 2

    def test_structural_prior_resimulation_fails_at_supported_boundary(self):
        def changing_model():
            branch = sample(Poisson(1), name="branch")
            if branch:
                sample(Normal(0, 1), name="child")
            return branch

        initial = run_with_trace(changing_model, interventions={"branch": 0})
        with pytest.raises(ValueError, match="prior resimulation requires fixed structure"):
            local_mh_sample(
                changing_model, num_samples=30, warmup=0, initial_trace=initial,
                prior_resimulation=True, kernel_weights=[0, 1], seed=7,
            )
