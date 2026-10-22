"""Public integration coverage for adaptive inference and chain APIs."""

import pytest

from cathedral import infer, infer_chains, model
from cathedral.distributions import Normal
from cathedral.inference import Kernel, PriorResimulationKernel, ProposalAssessment, ReplayTarget
from cathedral.primitives import observe, sample


@model
def _normal_model(offset=0.25):
    value = sample(Normal(0, 1), name="value")
    observe(Normal(value, 1), offset)
    return value


def test_public_adaptive_mixture_schedule_and_kernel_interfaces():
    posterior = infer(
        _normal_model,
        method="adaptive_mh",
        num_samples=12,
        warmup=4,
        prior_resimulation=True,
        kernel_weights=[1, 3],
        max_init_attempts=20,
        seed=72,
    )

    assert posterior.num_samples == 12
    assert posterior.info is not None
    assert posterior.info.extra["total_steps"] == 16
    assert set(posterior.info.extra["kernel_diagnostics"]) == {"1:local:value", "2:prior:value"}
    assert isinstance(posterior.info.extra["sampler_state"].kernels[1], PriorResimulationKernel)
    assert isinstance(ProposalAssessment(None, 0.0, 0.0).log_acceptance_ratio(posterior.traces[-1]), float)
    assert hasattr(Kernel, "step")
    assert ReplayTarget.__name__ == "ReplayTarget"


def test_infer_chains_is_available_at_top_level():
    chains = infer_chains(_normal_model, num_chains=2, num_samples=8, warmup=3, seed=19)
    assert chains.num_chains == 2
    assert chains.values(lambda value: value).shape == (2, 8)


def test_prior_mixture_continuation_with_lag_is_exact_and_cumulative():
    whole = infer(
        _normal_model,
        method="adaptive_mh",
        num_samples=16,
        warmup=5,
        lag=3,
        prior_resimulation=True,
        kernel_weights=[2, 1],
        seed=917,
    )
    first = infer(
        _normal_model,
        method="adaptive_mh",
        num_samples=7,
        warmup=5,
        lag=3,
        prior_resimulation=True,
        kernel_weights=[2, 1],
        seed=917,
    )
    resumed = first.extend(_normal_model, num_samples=9, lag=3)

    assert resumed.samples == whole.samples
    assert resumed.info is not None and whole.info is not None
    assert resumed.info.extra["total_steps"] == whole.info.extra["total_steps"] == 53
    assert resumed.info.acceptance_rate == whole.info.acceptance_rate
    assert resumed.info.extra["mean_squared_jump_distance"] == pytest.approx(
        whole.info.extra["mean_squared_jump_distance"]
    )
    assert resumed.info.extra["sampler_state"].step == 53


@pytest.mark.parametrize(
    "method, options",
    [
        ("rejection", {}),
        ("importance", {}),
        ("mh", {}),
        ("adaptive_mh", {"warmup": 0}),
        ("enumerate", {}),
    ],
)
def test_unknown_inference_kwargs_raise(method, options):
    with pytest.raises(TypeError, match="Unsupported keyword"):
        infer(_normal_model, method=method, num_samples=2, unsupported_option=True, **options)
