"""Tests for sequential independent multi-chain execution."""

import numpy as np
import pytest

from cathedral import model
from cathedral.chains import diagnose_chains, infer_chains
from cathedral.distributions import Normal
from cathedral.primitives import observe, sample


@model
def normal_location():
    value = sample(Normal(0.0, 1.0), name="value")
    observe(Normal(value, 1.0), 0.5)
    return {"value": value, "squared": value**2}


def test_infer_chains_reproducible_and_preserves_axes():
    first = infer_chains(normal_location, num_chains=2, num_samples=30, warmup=20, seed=9182)
    second = infer_chains(normal_location, num_chains=2, num_samples=30, warmup=20, seed=9182)

    first_values = first.values(lambda result: result["value"])
    second_values = second.values(lambda result: result["value"])
    assert first_values.shape == (2, 30)
    np.testing.assert_array_equal(first_values, second_values)
    assert not np.array_equal(first_values[0], first_values[1])


def test_named_query_summary_and_mapping():
    posterior = infer_chains(normal_location, num_chains=2, num_samples=40, warmup=20, seed=7)
    summary = posterior.summarize(lambda result: result["value"], name="location")
    summaries = posterior.summarize({"location": lambda result: result["value"]})

    assert summary.name == "location"
    assert np.isfinite(summary.mean)
    assert summary.diagnostics.status == "ok"
    assert summaries["location"].name == "location"


def test_complex_queries_are_rejected_without_losing_imaginary_values():
    posterior = infer_chains(normal_location, num_chains=2, num_samples=8, warmup=4, seed=7)
    with pytest.raises(TypeError, match="real numeric"):
        posterior.summarize(lambda result: result["value"] + 2j, name="complex")
    with pytest.raises(TypeError, match="real-valued"):
        diagnose_chains(np.ones((2, 8), dtype=complex) * (1 + 2j))
