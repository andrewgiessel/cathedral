"""Tests for Cathedral primitives."""

import math

import numpy as np
import pytest

from cathedral.distributions import Bernoulli, Normal
from cathedral.primitives import condition, factor, flip, observe, sample
from cathedral.trace import Rejected, run_with_trace


class TestSample:
    def test_sample_without_trace(self):
        """sample() works outside of a tracing context."""
        np.random.seed(42)
        val = sample(Normal(0, 1))
        assert isinstance(val, float)

    def test_sample_with_trace(self):
        def model():
            return sample(Normal(0, 1), name="x")

        trace = run_with_trace(model)
        assert "x" in trace.choices
        assert isinstance(trace.result, float)

    def test_sample_auto_address(self):
        def model():
            a = sample(Normal(0, 1))
            b = sample(Normal(0, 1))
            return a + b

        trace = run_with_trace(model)
        assert len(trace) == 2
        addresses = trace.addresses
        assert addresses[0] == "Normal"
        assert addresses[1] == "Normal__1"

    def test_sample_explicit_name(self):
        def model():
            return sample(Normal(0, 1), name="my_var")

        trace = run_with_trace(model)
        assert "my_var" in trace.choices


class TestFlip:
    def test_flip_without_trace(self):
        np.random.seed(42)
        val = flip(0.5)
        assert isinstance(val, bool)

    def test_flip_with_trace(self):
        def model():
            return flip(0.7)

        trace = run_with_trace(model)
        assert len(trace) == 1
        assert trace.result in (True, False)

    def test_flip_deterministic_p1(self):
        np.random.seed(42)
        for _ in range(10):
            assert flip(1.0) is True

    def test_flip_deterministic_p0(self):
        np.random.seed(42)
        for _ in range(10):
            assert flip(0.0) is False


class TestCondition:
    def test_condition_true_passes(self):
        def model():
            condition(True)
            return 42

        trace = run_with_trace(model)
        assert trace.result == 42

    def test_condition_false_rejects(self):
        def model():
            condition(False)
            return 42

        with pytest.raises(Rejected):
            run_with_trace(model)

    def test_condition_false_adds_neg_inf_score(self):
        """When condition fails, it adds -inf to score before raising."""

        def model():
            condition(False)

        with pytest.raises(Rejected):
            run_with_trace(model)


class TestObserve:
    def test_observe_adds_log_prob(self):
        def model():
            observe(Normal(0, 1), 0.0)
            return True

        trace = run_with_trace(model)
        expected = Normal(0, 1).log_prob(0.0)
        assert math.isclose(trace.log_score, expected, rel_tol=1e-6)

    def test_observe_multiple(self):
        def model():
            observe(Normal(0, 1), 0.0)
            observe(Normal(0, 1), 1.0)
            return True

        trace = run_with_trace(model)
        expected = Normal(0, 1).log_prob(0.0) + Normal(0, 1).log_prob(1.0)
        assert math.isclose(trace.log_score, expected, rel_tol=1e-6)

    def test_observe_without_trace(self):
        """observe() is a no-op outside tracing context."""
        observe(Normal(0, 1), 0.0)  # should not raise


class TestFactor:
    def test_factor_adds_score(self):
        def model():
            factor(-2.5)
            return True

        trace = run_with_trace(model)
        assert math.isclose(trace.log_score, -2.5)

    def test_factor_without_trace(self):
        """factor() is a no-op outside tracing context."""
        factor(-1.0)  # should not raise
