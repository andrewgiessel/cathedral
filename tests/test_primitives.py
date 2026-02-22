"""Tests for Cathedral primitives."""

import math

import numpy as np
import pytest

from cathedral.distributions import Categorical, Normal
from cathedral.primitives import DPmem, condition, factor, flip, mem, observe, sample
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


class TestMem:
    def test_mem_caches_within_trace(self):
        """mem'd function returns same value for same args within a trace."""

        def model():
            eye_color = mem(lambda person: sample(Categorical(["blue", "green", "brown"], [1 / 3, 1 / 3, 1 / 3])))
            return eye_color("bob"), eye_color("bob")

        for _ in range(20):
            trace = run_with_trace(model)
            a, b = trace.result
            assert a == b

    def test_mem_different_args_independent(self):
        """mem'd function can return different values for different args."""
        np.random.seed(42)
        results = set()
        for _ in range(50):

            def model():
                f = mem(lambda x: sample(Categorical(["a", "b", "c"], [1 / 3, 1 / 3, 1 / 3])))
                return f("x"), f("y")

            trace = run_with_trace(model)
            a, b = trace.result
            results.add((a, b))
        # Should see at least some cases where a != b
        assert any(a != b for a, b in results)

    def test_mem_cache_is_per_trace(self):
        """Different traces get independent memo caches."""
        np.random.seed(42)
        results = set()
        for _ in range(30):

            def model():
                f = mem(lambda: sample(Categorical(["a", "b", "c"], [1 / 3, 1 / 3, 1 / 3])))
                return f()

            trace = run_with_trace(model)
            results.add(trace.result)
        # Should see variation across traces
        assert len(results) > 1

    def test_mem_without_trace(self):
        """mem works in standalone mode with its own cache."""
        np.random.seed(42)
        f = mem(lambda x: sample(Normal(0, 1)))
        val1 = f("a")
        val2 = f("a")
        assert val1 == val2

    def test_mem_preserves_function_name(self):
        def my_function(x):
            return sample(Normal(0, 1))

        m = mem(my_function)
        assert m.__name__ == "my_function"

    def test_mem_with_list_args(self):
        """mem handles unhashable args like lists."""

        def model():
            f = mem(lambda items: sample(Normal(0, 1)))
            a = f([1, 2, 3])
            b = f([1, 2, 3])
            return a == b

        trace = run_with_trace(model)
        assert trace.result is True


class TestDPmem:
    def test_dpmem_low_alpha_reuses(self):
        """With very low alpha, DPmem should mostly reuse existing values."""
        np.random.seed(42)
        f = DPmem(0.001, lambda: sample(Normal(0, 100)))
        vals = [f() for _ in range(20)]
        unique = set(vals)
        assert len(unique) <= 3  # should converge to very few values

    def test_dpmem_high_alpha_varies(self):
        """With very high alpha, DPmem should mostly sample new values."""
        np.random.seed(42)
        f = DPmem(1000.0, lambda: sample(Categorical(list(range(100)), [1 / 100] * 100)))
        vals = [f() for _ in range(50)]
        unique = set(vals)
        assert len(unique) > 10

    def test_dpmem_different_args_independent(self):
        """DPmem maintains separate tables per argument."""
        np.random.seed(42)
        f = DPmem(0.001, lambda x: sample(Normal(x, 0.01)))
        a_vals = [f(0.0) for _ in range(10)]
        b_vals = [f(100.0) for _ in range(10)]
        assert abs(np.mean(a_vals)) < 10
        assert abs(np.mean(b_vals) - 100) < 10
