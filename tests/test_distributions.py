"""Tests for Cathedral distributions."""

import math

import numpy as np
import pytest

from cathedral.distributions import (
    Bernoulli,
    Beta,
    Categorical,
    Dirichlet,
    Gamma,
    HalfNormal,
    Normal,
    Poisson,
    Uniform,
)


class TestBernoulli:
    def test_sample_returns_bool(self):
        d = Bernoulli(0.5)
        assert isinstance(d.sample(), bool)

    def test_log_prob_true(self):
        d = Bernoulli(0.7)
        assert math.isclose(d.log_prob(True), math.log(0.7))

    def test_log_prob_false(self):
        d = Bernoulli(0.7)
        assert math.isclose(d.log_prob(False), math.log(0.3))

    def test_edge_p_zero(self):
        d = Bernoulli(0.0)
        assert d.log_prob(True) == -math.inf
        assert d.log_prob(False) == 0.0

    def test_edge_p_one(self):
        d = Bernoulli(1.0)
        assert d.log_prob(True) == 0.0
        assert d.log_prob(False) == -math.inf

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            Bernoulli(-0.1)
        with pytest.raises(ValueError):
            Bernoulli(1.1)

    def test_sampling_frequency(self):
        np.random.seed(42)
        d = Bernoulli(0.3)
        samples = [d.sample() for _ in range(10000)]
        freq = sum(samples) / len(samples)
        assert abs(freq - 0.3) < 0.03


class TestNormal:
    def test_sample_returns_float(self):
        d = Normal(0, 1)
        assert isinstance(d.sample(), float)

    def test_log_prob_at_mean(self):
        d = Normal(0, 1)
        lp = d.log_prob(0.0)
        expected = -0.5 * math.log(2 * math.pi)
        assert math.isclose(lp, expected, rel_tol=1e-6)

    def test_log_prob_symmetry(self):
        d = Normal(0, 1)
        assert math.isclose(d.log_prob(1.0), d.log_prob(-1.0))

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            Normal(0, 0)
        with pytest.raises(ValueError):
            Normal(0, -1)


class TestHalfNormal:
    def test_sample_positive(self):
        np.random.seed(42)
        d = HalfNormal(1.0)
        for _ in range(100):
            assert d.sample() >= 0

    def test_log_prob_negative_is_neg_inf(self):
        d = HalfNormal(1.0)
        assert d.log_prob(-1.0) == -math.inf


class TestBeta:
    def test_sample_in_unit_interval(self):
        np.random.seed(42)
        d = Beta(2, 5)
        for _ in range(100):
            s = d.sample()
            assert 0 <= s <= 1

    def test_log_prob_outside_bounds(self):
        d = Beta(1, 1)
        assert d.log_prob(-0.1) == -math.inf
        assert d.log_prob(1.1) == -math.inf

    def test_uniform_special_case(self):
        d = Beta(1, 1)
        assert math.isclose(d.log_prob(0.5), 0.0, abs_tol=1e-10)


class TestGamma:
    def test_sample_positive(self):
        np.random.seed(42)
        d = Gamma(2, 1)
        for _ in range(100):
            assert d.sample() > 0

    def test_log_prob_negative_is_neg_inf(self):
        d = Gamma(1, 1)
        assert d.log_prob(-1.0) == -math.inf


class TestUniform:
    def test_sample_in_bounds(self):
        np.random.seed(42)
        d = Uniform(2.0, 5.0)
        for _ in range(100):
            s = d.sample()
            assert 2.0 <= s <= 5.0

    def test_log_prob_in_bounds(self):
        d = Uniform(0, 10)
        expected = -math.log(10)
        assert math.isclose(d.log_prob(5.0), expected)

    def test_log_prob_out_of_bounds(self):
        d = Uniform(0, 1)
        assert d.log_prob(-0.1) == -math.inf
        assert d.log_prob(1.1) == -math.inf


class TestCategorical:
    def test_sample_from_values(self):
        d = Categorical(["a", "b", "c"], [0.2, 0.3, 0.5])
        for _ in range(100):
            assert d.sample() in ["a", "b", "c"]

    def test_log_prob(self):
        d = Categorical(["a", "b"], [0.3, 0.7])
        assert math.isclose(d.log_prob("a"), math.log(0.3))
        assert math.isclose(d.log_prob("b"), math.log(0.7))

    def test_log_prob_unknown_value(self):
        d = Categorical(["a", "b"], [0.5, 0.5])
        assert d.log_prob("c") == -math.inf


class TestPoisson:
    def test_sample_non_negative_int(self):
        np.random.seed(42)
        d = Poisson(3.0)
        for _ in range(100):
            s = d.sample()
            assert isinstance(s, int)
            assert s >= 0

    def test_log_prob_negative(self):
        d = Poisson(1.0)
        assert d.log_prob(-1) == -math.inf


class TestDirichlet:
    def test_sample_sums_to_one(self):
        np.random.seed(42)
        d = Dirichlet([1.0, 1.0, 1.0])
        s = d.sample()
        assert math.isclose(s.sum(), 1.0, rel_tol=1e-6)

    def test_sample_all_positive(self):
        np.random.seed(42)
        d = Dirichlet([2.0, 3.0])
        s = d.sample()
        assert all(x > 0 for x in s)
