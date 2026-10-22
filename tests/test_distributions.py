"""Tests for Cathedral distributions."""

import math

import numpy as np
import pytest

from cathedral import Multinomial as PublicMultinomial
from cathedral.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    Exponential,
    Gamma,
    HalfNormal,
    LogNormal,
    Multinomial,
    Normal,
    Poisson,
    StudentT,
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


class TestExponential:
    def test_sample_positive(self):
        d = Exponential(2.0)
        for _ in range(100):
            assert d.sample() >= 0

    def test_log_prob_at_zero(self):
        d = Exponential(2.0)
        assert math.isclose(d.log_prob(0.0), math.log(2.0))

    def test_log_prob_negative_is_neg_inf(self):
        d = Exponential(1.0)
        assert d.log_prob(-1.0) == -math.inf

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            Exponential(0)


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


class TestLogNormal:
    def test_sample_positive(self):
        d = LogNormal(0, 1)
        for _ in range(100):
            assert d.sample() > 0

    def test_log_prob_non_positive_is_neg_inf(self):
        d = LogNormal(0, 1)
        assert d.log_prob(0.0) == -math.inf
        assert d.log_prob(-1.0) == -math.inf

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            LogNormal(0, 0)


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


class TestBinomial:
    def test_sample_in_support(self):
        d = Binomial(10, 0.3)
        for _ in range(100):
            s = d.sample()
            assert isinstance(s, int)
            assert 0 <= s <= 10

    def test_support(self):
        assert Binomial(3, 0.5).support() == [0, 1, 2, 3]

    def test_log_prob_outside_support(self):
        d = Binomial(3, 0.5)
        assert d.log_prob(-1) == -math.inf
        assert d.log_prob(4) == -math.inf
        assert d.log_prob(1.5) == -math.inf

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            Binomial(-1, 0.5)
        with pytest.raises(ValueError):
            Binomial(3, -0.1)


class TestMultinomial:
    def test_sample_is_valid_count_vector(self):
        d = Multinomial(10, [0.2, 0.3, 0.5])
        for _ in range(100):
            sample = d.sample()
            assert isinstance(sample, np.ndarray)
            assert sample.shape == (3,)
            assert np.issubdtype(sample.dtype, np.integer)
            assert np.all(sample >= 0)
            assert sample.sum() == 10

    def test_log_prob(self):
        d = Multinomial(3, [0.25, 0.75])
        expected = math.log(3 * 0.25**2 * 0.75)
        assert math.isclose(d.log_prob([2, 1]), expected)

    def test_zero_trials_and_zero_probability_categories(self):
        d = Multinomial(0, [0.0, 0.25, 0.75])
        np.testing.assert_array_equal(d.sample(), [0, 0, 0])
        assert d.log_prob([0, 0, 0]) == 0.0
        assert Multinomial(2, [0.0, 1.0]).log_prob([1, 1]) == -math.inf

    @pytest.mark.parametrize("n", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_trial_count(self, n):
        with pytest.raises(ValueError, match="finite non-negative integer"):
            Multinomial(n, [0.5, 0.5])

    @pytest.mark.parametrize("probs", [[0.2, 0.3, 0.5000002], [0.2, 0.3, 0.4999998]])
    def test_normalizes_probabilities_accepted_within_sum_tolerance(self, probs):
        d = Multinomial(4, probs)
        assert math.isclose(float(d.probs.sum()), 1.0, abs_tol=1e-15)
        for _ in range(10):
            assert d.sample().sum() == 4
        assert math.isfinite(d.log_prob([1, 1, 2]))

    def test_not_enumerable_and_exported_at_package_top_level(self):
        assert PublicMultinomial is Multinomial
        assert Multinomial(1, [0.5, 0.5]).support() is None

    def test_seeded_inference_samples_array_values_consistently(self):
        from cathedral import infer, model, sample

        @model
        def counts():
            return sample(Multinomial(5, [0.2, 0.3, 0.5]), name="counts")

        first = infer(counts, method="importance", num_samples=20, seed=71)
        second = infer(counts, method="importance", num_samples=20, seed=71)
        for left, right in zip(first.samples, second.samples, strict=True):
            np.testing.assert_array_equal(left, right)
            assert left.sum() == 5
        for trace in first.traces:
            assert math.isclose(trace.log_joint, Multinomial(5, [0.2, 0.3, 0.5]).log_prob(trace.result))

    @pytest.mark.parametrize(
        "counts",
        [
            [-1, 4],
            [1.5, 1.5],
            [1, 1],
            [1, 1, 1],
            [math.nan, 3],
            ["invalid", "counts"],
            [1 + 0j, 2 + 0j],
            [10**400, 0],
        ],
    )
    def test_log_prob_outside_support(self, counts):
        assert Multinomial(3, [0.25, 0.75]).log_prob(counts) == -math.inf

    @pytest.mark.parametrize(
        ("n", "probs"),
        [
            (-1, [0.5, 0.5]),
            (1.5, [0.5, 0.5]),
            (3, []),
            (3, [[0.5, 0.5]]),
            (3, [0.5, -0.5, 1.0]),
            (3, [0.25, 0.25]),
            (3, [0.5, math.nan]),
            (3, [1 + 0j, 0 + 0j]),
        ],
    )
    def test_invalid_params(self, n, probs):
        with pytest.raises(ValueError):
            Multinomial(n, probs)


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


class TestStudentT:
    def test_sample_returns_float(self):
        d = StudentT(3)
        assert isinstance(d.sample(), float)

    def test_log_prob_at_location(self):
        d = StudentT(3, loc=2.0, scale=1.0)
        assert math.isfinite(d.log_prob(2.0))

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            StudentT(0)
        with pytest.raises(ValueError):
            StudentT(3, scale=0)
