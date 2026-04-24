"""Tests for cathedral.checks: prior/posterior predictive, conditioning, model comparison."""

import numpy as np
import pytest

from cathedral import model
from cathedral.checks import (
    compare_models,
    condition_acceptance_rate,
    posterior_predictive,
    prior_predictive,
)
from cathedral.distributions import Normal
from cathedral.model import Posterior, infer
from cathedral.primitives import condition, flip, observe, sample


class TestPriorPredictive:
    def test_returns_posterior(self):
        @model
        def coin():
            return flip(0.5)

        pp = prior_predictive(coin, num_samples=100)
        assert isinstance(pp, Posterior)
        assert pp.num_samples == 100

    def test_info_method(self):
        @model
        def coin():
            return flip(0.5)

        pp = prior_predictive(coin, num_samples=100)
        assert pp.info is not None
        assert pp.info.method == "prior_predictive"

    def test_acceptance_rate_no_condition(self):
        @model
        def coin():
            return flip(0.5)

        pp = prior_predictive(coin, num_samples=100)
        assert pp.info.acceptance_rate == 1.0

    def test_acceptance_rate_with_condition(self):
        np.random.seed(42)

        @model
        def conditioned():
            x = flip(0.5)
            condition(x)
            return x

        pp = prior_predictive(conditioned, num_samples=1000)
        assert pp.info.acceptance_rate is not None
        assert 0.4 < pp.info.acceptance_rate < 0.6

    def test_all_rejected_raises(self):
        @model
        def impossible():
            condition(False)
            return True

        with pytest.raises(RuntimeError, match="all forward samples were rejected"):
            prior_predictive(impossible, num_samples=100)

    def test_with_args(self):
        @model
        def biased(p):
            return flip(p)

        pp = prior_predictive(biased, 0.9, num_samples=200)
        mean_val = pp.mean()
        assert 0.7 < mean_val < 1.0

    def test_unwraps_model_decorator(self):
        @model
        def coin():
            return flip(0.7)

        pp = prior_predictive(coin, num_samples=500)
        assert 0.5 < pp.mean() < 0.9

    def test_seed_reproducible(self):
        @model
        def coin():
            return flip(0.5)

        np.random.seed(1)
        pp_a = prior_predictive(coin, num_samples=100, seed=123)
        np.random.seed(999)
        pp_b = prior_predictive(coin, num_samples=100, seed=123)

        assert pp_a.samples == pp_b.samples


class TestConditionAcceptanceRate:
    def test_no_condition(self):
        @model
        def coin():
            return flip(0.5)

        rate = condition_acceptance_rate(coin, num_samples=100)
        assert rate == 1.0

    def test_with_condition(self):
        np.random.seed(42)

        @model
        def conditioned():
            x = flip(0.3)
            condition(x)
            return x

        rate = condition_acceptance_rate(conditioned, num_samples=5000)
        assert 0.25 < rate < 0.35

    def test_impossible_condition(self):
        @model
        def impossible():
            condition(False)
            return True

        rate = condition_acceptance_rate(impossible, num_samples=100)
        assert rate == 0.0

    def test_with_args(self):
        np.random.seed(42)

        @model
        def threshold(t):
            x = sample(Normal(0, 1))
            condition(x > t)
            return x

        rate_easy = condition_acceptance_rate(threshold, 0.0, num_samples=5000)
        rate_hard = condition_acceptance_rate(threshold, 2.0, num_samples=5000)
        assert rate_easy > rate_hard


class TestPosteriorPredictive:
    def test_returns_posterior(self):
        np.random.seed(42)

        @model
        def obs():
            mu = sample(Normal(0, 5), name="mu")
            observe(Normal(mu, 1), 3.0)
            return mu

        post = infer(obs, method="importance", num_samples=200)
        pp = posterior_predictive(post, obs, num_samples=50)
        assert isinstance(pp, Posterior)
        assert pp.num_samples == 50

    def test_info_method(self):
        np.random.seed(42)

        @model
        def obs():
            mu = sample(Normal(0, 5), name="mu")
            observe(Normal(mu, 1), 3.0)
            return mu

        post = infer(obs, method="importance", num_samples=200)
        pp = posterior_predictive(post, obs, num_samples=50)
        assert pp.info is not None
        assert pp.info.method == "posterior_predictive"

    def test_default_num_samples(self):
        np.random.seed(42)

        @model
        def coin():
            return flip(0.5)

        post = infer(coin, method="rejection", num_samples=100)
        pp = posterior_predictive(post, coin)
        assert pp.num_samples == 100

    def test_default_num_samples_capped(self):
        np.random.seed(42)

        @model
        def coin():
            return flip(0.5)

        post = infer(coin, method="rejection", num_samples=1000)
        pp = posterior_predictive(post, coin)
        assert pp.num_samples == 500

    def test_predictive_values_reasonable(self):
        np.random.seed(42)

        @model
        def obs():
            mu = sample(Normal(0, 5), name="mu")
            observe(Normal(mu, 1), 3.0)
            observe(Normal(mu, 1), 3.5)
            return mu

        post = infer(obs, method="importance", num_samples=500)
        pp = posterior_predictive(post, obs, num_samples=100)
        assert 1.0 < pp.mean() < 5.0

    def test_seed_reproducible(self):
        @model
        def obs():
            mu = sample(Normal(0, 5), name="mu")
            observe(Normal(mu, 1), 3.0)
            return mu

        post = infer(obs, method="importance", num_samples=200, seed=77)

        np.random.seed(1)
        pp_a = posterior_predictive(post, obs, num_samples=50, seed=123)
        np.random.seed(999)
        pp_b = posterior_predictive(post, obs, num_samples=50, seed=123)

        assert pp_a.samples == pp_b.samples


class TestCompareModels:
    def test_basic_comparison(self):
        @model
        def model_a():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            condition(result)
            return fair

        @model
        def model_b():
            result = flip(0.9)
            condition(result)
            return True

        pa = infer(model_a, method="enumerate")
        pb = infer(model_b, method="enumerate")

        result = compare_models({"model_a": pa, "model_b": pb})
        assert "Model Comparison" in result
        assert "model_a" in result
        assert "model_b" in result
        assert "log ML" in result
        assert "Bayes Factors" in result
        assert "log BF" in result

    def test_missing_log_ml(self):
        @model
        def coin():
            x = flip(0.5)
            condition(x)
            return x

        p = infer(coin, method="rejection", num_samples=100)
        result = compare_models({"coin": p})
        assert "not available" in result

    def test_single_model_no_bayes_factors(self):
        @model
        def coin():
            return flip(0.6)

        p = infer(coin, method="enumerate")
        result = compare_models({"coin": p})
        assert "Bayes Factors" not in result

    def test_evidence_direction(self):
        @model
        def model_a():
            result = flip(0.9)
            condition(result)
            return True

        @model
        def model_b():
            result = flip(0.1)
            condition(result)
            return True

        pa = infer(model_a, method="enumerate")
        pb = infer(model_b, method="enumerate")

        result = compare_models({"high_prob": pa, "low_prob": pb})
        assert "Strong evidence for high_prob" in result
