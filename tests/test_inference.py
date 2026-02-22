"""Tests for Cathedral inference engines."""

import numpy as np
import pytest

from cathedral import model
from cathedral.distributions import Normal
from cathedral.inference.importance import importance_sample
from cathedral.inference.rejection import rejection_sample
from cathedral.primitives import condition, flip, observe, sample


class TestRejectionSampling:
    def test_simple_condition(self):
        """Rejection sampling should only return traces where condition holds."""

        def model_fn():
            x = flip(0.5)
            condition(x)
            return x

        traces = rejection_sample(model_fn, num_samples=100)
        assert len(traces) == 100
        assert all(t.result is True for t in traces)

    def test_sprinkler(self):
        """Classic sprinkler example from Goodman et al. 2008."""
        np.random.seed(42)

        def sprinkler():
            rain = flip(0.3)
            sprinkler_on = flip(0.5)
            if rain:
                wet = flip(0.9)
            elif sprinkler_on:
                wet = flip(0.8)
            else:
                wet = flip(0.1)
            condition(wet)
            return {"rain": rain, "sprinkler": sprinkler_on}

        traces = rejection_sample(sprinkler, num_samples=5000)
        rain_prob = sum(t.result["rain"] for t in traces) / len(traces)
        # P(rain | wet) should be around 0.36-0.55
        assert 0.30 < rain_prob < 0.60

    def test_max_attempts_exceeded(self):
        """Should raise RuntimeError if condition is too rare."""

        def impossible():
            condition(False)
            return True

        with pytest.raises(RuntimeError, match="Rejection sampling"):
            rejection_sample(impossible, num_samples=10, max_attempts=100)

    def test_with_args(self):
        """Model functions can accept arguments."""

        def biased_coin(p):
            result = flip(p)
            condition(result)
            return result

        traces = rejection_sample(biased_coin, args=(0.9,), num_samples=50)
        assert len(traces) == 50


class TestImportanceSampling:
    def test_all_observed_data(self):
        """Importance sampling should work with observe()."""
        np.random.seed(42)

        def coin_model():
            p = sample(Normal(0.5, 0.2))
            p = max(0.01, min(0.99, p))
            observe(Normal(p, 0.1), 0.7)
            return p

        traces = importance_sample(coin_model, num_samples=5000, resample=True)
        mean_p = np.mean([t.result for t in traces])
        # Should be pulled toward the observed 0.7
        assert 0.5 < mean_p < 0.8

    def test_linear_regression(self):
        """Importance sampling should recover approximate slope."""
        np.random.seed(42)
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ys = np.array([2.1, 4.0, 6.2, 7.9, 10.1])

        def line_model(xs, ys):
            slope = sample(Normal(0, 5))
            intercept = sample(Normal(0, 5))
            for x, y in zip(xs, ys, strict=False):
                observe(Normal(slope * x + intercept, 0.5), y)
            return {"slope": slope, "intercept": intercept}

        traces = importance_sample(line_model, args=(xs, ys), num_samples=10000)
        mean_slope = np.mean([t.result["slope"] for t in traces])
        assert 1.5 < mean_slope < 2.5

    def test_no_resample(self):
        """Without resampling, returns all traces."""

        def simple():
            x = sample(Normal(0, 1))
            observe(Normal(x, 0.1), 2.0)
            return x

        traces = importance_sample(simple, num_samples=100, resample=False)
        assert len(traces) == 100

    def test_impossible_observation(self):
        """Should raise if all samples are rejected via condition()."""

        def impossible():
            condition(False)
            return True

        with pytest.raises(RuntimeError, match="all samples were rejected"):
            importance_sample(impossible, num_samples=100)


class TestPosterior:
    def test_mean(self):
        np.random.seed(42)

        @model
        def simple():
            return flip(0.7)

        from cathedral.model import infer

        posterior = infer(simple, method="rejection", num_samples=5000)
        mean_val = posterior.mean()
        assert 0.60 < mean_val < 0.80

    def test_mean_with_key(self):
        np.random.seed(42)

        @model
        def two_coins():
            return {"a": flip(0.3), "b": flip(0.8)}

        from cathedral.model import infer

        posterior = infer(two_coins, method="rejection", num_samples=5000)
        assert 0.20 < posterior.mean("a") < 0.40
        assert 0.70 < posterior.mean("b") < 0.90

    def test_probability_with_key(self):
        np.random.seed(42)

        @model
        def coin():
            return {"heads": flip(0.6)}

        from cathedral.model import infer

        posterior = infer(coin, method="rejection", num_samples=5000)
        p = posterior.probability("heads")
        assert 0.50 < p < 0.70

    def test_probability_with_callable(self):
        np.random.seed(42)

        @model
        def die():
            return sample(Normal(5, 1))

        from cathedral.model import infer

        posterior = infer(die, method="rejection", num_samples=5000)
        p_gt_5 = posterior.probability(lambda x: x > 5)
        assert 0.40 < p_gt_5 < 0.60

    def test_histogram(self):
        np.random.seed(42)

        @model
        def coin():
            return flip(0.5)

        from cathedral.model import infer

        posterior = infer(coin, method="rejection", num_samples=5000)
        hist = posterior.histogram()
        assert True in hist
        assert False in hist
        assert abs(hist[True] - 0.5) < 0.05

    def test_credible_interval(self):
        np.random.seed(42)

        @model
        def normal_model():
            return sample(Normal(0, 1))

        from cathedral.model import infer

        posterior = infer(normal_model, method="rejection", num_samples=10000)
        lo, hi = posterior.credible_interval(level=0.95)
        assert lo < -1.5
        assert hi > 1.5

    def test_std(self):
        np.random.seed(42)

        @model
        def normal_model():
            return sample(Normal(0, 1))

        from cathedral.model import infer

        posterior = infer(normal_model, method="rejection", num_samples=10000)
        assert 0.8 < posterior.std() < 1.2

    def test_num_samples(self):
        @model
        def coin():
            return flip(0.5)

        from cathedral.model import infer

        posterior = infer(coin, method="rejection", num_samples=100)
        assert posterior.num_samples == 100
