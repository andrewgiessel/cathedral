"""Tests for Cathedral inference engines."""

import numpy as np
import pytest

from cathedral import model
from cathedral.distributions import Categorical, Normal, UniformDraw
from cathedral.inference.enumeration import enumerate_executions, marginals_from_traces
from cathedral.inference.importance import importance_sample
from cathedral.inference.mh import mh_sample
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

    def test_infer_no_resample_uses_weights(self):
        from cathedral.model import infer

        @model
        def weighted_coin():
            x = flip(0.5)
            observe(Normal(1.0 if x else 0.0, 0.1), 1.0)
            return x

        posterior = infer(weighted_coin, method="importance", num_samples=500, resample=False, seed=123)
        unweighted = sum(posterior.samples) / len(posterior.samples)

        assert posterior.ess is not None
        assert posterior.probability() > 0.9
        assert abs(posterior.probability() - unweighted) > 0.2

    def test_impossible_observation(self):
        """Should raise if all samples are rejected via condition()."""

        def impossible():
            condition(False)
            return True

        with pytest.raises(RuntimeError, match="all samples were rejected"):
            importance_sample(impossible, num_samples=100)


class TestSeededInference:
    def test_rejection_seed_reproducible(self):
        def two_coins():
            return {"a": flip(0.3), "b": flip(0.8)}

        np.random.seed(1)
        traces_a = rejection_sample(two_coins, num_samples=50, seed=123)
        np.random.seed(999)
        traces_b = rejection_sample(two_coins, num_samples=50, seed=123)

        assert [t.result for t in traces_a] == [t.result for t in traces_b]

    def test_importance_seed_reproducible_with_resampling(self):
        def gaussian():
            x = sample(Normal(0, 1))
            observe(Normal(x, 0.25), 0.5)
            return x

        np.random.seed(1)
        traces_a = importance_sample(gaussian, num_samples=100, seed=123)
        np.random.seed(999)
        traces_b = importance_sample(gaussian, num_samples=100, seed=123)

        assert [t.result for t in traces_a] == [t.result for t in traces_b]

    def test_mh_seed_reproducible(self):
        def biased():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            condition(result)
            return fair

        np.random.seed(1)
        traces_a = mh_sample(biased, num_samples=100, burn_in=50, seed=123)
        np.random.seed(999)
        traces_b = mh_sample(biased, num_samples=100, burn_in=50, seed=123)

        assert [t.result for t in traces_a] == [t.result for t in traces_b]

    def test_infer_seed_ignores_global_numpy_rng_state(self):
        from cathedral.model import infer

        @model
        def coin():
            return flip(0.7)

        np.random.seed(1)
        posterior_a = infer(coin, method="rejection", num_samples=100, seed=321)
        np.random.seed(999)
        posterior_b = infer(coin, method="rejection", num_samples=100, seed=321)

        assert posterior_a.samples == posterior_b.samples


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

    def test_summary(self):
        from cathedral.model import infer

        @model
        def normal_model():
            return sample(Normal(0, 1))

        posterior = infer(normal_model, method="rejection", num_samples=1000, seed=123)
        summary = posterior.summary(level=0.8)

        assert summary["num_samples"] == posterior.num_samples
        assert summary["mean"] == posterior.mean()
        assert summary["std"] == posterior.std()
        assert summary["credible_interval"] == posterior.credible_interval(0.8)
        assert summary["level"] == 0.8
        assert summary["ess"] == posterior.ess
        assert summary["has_fixed_structure"] == posterior.has_fixed_structure


class TestMHSampling:
    def test_simple_condition(self):
        """MH should produce traces that satisfy the condition."""

        def model_fn():
            x = flip(0.5)
            condition(x)
            return x

        traces = mh_sample(model_fn, num_samples=200, burn_in=100)
        assert len(traces) == 200
        assert all(t.result is True for t in traces)

    def test_biased_coin_posterior(self):
        """MH should approximate the correct posterior for a biased coin."""
        np.random.seed(42)

        def biased():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            condition(result)
            return fair

        traces = mh_sample(biased, num_samples=3000, burn_in=500)
        fair_prob = sum(t.result for t in traces) / len(traces)
        # P(fair | heads) = P(heads|fair)*P(fair) / P(heads)
        # = 0.5*0.5 / (0.5*0.5 + 0.9*0.5) = 0.25/0.7 ≈ 0.357
        assert 0.25 < fair_prob < 0.50

    def test_sprinkler(self):
        """Classic sprinkler example should give reasonable P(rain|wet)."""
        np.random.seed(123)

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

        traces = mh_sample(sprinkler, num_samples=5000, burn_in=1000)
        rain_prob = sum(t.result["rain"] for t in traces) / len(traces)
        assert 0.25 < rain_prob < 0.65

    def test_continuous_observe(self):
        """MH should work with observe()-based soft conditioning."""
        np.random.seed(42)

        def gaussian_mean():
            mu = sample(Normal(0, 5), name="mu")
            observe(Normal(mu, 1), 3.0)
            observe(Normal(mu, 1), 3.5)
            observe(Normal(mu, 1), 2.5)
            return mu

        traces = mh_sample(gaussian_mean, num_samples=3000, burn_in=1000)
        mean_mu = np.mean([t.result for t in traces])
        assert 1.5 < mean_mu < 4.5

    def test_num_samples_correct(self):
        """Should return exactly num_samples traces."""

        def coin():
            return flip(0.5)

        traces = mh_sample(coin, num_samples=50, burn_in=10)
        assert len(traces) == 50

    def test_burn_in_default(self):
        """Default burn_in should be num_samples // 2."""

        def coin():
            return flip(0.5)

        traces = mh_sample(coin, num_samples=100)
        assert len(traces) == 100

    def test_lag(self):
        """Lag/thinning should skip intermediate samples."""

        def coin():
            return flip(0.5)

        traces = mh_sample(coin, num_samples=50, burn_in=10, lag=3)
        assert len(traces) == 50

    def test_structural_changes(self):
        """MH should handle models where control flow depends on random choices."""
        np.random.seed(42)

        def branching():
            a = flip(0.5, name="a")
            if a:
                b = sample(Normal(0, 1), name="b")
                return {"a": a, "val": b}
            else:
                c = sample(Normal(5, 1), name="c")
                return {"a": a, "val": c}

        traces = mh_sample(branching, num_samples=2000, burn_in=500)
        a_prob = sum(t.result["a"] for t in traces) / len(traces)
        assert 0.3 < a_prob < 0.7

    def test_impossible_initial_trace(self):
        """Should raise if initial trace can't be found."""

        def impossible():
            condition(False)
            return True

        with pytest.raises(RuntimeError, match="could not find a valid initial trace"):
            mh_sample(impossible, num_samples=10, max_init_attempts=100)

    def test_with_model_args(self):
        """MH should correctly pass through model arguments."""
        np.random.seed(42)

        def threshold_model(threshold):
            x = sample(Normal(0, 1), name="x")
            condition(x > threshold)
            return x

        traces = mh_sample(threshold_model, args=(-1.0,), num_samples=500, burn_in=200)
        assert all(t.result > -1.0 for t in traces)

    def test_infer_mh_method(self):
        """MH should be accessible via infer() with method='mh'."""
        np.random.seed(42)
        from cathedral.model import infer

        @model
        def coin():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            condition(result)
            return fair

        posterior = infer(coin, method="mh", num_samples=1000, burn_in=200)
        assert posterior.num_samples == 1000
        assert 0.20 < posterior.probability() < 0.55


class TestConditionParameter:
    """Tests for the condition= keyword on infer()."""

    def test_rejection_with_condition(self):
        """External condition should filter results in rejection sampling."""
        np.random.seed(42)

        @model
        def coin():
            return flip(0.5)

        from cathedral.model import infer

        posterior = infer(coin, method="rejection", num_samples=100, condition=lambda x: x)
        assert all(r is True for r in posterior.samples)

    def test_condition_on_dict_return(self):
        """External condition should work with dict-returning models."""
        np.random.seed(42)

        @model
        def lawn():
            rain = flip(0.3)
            sprinkler = flip(0.5)
            wet = rain or sprinkler
            return {"rain": rain, "sprinkler": sprinkler, "wet": wet}

        from cathedral.model import infer

        posterior = infer(lawn, num_samples=2000, condition=lambda r: r["wet"])
        assert all(r["wet"] for r in posterior.samples)
        rain_prob = posterior.probability("rain")
        assert 0.25 < rain_prob < 0.65

    def test_same_model_different_conditions(self):
        """The same model should support multiple different conditions."""
        np.random.seed(42)

        @model
        def lawn():
            rain = flip(0.3)
            sprinkler = flip(0.5)
            wet = rain or sprinkler
            return {"rain": rain, "sprinkler": sprinkler, "wet": wet}

        from cathedral.model import infer

        p1 = infer(lawn, num_samples=2000, condition=lambda r: r["wet"])
        p2 = infer(lawn, num_samples=2000, condition=lambda r: r["wet"] and r["sprinkler"])

        rain_given_wet = p1.probability("rain")
        rain_given_wet_and_sprinkler = p2.probability("rain")
        assert rain_given_wet_and_sprinkler < rain_given_wet + 0.1

    def test_condition_with_enumerate(self):
        """External condition should produce exact posteriors with enumeration."""

        @model
        def two_flips():
            a = flip(0.5)
            b = flip(0.5)
            return {"a": a, "b": b}

        from cathedral.model import infer

        posterior = infer(
            two_flips,
            method="enumerate",
            condition=lambda r: r["a"] or r["b"],
        )
        # P(a | a or b) = P(a and (a or b)) / P(a or b)
        # = P(a) / P(a or b) = 0.5 / 0.75 = 2/3
        p_a = posterior.probability("a")
        assert abs(p_a - 2 / 3) < 1e-10

    def test_condition_with_mh(self):
        """External condition should work with MH sampling."""
        np.random.seed(42)

        @model
        def coin():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            return {"fair": fair, "result": result}

        from cathedral.model import infer

        posterior = infer(
            coin,
            method="mh",
            num_samples=2000,
            burn_in=500,
            condition=lambda r: r["result"],
        )
        p_fair = posterior.probability("fair")
        assert 0.20 < p_fair < 0.55

    def test_condition_with_importance(self):
        """External condition should work with importance sampling."""
        np.random.seed(42)
        from cathedral.distributions import Normal

        @model
        def gaussian():
            x = sample(Normal(0, 1), name="x")
            return x

        from cathedral.model import infer

        posterior = infer(
            gaussian,
            method="importance",
            num_samples=5000,
            condition=lambda x: x > 0,
        )
        assert all(r > 0 for r in posterior.samples)

    def test_condition_composes_with_inline(self):
        """External condition should compose with inline condition() calls."""
        np.random.seed(42)

        @model
        def model_with_inline():
            a = flip(0.5)
            b = flip(0.5)
            condition(a or b)  # inline: at least one True
            return {"a": a, "b": b}

        from cathedral.model import infer

        posterior = infer(
            model_with_inline,
            method="enumerate",
            condition=lambda r: r["a"],  # external: a must be True
        )
        # Both conditions: a=True and (a or b) => a=True always
        # So b is unconstrained: P(b) = 0.5
        p_b = posterior.probability("b")
        assert abs(p_b - 0.5) < 1e-10

    def test_condition_none_is_noop(self):
        """condition=None should behave identically to no condition."""
        np.random.seed(42)

        @model
        def coin():
            return flip(0.7)

        from cathedral.model import infer

        p1 = infer(coin, num_samples=5000)
        np.random.seed(42)
        p2 = infer(coin, num_samples=5000, condition=None)
        assert abs(p1.probability() - p2.probability()) < 0.05


class TestEnumeration:
    def test_single_flip(self):
        """Enumerating a single flip should produce exactly 2 traces."""

        def coin():
            return flip(0.7)

        traces = enumerate_executions(coin)
        assert len(traces) == 2
        marginals = marginals_from_traces(traces)
        assert abs(marginals[True] - 0.7) < 1e-10
        assert abs(marginals[False] - 0.3) < 1e-10

    def test_two_flips(self):
        """Two independent flips should produce 4 traces."""

        def two_coins():
            a = flip(0.5)
            b = flip(0.5)
            return (a, b)

        traces = enumerate_executions(two_coins)
        assert len(traces) == 4
        marginals = marginals_from_traces(traces)
        for val in [(True, True), (True, False), (False, True), (False, False)]:
            assert abs(marginals[val] - 0.25) < 1e-10

    def test_condition_exact(self):
        """Conditioning should produce exact posterior probabilities."""

        def biased_coin():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            condition(result)
            return fair

        traces = enumerate_executions(biased_coin)
        marginals = marginals_from_traces(traces)
        # P(fair|heads) = P(heads|fair)*P(fair) / P(heads)
        # = 0.5*0.5 / (0.5*0.5 + 0.9*0.5) = 0.25/0.7
        expected = 0.25 / 0.7
        assert abs(marginals[True] - expected) < 1e-10
        assert abs(marginals[False] - (1 - expected)) < 1e-10

    def test_sprinkler_exact(self):
        """Sprinkler model should give exact P(rain|wet)."""

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

        traces = enumerate_executions(sprinkler)
        marginals = marginals_from_traces(traces)

        # Compute exact P(rain|wet) by summing over all rain=True outcomes
        rain_prob = sum(p for val, p in marginals.items() if val["rain"])
        # P(wet) = 0.3*0.9 + 0.7*(0.5*0.8 + 0.5*0.1)
        #        = 0.27 + 0.7*0.45 = 0.27 + 0.315 = 0.585
        # P(rain|wet) = 0.27 / 0.585 ≈ 0.4615
        assert abs(rain_prob - 0.27 / 0.585) < 1e-10

    def test_categorical(self):
        """Enumeration should work with Categorical distributions."""

        def die_roll():
            die = sample(Categorical(["a", "b", "c"], [0.2, 0.3, 0.5]))
            return die

        traces = enumerate_executions(die_roll)
        assert len(traces) == 3
        marginals = marginals_from_traces(traces)
        assert abs(marginals["a"] - 0.2) < 1e-10
        assert abs(marginals["b"] - 0.3) < 1e-10
        assert abs(marginals["c"] - 0.5) < 1e-10

    def test_uniform_draw(self):
        """Enumeration should work with UniformDraw."""

        def pick():
            return sample(UniformDraw([10, 20, 30]))

        traces = enumerate_executions(pick)
        assert len(traces) == 3
        marginals = marginals_from_traces(traces)
        for val in [10, 20, 30]:
            assert abs(marginals[val] - 1 / 3) < 1e-10

    def test_structural_branching(self):
        """Enumeration should handle control-flow branching correctly."""

        def branching():
            a = flip(0.5)
            if a:
                b = flip(0.3)
                return ("a", b)
            else:
                c = flip(0.7)
                return ("not_a", c)

        traces = enumerate_executions(branching)
        assert len(traces) == 4
        marginals = marginals_from_traces(traces)
        assert abs(marginals[("a", True)] - 0.5 * 0.3) < 1e-10
        assert abs(marginals[("a", False)] - 0.5 * 0.7) < 1e-10
        assert abs(marginals[("not_a", True)] - 0.5 * 0.7) < 1e-10
        assert abs(marginals[("not_a", False)] - 0.5 * 0.3) < 1e-10

    def test_max_executions(self):
        """max_executions should limit the number of complete paths explored."""

        def many_flips():
            return (flip(0.5), flip(0.5), flip(0.5), flip(0.5))

        traces = enumerate_executions(many_flips, max_executions=4)
        assert len(traces) == 4

    def test_strategies_all_complete(self):
        """All strategies should produce the same marginals for exhaustive enumeration."""

        def coin():
            return flip(0.6)

        for strategy in ["depth_first", "breadth_first", "likely_first"]:
            traces = enumerate_executions(coin, strategy=strategy)
            marginals = marginals_from_traces(traces)
            assert abs(marginals[True] - 0.6) < 1e-10

    def test_all_paths_rejected(self):
        """Should raise if every path is rejected."""

        def impossible():
            x = flip(0.5)
            condition(False)
            return x

        with pytest.raises(RuntimeError, match="all execution paths were rejected"):
            enumerate_executions(impossible)

    def test_continuous_distribution_raises(self):
        """Enumeration should raise a clear error for continuous distributions."""

        def continuous_model():
            return sample(Normal(0, 1))

        with pytest.raises(RuntimeError, match="no finite support"):
            enumerate_executions(continuous_model)

    def test_infer_enumerate_method(self):
        """Enumeration should be accessible via infer(method='enumerate')."""
        from cathedral.model import infer

        @model
        def coin():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            condition(result)
            return fair

        posterior = infer(coin, method="enumerate")
        # With exact enumeration, P(fair|heads) = 5/14
        expected = 0.25 / 0.7
        assert abs(posterior.probability() - expected) < 1e-10

    def test_enumerate_vs_rejection_agreement(self):
        """Enumeration and rejection sampling should agree on a simple model."""
        np.random.seed(42)

        def model_fn():
            a = flip(0.4)
            b = flip(0.6)
            condition(a or b)
            return a

        enum_traces = enumerate_executions(model_fn)
        enum_marginals = marginals_from_traces(enum_traces)
        exact_p_a = enum_marginals[True]

        rej_traces = rejection_sample(model_fn, num_samples=20000)
        rej_p_a = sum(t.result for t in rej_traces) / len(rej_traces)

        assert abs(exact_p_a - rej_p_a) < 0.02
