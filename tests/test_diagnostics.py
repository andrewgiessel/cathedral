"""Tests for inference diagnostics: InferenceInfo, Posterior diagnostic properties."""

import pickle

import numpy as np
import pytest

from cathedral import model
from cathedral.distributions import Normal
from cathedral.model import Posterior, infer
from cathedral.primitives import condition, flip, observe, sample


class TestInferenceInfo:
    def test_rejection_info(self):
        @model
        def coin():
            x = flip(0.5)
            condition(x)
            return x

        p = infer(coin, method="rejection", num_samples=200)
        assert p.info is not None
        assert p.info.method == "rejection"
        assert p.info.num_samples == 200
        assert p.info.num_attempts is not None
        assert p.info.num_attempts >= 200
        assert p.info.acceptance_rate is not None
        assert 0 < p.info.acceptance_rate <= 1.0

    def test_importance_info(self):
        np.random.seed(42)

        @model
        def obs():
            mu = sample(Normal(0, 5))
            observe(Normal(mu, 1), 3.0)
            return mu

        p = infer(obs, method="importance", num_samples=500)
        assert p.info is not None
        assert p.info.method == "importance"
        assert p.info.log_weights is not None
        assert len(p.info.log_weights) > 0
        assert p.info.log_marginal_likelihood is not None
        assert p.info.ess is not None
        assert p.info.ess > 0

    def test_mh_info(self):
        np.random.seed(42)

        @model
        def coin():
            x = flip(0.5)
            condition(x)
            return x

        p = infer(coin, method="mh", num_samples=100, burn_in=50)
        assert p.info is not None
        assert p.info.method == "mh"
        assert p.info.acceptance_rate is not None
        assert 0 <= p.info.acceptance_rate <= 1.0
        assert p.info.extra["total_steps"] == 150
        assert p.info.extra["burn_in"] == 50
        assert p.info.extra["lag"] == 1

    def test_enumerate_info(self):
        @model
        def coin():
            return flip(0.6)

        p = infer(coin, method="enumerate")
        assert p.info is not None
        assert p.info.method == "enumerate"
        assert p.info.log_marginal_likelihood is not None
        assert abs(p.info.log_marginal_likelihood - 0.0) < 1e-10
        assert p.info.extra["num_paths"] == 2
        assert p.info.extra["exhaustive"] is True

    def test_enumerate_conditioned_log_ml(self):
        @model
        def biased():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            condition(result)
            return fair

        p = infer(biased, method="enumerate")
        assert p.info is not None
        expected_ml = np.log(0.5 * 0.5 + 0.5 * 0.9)
        assert abs(p.info.log_marginal_likelihood - expected_ml) < 1e-10


class TestPosteriorDiagnostics:
    def test_has_fixed_structure_true(self):
        @model
        def coin():
            return flip(0.5)

        p = infer(coin, method="rejection", num_samples=50)
        assert p.has_fixed_structure is True

    def test_has_fixed_structure_false(self):
        np.random.seed(42)

        @model
        def branching():
            a = flip(0.5, name="a")
            if a:
                sample(Normal(0, 1), name="b")
            else:
                sample(Normal(5, 1), name="c")
            return a

        p = infer(branching, method="mh", num_samples=200, burn_in=100)
        assert p.has_fixed_structure is False

    def test_acceptance_rate_property(self):
        @model
        def coin():
            x = flip(0.5)
            condition(x)
            return x

        p = infer(coin, method="rejection", num_samples=100)
        assert p.acceptance_rate is not None
        assert 0 < p.acceptance_rate <= 1.0

    def test_acceptance_rate_none_without_info(self):
        from cathedral.trace import Trace

        t = Trace(result=True, choices={}, log_score=0.0)
        p = Posterior([t])
        assert p.acceptance_rate is None

    def test_ess_from_info(self):
        np.random.seed(42)

        @model
        def obs():
            mu = sample(Normal(0, 5))
            observe(Normal(mu, 1), 3.0)
            return mu

        p = infer(obs, method="importance", num_samples=500)
        assert p.ess is not None
        assert p.ess > 0

    def test_ess_from_weights(self):
        from cathedral.trace import Trace

        t1 = Trace(result=1, choices={}, log_score=0.0)
        t2 = Trace(result=2, choices={}, log_score=0.0)
        weights = np.array([0.5, 0.5])
        p = Posterior([t1, t2], weights=weights)
        assert p.ess is not None
        assert abs(p.ess - 2.0) < 1e-10

    def test_ess_none_unweighted(self):
        from cathedral.trace import Trace

        t = Trace(result=True, choices={}, log_score=0.0)
        p = Posterior([t])
        assert p.ess is None

    def test_log_marginal_likelihood_property(self):
        @model
        def coin():
            return flip(0.6)

        p = infer(coin, method="enumerate")
        assert p.log_marginal_likelihood is not None
        assert abs(p.log_marginal_likelihood - 0.0) < 1e-10

    def test_log_marginal_likelihood_none(self):
        from cathedral.trace import Trace

        t = Trace(result=True, choices={}, log_score=0.0)
        p = Posterior([t])
        assert p.log_marginal_likelihood is None

    def test_diagnostics_string(self):
        @model
        def coin():
            x = flip(0.5)
            condition(x)
            return x

        p = infer(coin, method="rejection", num_samples=100)
        diag = p.diagnostics()
        assert "Posterior: 100 samples" in diag
        assert "method: rejection" in diag
        assert "acceptance rate:" in diag
        assert "fixed structure:" in diag

    def test_diagnostics_enumerate_shows_ess_from_weights(self):
        @model
        def coin():
            return flip(0.6)

        p = infer(coin, method="enumerate")
        diag = p.diagnostics()
        assert "ESS (from weights):" in diag

    def test_diagnostics_mh_shows_extra(self):
        np.random.seed(42)

        @model
        def coin():
            x = flip(0.5)
            condition(x)
            return x

        p = infer(coin, method="mh", num_samples=100, burn_in=50)
        diag = p.diagnostics()
        assert "total_steps:" in diag
        assert "burn_in:" in diag


class TestPosteriorPersistence:
    def test_save_load_round_trip(self, tmp_path):
        @model
        def coin():
            x = flip(0.7, name="x")
            return {"x": x}

        posterior = infer(coin, method="rejection", num_samples=25, seed=123)
        path = tmp_path / "posterior.pkl"

        posterior.save(path)
        loaded = Posterior.load(path)

        assert loaded.samples == posterior.samples
        assert loaded.num_samples == posterior.num_samples
        assert loaded.info is not None
        assert loaded.info.method == "rejection"
        assert loaded.traces[0].choices["x"].value == posterior.traces[0].choices["x"].value

    def test_save_load_preserves_weighted_variable_structure_posterior(self, tmp_path):
        @model
        def branching():
            a = flip(0.5, name="a")
            if a:
                sample(Normal(0, 1), name="b")
            else:
                sample(Normal(5, 1), name="c")
            observe(Normal(1.0 if a else 0.0, 0.1), 1.0)
            return a

        posterior = infer(branching, method="importance", num_samples=50, resample=False, seed=123)
        path = tmp_path / "weighted-variable.pkl"

        posterior.save(path)
        loaded = Posterior.load(path)

        assert loaded.has_fixed_structure is False
        assert loaded.ess == posterior.ess
        assert loaded.probability() == posterior.probability()
        assert loaded.info is not None
        assert loaded.info.log_weights is not None
        assert np.array_equal(loaded.info.log_weights, posterior.info.log_weights)

    def test_load_rejects_non_posterior_pickle(self, tmp_path):
        path = tmp_path / "not-a-posterior.pkl"
        with path.open("wb") as f:
            pickle.dump({"not": "a posterior"}, f)

        with pytest.raises(TypeError, match="Expected a Posterior pickle"):
            Posterior.load(path)


class TestPosteriorExtend:
    def test_extend_rejection_appends_samples(self):
        @model
        def coin():
            return flip(0.7)

        posterior = infer(coin, method="rejection", num_samples=20, seed=123)
        extended = posterior.extend(coin, num_samples=10, seed=456)

        assert extended.num_samples == 30
        assert extended.samples[:20] == posterior.samples
        assert extended.info is not None
        assert extended.info.method == "rejection"
        assert extended.info.num_samples == 30
        assert extended.info.num_attempts is not None

    def test_extend_weighted_importance_renormalizes_weights(self):
        @model
        def weighted_coin():
            x = flip(0.5)
            observe(Normal(1.0 if x else 0.0, 0.1), 1.0)
            return x

        posterior = infer(weighted_coin, method="importance", num_samples=20, resample=False, seed=123)
        extended = posterior.extend(weighted_coin, num_samples=30, seed=456)

        assert extended.num_samples == 50
        assert extended.samples[:20] == posterior.samples
        assert extended.info is not None
        assert extended.info.log_weights is not None
        assert len(extended.info.log_weights) == 50
        assert extended.ess is not None
        assert extended.probability() > 0.9

    def test_extend_mh_continues_and_appends_samples(self):
        @model
        def coin():
            fair = flip(0.5)
            result = flip(0.5 if fair else 0.9)
            condition(result)
            return fair

        posterior = infer(coin, method="mh", num_samples=20, burn_in=5, seed=123)
        extended = posterior.extend(coin, num_samples=10, seed=456)

        assert extended.num_samples == 30
        assert extended.samples[:20] == posterior.samples
        assert extended.info is not None
        assert extended.info.method == "mh"
        assert extended.info.extra["total_steps"] == 35

    def test_extend_enumerate_recomputes_without_duplicate_paths(self):
        @model
        def coin():
            return flip(0.6)

        posterior = infer(coin, method="enumerate")
        extended = posterior.extend(coin)

        assert extended.num_samples == 2
        assert extended.num_samples == posterior.num_samples
        assert abs(extended.probability() - 0.6) < 1e-10

    def test_extend_requires_inference_metadata(self):
        from cathedral.trace import Trace

        posterior = Posterior([Trace(result=True)])

        @model
        def coin():
            return flip(0.5)

        with pytest.raises(ValueError, match="requires inference metadata"):
            posterior.extend(coin, num_samples=10)
