"""Tests for Posterior.to_arviz() ArviZ bridge."""

import numpy as np
import pytest

from cathedral import model
from cathedral.distributions import Normal
from cathedral.model import infer
from cathedral.primitives import condition, flip, sample


class TestToArviz:
    def test_basic_conversion(self):
        pytest.importorskip("arviz")

        @model
        def coin():
            flip(0.5, name="x")
            return True

        p = infer(coin, method="rejection", num_samples=100)
        idata = p.to_arviz()
        assert hasattr(idata, "posterior")
        assert "x" in idata.posterior.data_vars

    def test_shape(self):
        pytest.importorskip("arviz")

        @model
        def coin():
            flip(0.5, name="x")
            flip(0.3, name="y")
            return True

        p = infer(coin, method="rejection", num_samples=100)
        idata = p.to_arviz()
        assert idata.posterior["x"].shape == (1, 100)
        assert idata.posterior["y"].shape == (1, 100)

    def test_numeric_values(self):
        pytest.importorskip("arviz")
        np.random.seed(42)

        @model
        def normal():
            sample(Normal(0, 1), name="z")
            return True

        p = infer(normal, method="rejection", num_samples=50)
        idata = p.to_arviz()
        values = idata.posterior["z"].values.flatten()
        assert len(values) == 50
        assert all(isinstance(v, float | np.floating) for v in values)

    def test_boolean_converted_to_float(self):
        pytest.importorskip("arviz")

        @model
        def coin():
            flip(0.5, name="x")
            return True

        p = infer(coin, method="rejection", num_samples=50)
        idata = p.to_arviz()
        values = idata.posterior["x"].values.flatten()
        assert all(v in (0.0, 1.0) for v in values)

    def test_variable_structure_raises(self):
        pytest.importorskip("arviz")
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
        with pytest.raises(ValueError, match="fixed-structure"):
            p.to_arviz()

    def test_enumerate_conversion(self):
        pytest.importorskip("arviz")

        @model
        def coin():
            fair = flip(0.5, name="fair")
            result = flip(0.5 if fair else 0.9, name="result")
            condition(result)
            return fair

        p = infer(coin, method="enumerate")
        idata = p.to_arviz()
        assert "fair" in idata.posterior.data_vars
        assert "result" in idata.posterior.data_vars
