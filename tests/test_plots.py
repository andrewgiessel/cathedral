"""Tests for cathedral.plots: matplotlib-based diagnostic plots."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure

from cathedral import model
from cathedral.distributions import Normal
from cathedral.model import infer
from cathedral.plots import plot_ess, plot_posterior, plot_trace_values, plot_weights
from cathedral.primitives import condition, flip, observe, sample


class TestPlotPosterior:
    def test_returns_figure(self):
        @model
        def coin():
            return flip(0.5)

        p = infer(coin, method="rejection", num_samples=100)
        fig = plot_posterior(p)
        assert isinstance(fig, Figure)

    def test_boolean_values_bar_chart(self):
        @model
        def coin():
            return flip(0.5)

        p = infer(coin, method="rejection", num_samples=100)
        fig = plot_posterior(p)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Probability"

    def test_numeric_values_histogram(self):
        np.random.seed(42)

        @model
        def normal():
            return sample(Normal(0, 1))

        p = infer(normal, method="rejection", num_samples=200)
        fig = plot_posterior(p)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Density"

    def test_kde_mode(self):
        np.random.seed(42)

        @model
        def normal():
            return sample(Normal(0, 1))

        p = infer(normal, method="rejection", num_samples=200)
        fig = plot_posterior(p, kind="kde")
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Density"

    def test_with_key(self):
        np.random.seed(42)

        @model
        def two():
            return {"a": flip(0.5), "b": sample(Normal(0, 1))}

        p = infer(two, method="rejection", num_samples=100)
        fig = plot_posterior(p, key="a")
        ax = fig.axes[0]
        assert "[a]" in ax.get_title()

    def test_title_includes_method(self):
        @model
        def coin():
            return flip(0.5)

        p = infer(coin, method="rejection", num_samples=50)
        fig = plot_posterior(p)
        ax = fig.axes[0]
        assert "rejection" in ax.get_title()

    def test_custom_ax(self):
        import matplotlib.pyplot as plt

        @model
        def coin():
            return flip(0.5)

        p = infer(coin, method="rejection", num_samples=50)
        fig, ax = plt.subplots()
        result = plot_posterior(p, ax=ax)
        assert result is fig


class TestPlotWeights:
    def test_returns_figure(self):
        np.random.seed(42)

        @model
        def obs():
            mu = sample(Normal(0, 5))
            observe(Normal(mu, 1), 3.0)
            return mu

        p = infer(obs, method="importance", num_samples=200)
        fig = plot_weights(p)
        assert isinstance(fig, Figure)

    def test_title_shows_ess(self):
        np.random.seed(42)

        @model
        def obs():
            mu = sample(Normal(0, 5))
            observe(Normal(mu, 1), 3.0)
            return mu

        p = infer(obs, method="importance", num_samples=200)
        fig = plot_weights(p)
        ax = fig.axes[0]
        assert "ESS=" in ax.get_title()

    def test_raises_without_log_weights(self):
        @model
        def coin():
            return flip(0.5)

        p = infer(coin, method="rejection", num_samples=50)
        with pytest.raises(ValueError, match="log_weights"):
            plot_weights(p)


class TestPlotTraceValues:
    def test_returns_figure(self):
        np.random.seed(42)

        @model
        def coin():
            x = flip(0.5, name="x")
            condition(x)
            return x

        p = infer(coin, method="mh", num_samples=100, burn_in=50)
        fig = plot_trace_values(p, "x")
        assert isinstance(fig, Figure)

    def test_title_includes_address(self):
        np.random.seed(42)

        @model
        def coin():
            x = flip(0.5, name="x")
            condition(x)
            return x

        p = infer(coin, method="mh", num_samples=100, burn_in=50)
        fig = plot_trace_values(p, "x")
        ax = fig.axes[0]
        assert "x" in ax.get_title()

    def test_missing_address_uses_nan(self):
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
        fig = plot_trace_values(p, "b")
        assert isinstance(fig, Figure)


class TestPlotEss:
    def test_returns_figure(self):
        np.random.seed(42)

        @model
        def two():
            return {"a": flip(0.5, name="a"), "b": flip(0.3, name="b")}

        p = infer(two, method="rejection", num_samples=200)
        fig = plot_ess(p)
        assert isinstance(fig, Figure)

    def test_raises_variable_structure(self):
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
            plot_ess(p)

    def test_title_shows_n(self):
        np.random.seed(42)

        @model
        def coin():
            flip(0.5, name="x")
            return True

        p = infer(coin, method="rejection", num_samples=100)
        fig = plot_ess(p)
        ax = fig.axes[0]
        assert "n=100" in ax.get_title()
