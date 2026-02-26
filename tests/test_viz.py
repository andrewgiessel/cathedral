"""Tests for cathedral.viz: trace formatting, posterior analysis, DOT output."""

import numpy as np
import pytest

from cathedral import model
from cathedral.distributions import Normal
from cathedral.model import Posterior, infer
from cathedral.primitives import condition, flip, sample
from cathedral.trace import Choice, Trace, run_with_trace
from cathedral.viz import (
    address_frequency,
    compare_traces,
    format_trace,
    structure_summary,
    trace_to_dot,
)


class TestFormatTrace:
    def test_basic_format(self):
        def coin():
            return flip(0.5)

        trace = run_with_trace(coin)
        result = format_trace(trace)
        assert "Trace" in result
        assert "log_joint" in result
        assert "1 choices" in result

    def test_multiple_choices(self):
        def two_coins():
            a = flip(0.5, name="a")
            b = flip(0.5, name="b")
            return (a, b)

        trace = run_with_trace(two_coins)
        result = format_trace(trace)
        assert "2 choices" in result
        assert "a" in result
        assert "b" in result

    def test_show_dist(self):
        def coin():
            return flip(0.5, name="x")

        trace = run_with_trace(coin)
        result = format_trace(trace, show_dist=True)
        assert "Bernoulli" in result

    def test_hide_log_prob(self):
        def coin():
            return flip(0.5, name="x")

        trace = run_with_trace(coin)
        result_with = format_trace(trace, show_log_prob=True)
        result_without = format_trace(trace, show_log_prob=False)
        assert "(" in result_with
        assert "0." not in result_without or "log_joint" in result_without

    def test_scope_grouping(self):
        def nested():
            a = flip(0.5, name="a")
            b = flip(0.5, name="b")
            return (a, b)

        trace = run_with_trace(nested, capture_scopes=True)
        result = format_trace(trace)
        assert "a" in result
        assert "b" in result


class TestStructureSummary:
    def test_fixed_structure(self):
        @model
        def coin():
            flip(0.5, name="x")
            return True

        p = infer(coin, method="rejection", num_samples=50)
        summary = structure_summary(p)
        assert "Distinct structures: 1" in summary
        assert "Total unique addresses: 1" in summary

    def test_variable_structure(self):
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
        summary = structure_summary(p)
        assert "Distinct structures: 2" in summary
        assert "Variable" in summary

    def test_always_present_addresses(self):
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
        summary = structure_summary(p)
        assert "Always present" in summary
        assert "a" in summary


class TestAddressFrequency:
    def test_fixed_structure_all_1(self):
        @model
        def coin():
            flip(0.5, name="x")
            flip(0.3, name="y")
            return True

        p = infer(coin, method="rejection", num_samples=50)
        freq = address_frequency(p)
        assert freq["x"] == 1.0
        assert freq["y"] == 1.0

    def test_variable_structure(self):
        np.random.seed(42)

        @model
        def branching():
            a = flip(0.5, name="a")
            if a:
                sample(Normal(0, 1), name="b")
            else:
                sample(Normal(5, 1), name="c")
            return a

        p = infer(branching, method="mh", num_samples=500, burn_in=200)
        freq = address_frequency(p)
        assert freq["a"] == 1.0
        assert 0.3 < freq["b"] < 0.7
        assert 0.3 < freq["c"] < 0.7
        assert abs(freq["b"] + freq["c"] - 1.0) < 0.01

    def test_empty_posterior_returns_empty(self):
        from cathedral.trace import Trace

        t = Trace(result=True, choices={}, log_score=0.0)
        p = Posterior([t])
        freq = address_frequency(p)
        assert freq == {}


class TestCompareTraces:
    def test_same_structure(self):
        np.random.seed(42)

        def coin():
            flip(0.5, name="x")
            flip(0.3, name="y")
            return True

        t1 = run_with_trace(coin)
        t2 = run_with_trace(coin)
        result = compare_traces(t1, t2)
        assert "Trace Comparison" in result
        assert "Shared" in result

    def test_different_structure(self):
        np.random.seed(42)

        def model_a():
            flip(0.5, name="a")
            flip(0.5, name="b")
            return True

        def model_b():
            flip(0.5, name="b")
            flip(0.5, name="c")
            return True

        t1 = run_with_trace(model_a)
        t2 = run_with_trace(model_b)
        result = compare_traces(t1, t2)
        assert "Appeared in B" in result
        assert "Disappeared from A" in result
        assert "+ c" in result
        assert "- a" in result

    def test_same_structure_counts_changes(self):
        from cathedral.distributions import Bernoulli
        from cathedral.trace import Choice, Trace

        c1 = Choice("x", Bernoulli(0.5), True, np.log(0.5))
        c2 = Choice("x", Bernoulli(0.5), False, np.log(0.5))
        t1 = Trace(result=True, choices={"x": c1}, log_score=0.0)
        t2 = Trace(result=False, choices={"x": c2}, log_score=0.0)

        result = compare_traces(t1, t2)
        assert "Same structure, 1/1 values differ" in result


class TestTraceToDot:
    def test_produces_valid_dot(self):
        def coin():
            flip(0.5, name="x")
            return True

        trace = run_with_trace(coin)
        dot = trace_to_dot(trace)
        assert dot.startswith("digraph trace {")
        assert dot.endswith("}")
        assert "n_x" in dot

    def test_custom_label(self):
        def coin():
            flip(0.5, name="x")
            return True

        trace = run_with_trace(coin)
        dot = trace_to_dot(trace, label="My Trace")
        assert "My Trace" in dot

    def test_show_log_prob(self):
        def coin():
            flip(0.5, name="x")
            return True

        trace = run_with_trace(coin)
        dot_with = trace_to_dot(trace, show_log_prob=True)
        dot_without = trace_to_dot(trace, show_log_prob=False)
        assert "lp=" in dot_with
        assert "lp=" not in dot_without

    def test_show_dist(self):
        def coin():
            flip(0.5, name="x")
            return True

        trace = run_with_trace(coin)
        dot = trace_to_dot(trace, show_dist=True)
        assert "Bernoulli" in dot

    def test_scope_creates_subgraphs(self):
        def nested():
            flip(0.5, name="a")
            flip(0.5, name="b")
            return True

        trace = run_with_trace(nested, capture_scopes=True)
        dot = trace_to_dot(trace)
        assert "n_a" in dot
        assert "n_b" in dot

    def test_multiple_choices_have_edges(self):
        def two():
            flip(0.5, name="a")
            flip(0.5, name="b")
            return True

        trace = run_with_trace(two)
        dot = trace_to_dot(trace)
        assert "n_a -> n_b" in dot

    def test_single_choice_no_edges(self):
        def one():
            return flip(0.5, name="x")

        trace = run_with_trace(one)
        dot = trace_to_dot(trace)
        assert "->" not in dot
