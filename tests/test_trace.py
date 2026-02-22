"""Tests for Cathedral trace infrastructure."""

import pytest

from cathedral.distributions import Bernoulli, Normal
from cathedral.trace import Choice, Rejected, Trace, TraceContext, get_trace_context, run_with_trace


class TestChoice:
    def test_choice_fields(self):
        d = Bernoulli(0.5)
        c = Choice(address="flip", distribution=d, value=True, log_prob=-0.693)
        assert c.address == "flip"
        assert c.value is True


class TestTrace:
    def test_empty_trace(self):
        t = Trace()
        assert len(t) == 0
        assert t.log_joint == 0.0
        assert t.addresses == []

    def test_log_joint_with_choices(self):
        t = Trace()
        d = Bernoulli(0.5)
        t.choices["a"] = Choice("a", d, True, -0.693)
        t.choices["b"] = Choice("b", d, False, -0.693)
        assert abs(t.log_joint - (-0.693 * 2)) < 1e-6

    def test_log_joint_includes_score(self):
        t = Trace()
        t.log_score = -1.5
        d = Bernoulli(0.5)
        t.choices["a"] = Choice("a", d, True, -0.693)
        expected = -0.693 + (-1.5)
        assert abs(t.log_joint - expected) < 1e-6


class TestTraceContext:
    def test_fresh_address_unique(self):
        ctx = TraceContext()
        a1 = ctx.fresh_address("flip")
        a2 = ctx.fresh_address("flip")
        a3 = ctx.fresh_address("flip")
        assert a1 != a2 != a3
        assert a1 == "flip"
        assert a2 == "flip__1"
        assert a3 == "flip__2"

    def test_fresh_address_different_prefixes(self):
        ctx = TraceContext()
        a1 = ctx.fresh_address("Bernoulli")
        a2 = ctx.fresh_address("Normal")
        assert a1 == "Bernoulli"
        assert a2 == "Normal"

    def test_record_choice(self):
        ctx = TraceContext()
        d = Normal(0, 1)
        ctx.record_choice("x", d, 1.5)
        assert "x" in ctx.trace.choices
        assert ctx.trace.choices["x"].value == 1.5

    def test_interventions(self):
        ctx = TraceContext(interventions={"x": 42})
        assert ctx.has_intervention("x")
        assert ctx.get_intervention("x") == 42
        assert not ctx.has_intervention("y")


class TestRunWithTrace:
    def test_basic_tracing(self):
        from cathedral.primitives import flip

        def model():
            return flip(0.5)

        trace = run_with_trace(model)
        assert trace.result in (True, False)
        assert len(trace) == 1

    def test_tracing_records_multiple_choices(self):
        from cathedral.primitives import flip

        def model():
            a = flip(0.3)
            b = flip(0.7)
            return a and b

        trace = run_with_trace(model)
        assert len(trace) == 2

    def test_rejected_propagates(self):
        from cathedral.primitives import condition, flip

        def model():
            condition(False)
            return True

        with pytest.raises(Rejected):
            run_with_trace(model)

    def test_interventions_override_sampling(self):
        from cathedral.primitives import sample

        def model():
            return sample(Normal(0, 1), name="x")

        trace = run_with_trace(model, interventions={"x": 99.0})
        assert trace.result == 99.0
        assert trace.choices["x"].value == 99.0

    def test_context_is_none_after_run(self):
        def model():
            assert get_trace_context() is not None
            return 1

        run_with_trace(model)
        assert get_trace_context() is None

    def test_context_cleaned_up_on_exception(self):
        def model():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            run_with_trace(model)
        assert get_trace_context() is None

    def test_args_passed_to_model(self):
        def model(x, y):
            return x + y

        trace = run_with_trace(model, args=(3, 4))
        assert trace.result == 7
