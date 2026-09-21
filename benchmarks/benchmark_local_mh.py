"""Reproducible comparison of legacy and adaptive local MH.

The default run preserves the original fixed budget: three independent
replicates, each with four sequential chains, 500 warmup transitions, and
1,000 retained draws per chain.  The release budget is explicitly
``--warmup 2000 --draws 8000 --replicates 3``. Timings include initialization,
validation, and warmup. No samples or traces are printed.

Usage:
    .venv/bin/python benchmarks/benchmark_local_mh.py
    .venv/bin/python benchmarks/benchmark_local_mh.py --warmup 2000 --draws 8000 --replicates 3
    .venv/bin/python benchmarks/benchmark_local_mh.py --case mixed_discrete_continuous
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad

from cathedral import Binomial, Normal, infer, model, observe, sample
from cathedral.chains import diagnose_chains

REPLICATES = 3
CHAINS = 4
WARMUP = 500
DRAWS = 1_000
BASE_SEED = 20250308
RHAT_FAILURE = 1.01


@dataclass(frozen=True)
class Reference:
    mean: float
    variance: float


@dataclass(frozen=True)
class Query:
    name: str
    estimand: Callable[[Any], float]
    reference: Reference


@dataclass(frozen=True)
class Case:
    name: str
    model_fn: Callable[..., Any]
    args: tuple[Any, ...]
    queries: tuple[Query, ...]
    block: tuple[str, ...]


@dataclass(frozen=True)
class Method:
    name: str
    infer_kwargs: dict[str, Any]


@dataclass(frozen=True)
class Run:
    wall_s: float
    invocations: int
    errors: tuple[tuple[float, float], ...]
    rhat: float
    ess_per_second: float
    ess_per_invocation: float
    mcse: float
    diagnostic_status: str


@model
def concentrated_normal() -> float:
    x = sample(Normal(0.0, 1.0), name="x")
    observe(Normal(x, 0.1), 0.75)
    return x


@model
def correlated_gaussian() -> float:
    x = sample(Normal(0.0, 1.0), name="x")
    y = sample(Normal(0.0, 1.0), name="y")
    observe(Normal(x + y, 0.1), 1.0)
    return x


@model
def mixed_discrete_continuous() -> dict[str, float | int]:
    """Analytic mixture fixture: k~Bernoulli(.5), x|k~N(k,1), y=0.7."""
    k = sample(Binomial(1, 0.5), name="k")
    x = sample(Normal(k, 1.0), name="x")
    observe(Normal(x, 0.5), 0.7)
    return {"k": k, "x": x}


def _linear_regression_moments(xs: np.ndarray, ys: np.ndarray, bounds: tuple[float, float]) -> Reference:
    """Numerically integrate the example's noise scale for slope moments."""
    design = np.column_stack((xs, np.ones_like(xs)))
    prior_precision = np.eye(2) / 25.0

    def log_density(log_noise: float) -> float:
        noise = math.exp(log_noise)
        # Determinant/Woodbury identities avoid fragile n-by-n covariances.
        small = np.eye(2) + 25.0 * (design.T @ design) / noise**2
        _, small_log_determinant = np.linalg.slogdet(small)
        log_determinant = len(xs) * math.log(noise**2) + small_log_determinant
        projected = design.T @ ys
        quadratic = float(ys @ ys / noise**2 - 25.0 * projected @ np.linalg.solve(small, projected) / noise**4)
        log_likelihood = -0.5 * (len(xs) * math.log(2.0 * math.pi) + log_determinant + quadratic)
        log_half_normal = 0.5 * math.log(2.0 / math.pi) - math.log(2.0) - noise**2 / 8.0
        return log_likelihood + log_half_normal + log_noise

    offset = log_density(math.log(0.5))

    def moments(log_noise: float) -> tuple[float, float, float]:
        noise = math.exp(log_noise)
        precision = prior_precision + design.T @ design / noise**2
        covariance = np.linalg.inv(precision)
        mean = covariance @ design.T @ ys / noise**2
        weight = math.exp(log_density(log_noise) - offset)
        return weight, weight * float(mean[0]), weight * float(covariance[0, 0] + mean[0] ** 2)

    normalizer = quad(lambda log_noise: moments(log_noise)[0], *bounds, epsabs=1e-10)[0]
    mean = quad(lambda log_noise: moments(log_noise)[1], *bounds, epsabs=1e-10)[0] / normalizer
    second_moment = quad(lambda log_noise: moments(log_noise)[2], *bounds, epsabs=1e-10)[0] / normalizer
    return Reference(mean=mean, variance=second_moment - mean**2)


def _linear_regression_reference(xs: np.ndarray, ys: np.ndarray) -> Reference:
    """Verify integration stability across nominal and wider log-noise bounds."""
    primary = _linear_regression_moments(xs, ys, (-12.0, 5.0))
    alternate = _linear_regression_moments(xs, ys, (-16.0, 7.0))
    if not (
        math.isclose(primary.mean, alternate.mean, rel_tol=1e-7, abs_tol=1e-9)
        and math.isclose(primary.variance, alternate.variance, rel_tol=1e-7, abs_tol=1e-9)
    ):
        raise RuntimeError("linear regression reference changed under wider quadrature bounds")
    return alternate


def _linear_regression_case() -> Case:
    """Build the actual example's fixed data and call its model unchanged."""
    example_path = Path(__file__).parents[1] / "examples" / "linear_regression.py"
    namespace: dict[str, Any] = {"__name__": "benchmark_linear_regression"}
    exec(compile(example_path.read_text(), example_path, "exec"), namespace)  # noqa: S102
    line_model = namespace["line_model"]
    rng = np.random.RandomState(42)
    xs = np.linspace(0, 5, 20)
    ys = 2.0 * xs + 1.0 + rng.normal(0, 0.5, len(xs))
    return Case(
        name="linear_regression_example",
        model_fn=line_model,
        args=(xs, ys),
        queries=(Query("slope", lambda result: float(result["slope"]), _linear_regression_reference(xs, ys)),),
        block=("slope", "intercept", "noise"),
    )


def _normal_reference(prior_variance: float, likelihood_variance: float, observation: float) -> Reference:
    precision = 1.0 / prior_variance + 1.0 / likelihood_variance
    variance = 1.0 / precision
    return Reference(mean=variance * observation / likelihood_variance, variance=variance)


def _mixed_reference(query: str) -> Reference:
    """Exact moments after summing the two k values independently of kernels."""
    probability_k_one = 1.0 / (1.0 + math.exp(-0.16))
    if query == "k":
        return Reference(probability_k_one, probability_k_one * (1.0 - probability_k_one))
    # x | k,y has precision 1 + 1/0.25 = 5 and mean
    # (k + y/0.25)/5 = 0.56 + 0.2*k. Include between-component variance.
    mean = 0.56 + 0.2 * probability_k_one
    return Reference(mean, 0.2 + 0.04 * probability_k_one * (1.0 - probability_k_one))


CASES = (
    Case(
        name="concentrated_normal",
        model_fn=concentrated_normal,
        args=(),
        queries=(Query("x", float, _normal_reference(1.0, 0.01, 0.75)),),
        block=("x",),
    ),
    Case(
        name="correlated_gaussian",
        model_fn=correlated_gaussian,
        args=(),
        queries=(Query("x", float, Reference(mean=1.0 / 2.01, variance=1.0 - 1.0 / 2.01)),),
        block=("x", "y"),
    ),
    Case(
        name="mixed_discrete_continuous",
        model_fn=mixed_discrete_continuous,
        args=(),
        queries=(
            Query("x", lambda result: float(result["x"]), _mixed_reference("x")),
            Query("k_prob", lambda result: float(result["k"]), _mixed_reference("k")),
        ),
        # The mixed discrete/continuous pair is intentionally scalar-only:
        # adaptive blocks support continuous sites only.
        block=(),
    ),
    _linear_regression_case(),
)


def _methods(case: Case) -> tuple[Method, ...]:
    adaptive_common = {"warmup": WARMUP, "prior_resimulation": False}
    methods = [
        Method("legacy_mh", {"burn_in": WARMUP}),
        Method("adaptive_scalar", adaptive_common),
    ]
    if len(case.block) > 1:
        methods.append(Method("adaptive_block_scalar", {**adaptive_common, "blocks": [case.block]}))
    return tuple(methods)


def _counted(model_fn: Callable[..., Any], counter: list[int]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        counter[0] += 1
        return model_fn(*args, **kwargs)

    return wrapped


def _run(case: Case, method: Method, *, seed: int) -> Run:
    counter = [0]
    model_fn = _counted(case.model_fn, counter)
    chains: list[np.ndarray] = []
    query_chains: list[list[np.ndarray]] = [[] for _ in case.queries]
    start = time.perf_counter()
    for chain_index in range(CHAINS):
        posterior = infer(
            model_fn,
            *case.args,
            method="mh" if method.name == "legacy_mh" else "adaptive_mh",
            num_samples=DRAWS,
            seed=np.random.SeedSequence([seed, chain_index]),
            **method.infer_kwargs,
        )
        for query_index, query in enumerate(case.queries):
            values = np.asarray([query.estimand(value) for value in posterior.samples], dtype=float)
            if values.shape != (DRAWS,) or not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"{case.name}/{method.name}/{query.name}: retained values were not finite scalar draws"
                )
            query_chains[query_index].append(values)
        chains.append(query_chains[0][-1])
    elapsed = time.perf_counter() - start
    values = np.stack(chains)
    diagnostics = diagnose_chains(values)
    errors = tuple(
        (
            abs(float(np.mean(query_chains[index])) - query.reference.mean),
            abs(float(np.var(query_chains[index], ddof=1)) - query.reference.variance),
        )
        for index, query in enumerate(case.queries)
    )
    ess_rate = diagnostics.ess_bulk / elapsed if diagnostics.status == "ok" and elapsed > 0 else math.nan
    return Run(
        wall_s=elapsed,
        invocations=counter[0],
        errors=errors,
        rhat=diagnostics.rhat,
        ess_per_second=ess_rate,
        ess_per_invocation=diagnostics.ess_bulk / counter[0] if diagnostics.status == "ok" and counter[0] else math.nan,
        mcse=diagnostics.mcse_mean,
        diagnostic_status=diagnostics.status,
    )


def _format(values: list[float | int | None], digits: int = 3) -> str:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(present) != len(values) or not present:
        return "n/a"
    if len(present) == 1:
        return f"{present[0]:.{digits}g}"
    mean = statistics.mean(present)
    # t(0.975, 2) for the predeclared three independent replicates.
    half_width = (
        4.303 * statistics.stdev(present) / math.sqrt(len(present)) if len(present) == 3 else statistics.stdev(present)
    )
    return f"{mean:.{digits}g}[{mean - half_width:.{digits}g},{mean + half_width:.{digits}g}]"


def _failure_notes(runs: list[Run]) -> str:
    failures = [run for run in runs if run.diagnostic_status != "ok" or run.rhat > RHAT_FAILURE]
    if not failures:
        return "none"
    rhat_failures = sum(run.rhat > RHAT_FAILURE for run in failures if math.isfinite(run.rhat))
    statuses = sorted({run.diagnostic_status for run in failures})
    return f"{len(failures)}/{len(runs)} (rhat>{RHAT_FAILURE}: {rhat_failures}; status={','.join(statuses)})"


def _print_summary(case: Case, method: Method, runs: list[Run]) -> None:
    error_columns = (
        " ".join(
            f"{query.name}_mean_abs_err={_format([run.errors[index][0] for run in runs])} "
            f"{query.name}_var_abs_err={_format([run.errors[index][1] for run in runs])}"
            for index, query in enumerate(case.queries)
        )
        + " "
    )
    print(
        f"{case.name}/{method.name} "
        f"wall_s={_format([run.wall_s for run in runs])} "
        f"invocations={_format([run.invocations for run in runs], 4)} "
        f"{error_columns}"
        f"rhat={_format([run.rhat for run in runs])} "
        f"bulk_ess_s={_format([run.ess_per_second for run in runs])} "
        f"bulk_ess_invocation={_format([run.ess_per_invocation for run in runs])} "
        f"mcse={_format([run.mcse for run in runs])} "
        f"failures={_failure_notes(runs)}"
    )


def _smoke() -> None:
    """Bounded API/correctness fixture; not benchmark evidence."""
    global CHAINS, WARMUP, DRAWS
    old_budget = (CHAINS, WARMUP, DRAWS)
    CHAINS, WARMUP, DRAWS = 2, 10, 20
    try:
        for case in CASES:
            for method in _methods(case):
                run = _run(case, method, seed=BASE_SEED)
                if run.invocations < CHAINS * DRAWS:
                    raise RuntimeError(f"{case.name}/{method.name}: retained sampling made too few model invocations")
        print("smoke=ok")
    finally:
        CHAINS, WARMUP, DRAWS = old_budget


def main() -> None:
    global WARMUP, DRAWS, REPLICATES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a small API fixture, not the benchmark budget.")
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Warmup transitions per chain.")
    parser.add_argument("--draws", type=int, default=DRAWS, help="Retained draws per chain.")
    parser.add_argument("--replicates", type=int, default=REPLICATES, help="Independent replicate count.")
    parser.add_argument("--case", choices=[case.name for case in CASES], help="Run one named case.")
    args = parser.parse_args()
    if min(args.warmup, args.draws, args.replicates) < 1:
        parser.error("--warmup, --draws, and --replicates must be positive")
    WARMUP, DRAWS, REPLICATES = args.warmup, args.draws, args.replicates
    if args.smoke:
        _smoke()
        return

    print(
        f"budget=replicates:{REPLICATES},chains:{CHAINS},warmup:{WARMUP},retained:{DRAWS}; "
        f"base_seed={BASE_SEED}; uncertainty=mean[approx_95pct_t_CI_across_replicates]"
    )
    print("linear_regression_example=compatible (explicit named scalar sites: slope, intercept, noise)")
    selected_cases = tuple(case for case in CASES if args.case is None or case.name == args.case)
    for case_index, case in enumerate(selected_cases):
        for method_index, method in enumerate(_methods(case)):
            runs = [
                _run(case, method, seed=BASE_SEED + 10_000 * case_index + 100 * method_index + replicate)
                for replicate in range(REPLICATES)
            ]
            _print_summary(case, method, runs)
        if len(case.block) == 1:
            print(
                f"{case.name:25s} adaptive_block_scalar   "
                "incompatible (adaptive_mh blocks require at least two distinct named sites)"
            )


if __name__ == "__main__":
    main()
