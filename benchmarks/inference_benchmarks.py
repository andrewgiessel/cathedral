"""Benchmark seeded serial inference on representative models.

Usage:
    uv run python benchmarks/inference_benchmarks.py
    uv run python benchmarks/inference_benchmarks.py --repeats 3 --sample-multiplier 10
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from cathedral import Bernoulli, Beta, HalfNormal, Normal, UniformDraw, condition, infer, model, observe, sample


@model
def sprinkler():
    """Classic rejection-sampling benchmark with hard conditioning."""
    rain = sample(Bernoulli(0.3), name="rain")
    sprinkler_on = sample(Bernoulli(0.5), name="sprinkler")
    if rain:
        wet = sample(Bernoulli(0.9), name="wet_if_rain")
    elif sprinkler_on:
        wet = sample(Bernoulli(0.8), name="wet_if_sprinkler")
    else:
        wet = sample(Bernoulli(0.1), name="wet_if_dry")
    condition(wet)
    return {"rain": rain, "sprinkler": sprinkler_on}


@model
def bag_of_marbles(draws: list[str]):
    """Simple categorical rejection benchmark."""
    bag = sample(UniformDraw(["A", "B"]), name="bag")
    p_blue = 0.8 if bag == "A" else 0.2
    for i, draw in enumerate(draws):
        condition(sample(Bernoulli(p_blue), name=f"draw_{i}") == (draw == "blue"))
    return bag


@model
def coin_bias_estimation(observations: list[bool]):
    """Importance-sampling benchmark from the examples."""
    p = sample(Beta(1, 1), name="p")
    for outcome in observations:
        observe(Bernoulli(p), outcome)
    return p


@model
def bayesian_regression(xs: list[float], ys: list[float]):
    """Importance-sampling regression benchmark."""
    slope = sample(Normal(0, 5), name="slope")
    intercept = sample(Normal(0, 5), name="intercept")
    noise = sample(HalfNormal(2), name="noise")
    for x, y in zip(xs, ys, strict=False):
        observe(Normal(slope * x + intercept, noise), y)
    return {"slope": slope, "intercept": intercept, "noise": noise}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    method: str
    model_fn: Callable[..., Any]
    args: tuple[Any, ...]
    num_samples: int


@dataclass(frozen=True)
class BenchmarkResult:
    elapsed_s: float
    attempts: int | None
    ess: float | None


CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="rejection_sprinkler",
        method="rejection",
        model_fn=sprinkler,
        args=(),
        num_samples=4000,
    ),
    BenchmarkCase(
        name="rejection_bag_of_marbles",
        method="rejection",
        model_fn=bag_of_marbles,
        args=(["blue", "blue", "red"],),
        num_samples=4000,
    ),
    BenchmarkCase(
        name="importance_coin_bias_estimation",
        method="importance",
        model_fn=coin_bias_estimation,
        args=([True, True, True, True, True, True, True, False, False, False],),
        num_samples=4000,
    ),
    BenchmarkCase(
        name="importance_bayesian_regression",
        method="importance",
        model_fn=bayesian_regression,
        args=(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2.1, 4.3, 5.8, 8.2, 9.9, 12.1, 14.0, 16.1],
        ),
        num_samples=4000,
    ),
]


def run_case(case: BenchmarkCase, *, seed: int, num_samples: int) -> BenchmarkResult:
    start = time.perf_counter()
    posterior = infer(
        case.model_fn,
        *case.args,
        method=case.method,
        num_samples=num_samples,
        seed=seed,
    )
    elapsed_s = time.perf_counter() - start
    info = posterior.info
    attempts = info.num_attempts if info is not None else None
    ess = posterior.ess
    return BenchmarkResult(elapsed_s=elapsed_s, attempts=attempts, ess=ess)


def summarize_metric(values: Sequence[float | None]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    if len(present) == 1:
        return f"{present[0]:.2f}"
    return f"{statistics.mean(present):.2f} +/- {statistics.pstdev(present):.2f}"


def attempts_per_second(result: BenchmarkResult) -> float | None:
    if result.attempts is None or result.elapsed_s == 0:
        return None
    return result.attempts / result.elapsed_s


def ess_per_second(result: BenchmarkResult) -> float | None:
    if result.ess is None or result.elapsed_s == 0:
        return None
    return result.ess / result.elapsed_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Number of benchmark repetitions per case")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for reproducible benchmark runs")
    parser.add_argument(
        "--sample-multiplier",
        type=int,
        default=1,
        help="Multiply each case's default num_samples by this factor",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional subset of case names to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_multiplier < 1:
        raise SystemExit("--sample-multiplier must be >= 1")
    cases = CASES if args.cases is None else [case for case in CASES if case.name in set(args.cases)]
    if not cases:
        raise SystemExit("No benchmark cases selected.")

    print("Cathedral inference benchmarks")
    print(f"repeats={args.repeats} seed={args.seed} sample_multiplier={args.sample_multiplier}")
    print("")

    for case in cases:
        effective_num_samples = case.num_samples * args.sample_multiplier
        print(f"[{case.name}] method={case.method} num_samples={effective_num_samples}")
        print("wall_s           | attempts_per_s   | ess             | ess_per_s")
        print("---------------- | ---------------- | --------------- | ----------------")
        runs = [
            run_case(
                case,
                seed=args.seed + repeat,
                num_samples=effective_num_samples,
            )
            for repeat in range(args.repeats)
        ]
        wall = summarize_metric([run.elapsed_s for run in runs])
        attempts = summarize_metric([attempts_per_second(run) for run in runs])
        ess = summarize_metric([run.ess for run in runs])
        ess_rate = summarize_metric([ess_per_second(run) for run in runs])
        print(f"{wall:<16} | {attempts:<16} | {ess:<15} | {ess_rate:<16}")
        print("")


if __name__ == "__main__":
    main()
