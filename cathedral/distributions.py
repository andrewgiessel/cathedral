"""Probability distributions for Cathedral.

Each distribution supports sampling and log-probability evaluation.
Backed by scipy.stats internally, with a minimal interface designed
for future compatibility with NumPyro/PyTorch backends.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import stats

from cathedral._rng import get_active_rng


class Distribution(ABC):
    """Base class for probability distributions."""

    @abstractmethod
    def sample(self) -> Any:
        """Draw a random sample from this distribution."""

    @abstractmethod
    def log_prob(self, value: Any) -> float:
        """Compute the log-probability (or log-density) of a value."""

    def prob(self, value: Any) -> float:
        """Compute the probability (or density) of a value."""
        return math.exp(self.log_prob(value))

    def support(self) -> list[Any] | None:
        """Return the finite support of this distribution, or None if infinite/continuous."""
        return None


class Bernoulli(Distribution):
    """Bernoulli distribution: returns True with probability p."""

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p

    def sample(self) -> bool:
        return bool(get_active_rng().random() < self.p)

    def log_prob(self, value: Any) -> float:
        if value:
            return math.log(self.p) if self.p > 0 else -math.inf
        return math.log(1 - self.p) if self.p < 1 else -math.inf

    def support(self) -> list[bool]:
        return [True, False]

    def __repr__(self) -> str:
        return f"Bernoulli(p={self.p})"


class Categorical(Distribution):
    """Categorical distribution over a list of values with given probabilities."""

    def __init__(self, values: list, probs: list[float]):
        if len(values) != len(probs):
            raise ValueError("values and probs must have the same length")
        total = sum(probs)
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(f"probs must sum to 1, got {total}")
        self.values = list(values)
        self.probs = list(probs)
        self._log_probs = {v: math.log(p) if p > 0 else -math.inf for v, p in zip(values, probs, strict=False)}

    def sample(self) -> Any:
        idx = get_active_rng().choice(len(self.values), p=self.probs)
        return self.values[idx]

    def log_prob(self, value: Any) -> float:
        if value in self._log_probs:
            return self._log_probs[value]
        return -math.inf

    def support(self) -> list[Any]:
        return list(self.values)

    def __repr__(self) -> str:
        return f"Categorical(values={self.values}, probs={self.probs})"


_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


class Normal(Distribution):
    """Normal (Gaussian) distribution."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.mu = mu
        self.sigma = sigma
        self._log_sigma = math.log(sigma)

    def sample(self) -> float:
        return float(get_active_rng().normal(self.mu, self.sigma))

    def log_prob(self, value: Any) -> float:
        z = (value - self.mu) / self.sigma
        return -0.5 * z * z - self._log_sigma - _LOG_SQRT_2PI

    def __repr__(self) -> str:
        return f"Normal(mu={self.mu}, sigma={self.sigma})"


_LOG_2 = math.log(2)


class HalfNormal(Distribution):
    """Half-normal distribution (positive values only)."""

    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self._log_sigma = math.log(sigma)

    def sample(self) -> float:
        return float(abs(get_active_rng().normal(0, self.sigma)))

    def log_prob(self, value: Any) -> float:
        if value < 0:
            return -math.inf
        z = value / self.sigma
        return _LOG_2 - 0.5 * z * z - self._log_sigma - _LOG_SQRT_2PI

    def __repr__(self) -> str:
        return f"HalfNormal(sigma={self.sigma})"


class Beta(Distribution):
    """Beta distribution on [0, 1]."""

    def __init__(self, a: float = 1.0, b: float = 1.0):
        if a <= 0 or b <= 0:
            raise ValueError(f"a and b must be positive, got a={a}, b={b}")
        self.a = a
        self.b = b
        self._dist = stats.beta(a, b)

    def sample(self) -> float:
        return float(get_active_rng().beta(self.a, self.b))

    def log_prob(self, value: Any) -> float:
        if not 0 <= value <= 1:
            return -math.inf
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"Beta(a={self.a}, b={self.b})"


class Gamma(Distribution):
    """Gamma distribution (shape/rate parameterization)."""

    def __init__(self, shape: float = 1.0, rate: float = 1.0):
        if shape <= 0 or rate <= 0:
            raise ValueError(f"shape and rate must be positive, got shape={shape}, rate={rate}")
        self.shape = shape
        self.rate = rate
        self._scale = 1.0 / rate
        self._dist = stats.gamma(a=shape, scale=self._scale)

    def sample(self) -> float:
        return float(get_active_rng().gamma(self.shape, self._scale))

    def log_prob(self, value: Any) -> float:
        if value < 0:
            return -math.inf
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"Gamma(shape={self.shape}, rate={self.rate})"


class Exponential(Distribution):
    """Exponential distribution with rate parameterization."""

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self._scale = 1.0 / rate
        self._dist = stats.expon(scale=self._scale)

    def sample(self) -> float:
        return float(get_active_rng().exponential(self._scale))

    def log_prob(self, value: Any) -> float:
        if value < 0:
            return -math.inf
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"Exponential(rate={self.rate})"


class Uniform(Distribution):
    """Uniform distribution on [low, high]."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        if low >= high:
            raise ValueError(f"low must be less than high, got low={low}, high={high}")
        self.low = low
        self.high = high
        self._log_prob_val = -math.log(high - low)

    def sample(self) -> float:
        return float(get_active_rng().uniform(self.low, self.high))

    def log_prob(self, value: Any) -> float:
        if self.low <= value <= self.high:
            return self._log_prob_val
        return -math.inf

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"


class LogNormal(Distribution):
    """Log-normal distribution parameterized by underlying normal mu/sigma."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.lognorm(s=sigma, scale=math.exp(mu))

    def sample(self) -> float:
        return float(get_active_rng().lognormal(self.mu, self.sigma))

    def log_prob(self, value: Any) -> float:
        if value <= 0:
            return -math.inf
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"LogNormal(mu={self.mu}, sigma={self.sigma})"


class Poisson(Distribution):
    """Poisson distribution."""

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self._dist = stats.poisson(mu=rate)

    def sample(self) -> int:
        return int(get_active_rng().poisson(self.rate))

    def log_prob(self, value: Any) -> float:
        if value < 0 or value != int(value):
            return -math.inf
        return float(self._dist.logpmf(int(value)))

    def __repr__(self) -> str:
        return f"Poisson(rate={self.rate})"


class Binomial(Distribution):
    """Binomial distribution: number of successes in n Bernoulli trials."""

    def __init__(self, n: int, p: float):
        if n < 0 or n != int(n):
            raise ValueError(f"n must be a non-negative integer, got {n}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.n = int(n)
        self.p = p
        self._dist = stats.binom(n=self.n, p=p)

    def sample(self) -> int:
        return int(get_active_rng().binomial(self.n, self.p))

    def log_prob(self, value: Any) -> float:
        if value < 0 or value > self.n or value != int(value):
            return -math.inf
        return float(self._dist.logpmf(int(value)))

    def support(self) -> list[int]:
        return list(range(self.n + 1))

    def __repr__(self) -> str:
        return f"Binomial(n={self.n}, p={self.p})"


class UniformDraw(Distribution):
    """Uniform draw from a finite set of values."""

    def __init__(self, values: list):
        if len(values) == 0:
            raise ValueError("values must be non-empty")
        self.values = list(values)
        self._log_p = -math.log(len(values))

    def sample(self) -> Any:
        idx = get_active_rng().integers(len(self.values))
        return self.values[idx]

    def log_prob(self, value: Any) -> float:
        if value in self.values:
            return self._log_p
        return -math.inf

    def support(self) -> list[Any]:
        return list(self.values)

    def __repr__(self) -> str:
        return f"UniformDraw(values={self.values})"


class Geometric(Distribution):
    """Geometric distribution (number of failures before first success)."""

    def __init__(self, p: float = 0.5):
        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p
        self._dist = stats.geom(p)

    def sample(self) -> int:
        return int(get_active_rng().geometric(self.p)) - 1

    def log_prob(self, value: Any) -> float:
        if value < 0 or value != int(value):
            return -math.inf
        return float(self._dist.logpmf(int(value) + 1))

    def __repr__(self) -> str:
        return f"Geometric(p={self.p})"


class Dirichlet(Distribution):
    """Dirichlet distribution over probability simplices."""

    def __init__(self, alpha: list[float] | np.ndarray):
        alpha_arr = np.asarray(alpha, dtype=float)
        if np.any(alpha_arr <= 0):
            raise ValueError("all alpha values must be positive")
        self.alpha = alpha_arr
        self._dist = stats.dirichlet(alpha_arr)

    def sample(self) -> np.ndarray:
        return get_active_rng().dirichlet(self.alpha)

    def log_prob(self, value: Any) -> float:
        value = np.asarray(value, dtype=float)
        if not math.isclose(value.sum(), 1.0, rel_tol=1e-6):
            return -math.inf
        if np.any(value < 0):
            return -math.inf
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"Dirichlet(alpha={self.alpha.tolist()})"


class StudentT(Distribution):
    """Student's t distribution."""

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0):
        if df <= 0:
            raise ValueError(f"df must be positive, got {df}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.df = df
        self.loc = loc
        self.scale = scale
        self._dist = stats.t(df=df, loc=loc, scale=scale)

    def sample(self) -> float:
        return float(self.loc + self.scale * get_active_rng().standard_t(self.df))

    def log_prob(self, value: Any) -> float:
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"StudentT(df={self.df}, loc={self.loc}, scale={self.scale})"
