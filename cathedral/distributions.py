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


class Bernoulli(Distribution):
    """Bernoulli distribution: returns True with probability p."""

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p

    def sample(self) -> bool:
        return bool(np.random.random() < self.p)

    def log_prob(self, value: Any) -> float:
        if value:
            return math.log(self.p) if self.p > 0 else -math.inf
        return math.log(1 - self.p) if self.p < 1 else -math.inf

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
        self._log_probs = {v: math.log(p) if p > 0 else -math.inf for v, p in zip(values, probs)}

    def sample(self) -> Any:
        idx = np.random.choice(len(self.values), p=self.probs)
        return self.values[idx]

    def log_prob(self, value: Any) -> float:
        if value in self._log_probs:
            return self._log_probs[value]
        return -math.inf

    def __repr__(self) -> str:
        return f"Categorical(values={self.values}, probs={self.probs})"


class Normal(Distribution):
    """Normal (Gaussian) distribution."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.norm(loc=mu, scale=sigma)

    def sample(self) -> float:
        return float(np.random.normal(self.mu, self.sigma))

    def log_prob(self, value: Any) -> float:
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"Normal(mu={self.mu}, sigma={self.sigma})"


class HalfNormal(Distribution):
    """Half-normal distribution (positive values only)."""

    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self._dist = stats.halfnorm(scale=sigma)

    def sample(self) -> float:
        return float(abs(np.random.normal(0, self.sigma)))

    def log_prob(self, value: Any) -> float:
        if value < 0:
            return -math.inf
        return float(self._dist.logpdf(value))

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
        return float(np.random.beta(self.a, self.b))

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
        return float(np.random.gamma(self.shape, self._scale))

    def log_prob(self, value: Any) -> float:
        if value < 0:
            return -math.inf
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"Gamma(shape={self.shape}, rate={self.rate})"


class Uniform(Distribution):
    """Uniform distribution on [low, high]."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        if low >= high:
            raise ValueError(f"low must be less than high, got low={low}, high={high}")
        self.low = low
        self.high = high
        self._log_prob_val = -math.log(high - low)

    def sample(self) -> float:
        return float(np.random.uniform(self.low, self.high))

    def log_prob(self, value: Any) -> float:
        if self.low <= value <= self.high:
            return self._log_prob_val
        return -math.inf

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"


class Poisson(Distribution):
    """Poisson distribution."""

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self._dist = stats.poisson(mu=rate)

    def sample(self) -> int:
        return int(np.random.poisson(self.rate))

    def log_prob(self, value: Any) -> float:
        if value < 0 or value != int(value):
            return -math.inf
        return float(self._dist.logpmf(int(value)))

    def __repr__(self) -> str:
        return f"Poisson(rate={self.rate})"


class Dirichlet(Distribution):
    """Dirichlet distribution over probability simplices."""

    def __init__(self, alpha: list[float] | np.ndarray):
        alpha_arr = np.asarray(alpha, dtype=float)
        if np.any(alpha_arr <= 0):
            raise ValueError("all alpha values must be positive")
        self.alpha = alpha_arr
        self._dist = stats.dirichlet(alpha_arr)

    def sample(self) -> np.ndarray:
        return np.random.dirichlet(self.alpha)

    def log_prob(self, value: Any) -> float:
        value = np.asarray(value, dtype=float)
        if not math.isclose(value.sum(), 1.0, rel_tol=1e-6):
            return -math.inf
        if np.any(value < 0):
            return -math.inf
        return float(self._dist.logpdf(value))

    def __repr__(self) -> str:
        return f"Dirichlet(alpha={self.alpha.tolist()})"
