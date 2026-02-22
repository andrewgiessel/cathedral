"""Cathedral: A Pythonic probabilistic programming library.

Cathedral provides Church/WebPPL-level expressiveness with pure Python syntax.
Write probabilistic models as decorated Python functions using primitives like
flip(), sample(), condition(), and observe(), then run inference to get posteriors.

Example:
    from cathedral import model, infer, flip, condition

    @model
    def coin_bias():
        fair = flip(0.5)
        result = flip(0.5 if fair else 0.9)
        condition(result)
        return fair

    posterior = infer(coin_bias, num_samples=5000)
    print(posterior.probability())  # P(fair | heads)
"""

from cathedral.distributions import (
    Bernoulli,
    Beta,
    Categorical,
    Dirichlet,
    Distribution,
    Gamma,
    HalfNormal,
    Normal,
    Poisson,
    Uniform,
)
from cathedral.model import Posterior, infer, model
from cathedral.primitives import condition, factor, flip, observe, sample
from cathedral.trace import Rejected, Trace

__all__ = [
    # Model and inference
    "model",
    "infer",
    "Posterior",
    # Primitives
    "sample",
    "flip",
    "condition",
    "observe",
    "factor",
    # Distributions
    "Distribution",
    "Bernoulli",
    "Beta",
    "Categorical",
    "Dirichlet",
    "Gamma",
    "HalfNormal",
    "Normal",
    "Poisson",
    "Uniform",
    # Trace
    "Trace",
    "Rejected",
]
