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
    Geometric,
    HalfNormal,
    Normal,
    Poisson,
    Uniform,
    UniformDraw,
)
from cathedral.model import InferenceInfo, Posterior, infer, model
from cathedral.primitives import DPmem, condition, factor, flip, mem, observe, sample
from cathedral.trace import NeedsEnumeration, Rejected, Trace

__all__ = [
    "Bernoulli",
    "Beta",
    "Categorical",
    "DPmem",
    "Dirichlet",
    "Distribution",
    "Gamma",
    "Geometric",
    "HalfNormal",
    "InferenceInfo",
    "NeedsEnumeration",
    "Normal",
    "Poisson",
    "Posterior",
    "Rejected",
    "Trace",
    "Uniform",
    "UniformDraw",
    "condition",
    "factor",
    "flip",
    "infer",
    "mem",
    "model",
    "observe",
    "sample",
]
