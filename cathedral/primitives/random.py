import numpy as np

from cathedral.core.eval import register_primitive


def flip_dist(env, args):
    """Distribution function for flip."""
    p = args[0] if args else 0.5
    return {True: p, False: 1 - p}


def flip(p=0.5):
    """Return True with probability p, False with probability 1-p."""
    return np.random.random() < p


def random_integer(n):
    """Return a random integer between 0 and n-1."""
    return np.random.randint(n)


def random_real():
    """Return a random real number between 0 and 1."""
    return np.random.random()


def beta_dist(a, b):
    """Distribution function for beta."""
    # This is a simplification - in a full implementation we'd return
    # a probability density function
    return {
        "pdf": lambda x: (x ** (a - 1) * (1 - x) ** (b - 1))
        / (np.math.gamma(a) * np.math.gamma(b) / np.math.gamma(a + b))
    }


def beta(a, b):
    """Return a sample from beta distribution with parameters a, b."""
    return np.random.beta(a, b)


def normal_dist(mu, sigma):
    """Distribution function for normal."""
    # This is a simplification - in a full implementation we'd return
    # a probability density function
    return {"pdf": lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)}


def normal(mu, sigma):
    """Return a sample from normal distribution with parameters mu, sigma."""
    return np.random.normal(mu, sigma)


def gamma_dist(shape, scale):
    """Distribution function for gamma."""
    # This is a simplification - in a full implementation we'd return
    # a probability density function
    return {"pdf": lambda x: (x ** (shape - 1) * np.exp(-x / scale)) / (scale**shape * np.math.gamma(shape))}


def gamma(shape, scale):
    """Return a sample from gamma distribution with parameters shape, scale."""
    return np.random.gamma(shape, scale)


def uniform_dist(a, b):
    """Distribution function for uniform."""
    # This is a simplification - in a full implementation we'd return
    # a probability density function
    return {"pdf": lambda x: 1 / (b - a) if a <= x < b else 0}


def uniform(a, b):
    """Return a sample from uniform distribution between a and b."""
    return np.random.uniform(a, b)


def multinomial_dist(values, probs):
    """Distribution function for multinomial."""
    return {val: p for val, p in zip(values, probs, strict=False)}


def multinomial(values, probs):
    """Return a sample from multinomial distribution."""
    idx = np.random.choice(len(values), p=probs)
    return values[idx]


def dirichlet(alpha):
    """Return a sample from Dirichlet distribution with parameter alpha."""
    return np.random.dirichlet(alpha)


def register_random_primitives():
    """Register all the random primitives."""
    register_primitive("flip", flip, flip_dist)
    register_primitive("random-integer", random_integer)
    register_primitive("random-real", random_real)
    register_primitive("beta", beta, beta_dist)
    register_primitive("normal", normal, normal_dist)
    register_primitive("gamma", gamma, gamma_dist)
    register_primitive("uniform", uniform, uniform_dist)
    register_primitive("multinomial", multinomial, multinomial_dist)
    register_primitive("dirichlet", dirichlet)
