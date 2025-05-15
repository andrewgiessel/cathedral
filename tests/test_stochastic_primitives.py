import numpy as np

from cathedral.core.eval import eval_church
from cathedral.primitives.random import register_random_primitives

# Register random primitives before testing
register_random_primitives()


def test_flip():
    """Test the flip primitive."""
    # Without arguments, flip should return True or False
    result = eval_church("(flip)", {})
    assert result in [True, False]

    # With a weight argument, flip should respect the weight
    np.random.seed(42)  # For reproducibility
    results = [eval_church("(flip 0.8)", {}) for _ in range(1000)]
    true_count = results.count(True)
    # Should be roughly 800, allow for some randomness
    assert 750 < true_count < 850


def test_random_integer():
    """Test the random-integer primitive."""
    # Should return an integer between 0 and n-1
    result = eval_church("(random-integer 10)", {})
    assert isinstance(result, int)
    assert 0 <= result < 10


def test_random_real():
    """Test the random-real primitive."""
    # Should return a float between 0 and 1
    result = eval_church("(random-real)", {})
    assert isinstance(result, float)
    assert 0 <= result < 1


def test_normal():
    """Test the normal primitive."""
    # Should return a sample from normal distribution
    result = eval_church("(normal 0 1)", {})
    assert isinstance(result, float)

    # Test with specific mean and std
    np.random.seed(42)
    samples = [eval_church("(normal 5 2)", {}) for _ in range(1000)]
    mean = np.mean(samples)
    std = np.std(samples)
    assert 4.8 < mean < 5.2
    assert 1.8 < std < 2.2


def test_uniform():
    """Test the uniform primitive."""
    # Should return a sample from uniform distribution
    result = eval_church("(uniform 10 20)", {})
    assert isinstance(result, float)
    assert 10 <= result < 20


def test_beta():
    """Test the beta primitive."""
    # Should return a sample from beta distribution
    result = eval_church("(beta 1 1)", {})
    assert isinstance(result, float)
    assert 0 <= result <= 1

    # Test with specific alpha and beta
    np.random.seed(42)
    samples = [eval_church("(beta 2 5)", {}) for _ in range(1000)]
    mean = np.mean(samples)
    # Mean of Beta(a,b) is a/(a+b) = 2/(2+5) = 2/7 ≈ 0.286
    assert 0.25 < mean < 0.32


def test_gamma():
    """Test the gamma primitive."""
    # Should return a sample from gamma distribution
    result = eval_church("(gamma 1 1)", {})
    assert isinstance(result, float)
    assert result > 0

    # Test with specific shape and scale
    np.random.seed(42)
    samples = [eval_church("(gamma 2 3)", {}) for _ in range(1000)]
    mean = np.mean(samples)
    # Mean of Gamma(k,θ) is k*θ = 2*3 = 6
    assert 5.5 < mean < 6.5
