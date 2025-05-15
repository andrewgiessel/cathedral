import numpy as np
import pytest

from cathedral.core.eval import eval_church
from cathedral.core.query import register_query_primitives
from cathedral.primitives.memoization import register_memoization_primitives
from cathedral.primitives.random import register_random_primitives

# Register primitives before testing
register_random_primitives()
register_memoization_primitives()
register_query_primitives()


def test_simple_query():
    """Test a simple query condition."""
    # Query to get distribution of flips conditioned on being True
    church_program = """
    (query '(flip 0.3) (lambda (x) x))
    """
    # All results should be True
    result = eval_church(church_program, {})
    assert result == True


def test_query_simple_bayes():
    """Test a simple Bayesian inference problem."""
    # Prior P(A) = 0.3, Likelihood P(B|A) = 0.8, P(B|¬A) = 0.4
    # Query P(A|B)
    church_program = """
    (define samples 
      (repeat 1000
        (lambda ()
          (query 
            '(define a (flip 0.3))
            '(define b (if a (flip 0.8) (flip 0.4)))
            '(define c (if a (flip 0.9) (flip 0.2)))
            'a
            'b))))
    
    ; Compute empirical probability
    (/ (sum samples) 1000)
    """

    # Define helper functions for this test
    env = {}
    eval_church("(define repeat (lambda (n f) (if (= n 0) '() (cons (f) (repeat (- n 1) f)))))", env)
    eval_church("(define sum (lambda (lst) (if (null? lst) 0 (+ (if (first lst) 1 0) (sum (rest lst))))))", env)

    # Run the query
    result = eval_church(church_program, env)

    # Should be approximately P(A|B) = P(B|A)P(A)/P(B) = (0.8*0.3)/(0.8*0.3 + 0.4*0.7) = 0.24/0.52 ≈ 0.462
    assert 0.4 < result < 0.55  # Give a wide margin for randomness


def test_lex_query():
    """Test lexicalized query."""
    # Simple grass-sprinkler-rain example from the paper
    church_program = """
    (lex-query
      '((grass-is-wet (mem (lambda (day)
                           (if (or (and (rain day) (flip 0.9))
                                   (and (sprinkler day) (flip 0.8)))
                               True
                               (flip 0.1)))))
        (rain (mem (lambda (day) (flip 0.3))))
        (sprinkler (mem (lambda (day) (flip 0.5)))))
      '(rain 'day2)
      '(grass-is-wet 'day2))
    """

    # Run multiple times and collect results
    np.random.seed(42)
    env = {}
    samples = [eval_church(church_program, env) for _ in range(100)]
    prob_rain_given_wet = sum(samples) / len(samples)

    # Reasonable range for P(rain|wet) in this model
    assert 0.3 < prob_rain_given_wet < 0.7


def test_mcmc_query():
    """Test MCMC-based query with a more complex model."""
    # Skip this test for now as MCMC will be implemented later
    pytest.skip("MCMC query implementation will be added later")

    # Model for testing will be a mixture model
    church_program = """
    (mh-query
      1000  ; num samples
      100   ; burn-in
      '(define mixture-component (if (flip 0.7) 0 1))
      '(define x (if (= mixture-component 0)
                   (normal 0 1)
                   (normal 5 1)))
      'mixture-component
      '(and (> x 4) (< x 6)))
    """

    # We expect mixture component 1 to be more likely given 4 < x < 6
    samples = eval_church(church_program, {})
    prob_component1 = sum(samples) / len(samples)
    assert prob_component1 > 0.9  # Component 1 should be highly probable
