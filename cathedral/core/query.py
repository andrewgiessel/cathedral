from cathedral.core.eval import eval_church, register_primitive


def query(expr, pred, env=None):
    """Sample from the conditional distribution.

    Args:
        expr: The expression to evaluate
        pred: A predicate function that takes the value of expr and returns True/False
        env: The environment to evaluate in (optional)

    Returns:
        A sample from the conditional distribution of expr given pred is True.
    """
    if env is None:
        env = {}

    while True:
        val = eval_church(expr, env)
        if pred(val):
            return val


def lex_query(lexicon, expr, pred, env=None):
    """Lexicalized query with shared random values.

    Args:
        lexicon: A list of bindings (name, definition) to include in the query environment
        expr: The expression to evaluate
        pred: The predicate expression to evaluate
        env: The environment to evaluate in (optional)

    Returns:
        A sample from the conditional distribution.
    """
    if env is None:
        env = {}

    # Create a local environment with the lexicon bindings
    local_env = env.copy()
    for name, defn in eval_church(lexicon, env):
        local_env[name] = eval_church(defn, local_env)

    # Evaluate the expression in the local environment
    val = eval_church(expr, local_env)

    # Evaluate the predicate in the local environment
    pred_result = eval_church(pred, local_env)

    # If predicate is True, return the value, otherwise try again
    if pred_result:
        return val
    else:
        return lex_query(lexicon, expr, pred, env)


def mh_query(num_samples, burn_in, expr, pred, env=None):
    """Metropolis-Hastings query implementation.

    This is a placeholder for the MCMC-based implementation described in the paper.
    Currently it just implements rejection sampling.

    Args:
        num_samples: Number of samples to draw
        burn_in: Number of burn-in samples to discard
        expr: The expression to evaluate
        pred: The predicate expression
        env: The environment to evaluate in (optional)

    Returns:
        A list of samples from the conditional distribution.
    """
    if env is None:
        env = {}

    # For now, just implement as rejection sampling
    samples = []
    while len(samples) < num_samples + burn_in:
        val = eval_church(expr, env)
        pred_result = eval_church(pred, env)
        if pred_result:
            samples.append(val)

    # Discard burn-in samples
    return samples[burn_in:]


def register_query_primitives():
    """Register all query primitives."""
    register_primitive("query", query)
    register_primitive("lex-query", lex_query)
    register_primitive("mh-query", mh_query)
