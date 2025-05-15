import numpy as np

from cathedral.core.eval import register_primitive
from cathedral.primitives.random import beta


class MemoizedProcedure:
    """A memoized procedure that caches results for each set of arguments."""

    def __init__(self, proc):
        """Initialize with a procedure to memoize.

        Args:
            proc: A callable procedure
        """
        self.proc = proc
        self.cache = {}

    def __call__(self, *args):
        """Apply the memoized procedure to arguments.

        If the procedure has been called with these arguments before,
        returns the cached result. Otherwise, evaluates the procedure
        and caches the result.
        """
        # Convert args to a hashable representation
        args_key = self._make_hashable(args)

        # Return cached value if it exists
        if args_key in self.cache:
            return self.cache[args_key]

        # Otherwise, evaluate and cache
        result = self.proc(*args)
        self.cache[args_key] = result
        return result

    def _make_hashable(self, args):
        """Convert arguments to a hashable representation for caching."""
        hashable_args = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                hashable_args.append(tuple(self._make_hashable(arg)))
            elif isinstance(arg, dict):
                hashable_args.append(tuple(sorted((k, self._make_hashable((v,))[0]) for k, v in arg.items())))
            elif isinstance(arg, np.ndarray):
                hashable_args.append(arg.tobytes())
            else:
                hashable_args.append(arg)
        return tuple(hashable_args)


class DPMemoizedProcedure:
    """A Dirichlet Process memoized procedure.

    This implements the stochastic memoizer with Dirichlet Process caching as described
    in the Cathedral paper. It caches results for each set of arguments but may
    stochastically generate new values based on concentration parameter alpha.
    """

    def __init__(self, alpha, proc):
        """Initialize with a concentration parameter and procedure.

        Args:
            alpha: Concentration parameter (0 = always reuse cached values,
                  infinity = always generate new values)
            proc: A callable procedure
        """
        self.alpha = alpha
        self.proc = proc
        # For each set of arguments, we maintain:
        # - A list of values previously returned
        # - A list of counts for how many times each value has been returned
        self.caches = {}

    def __call__(self, *args):
        """Apply the DP-memoized procedure to arguments.

        Returns a value from the cache with probability proportional to how many
        times it has been returned before, or a new value with probability
        proportional to alpha.
        """
        # Convert args to a hashable representation
        args_key = self._make_hashable(args)

        # If we haven't seen these args before, initialize cache
        if args_key not in self.caches:
            self.caches[args_key] = {"values": [], "counts": []}

        cache = self.caches[args_key]

        # Decide whether to reuse a cached value or generate a new one
        if not cache["values"] or np.random.random() < self.alpha / (self.alpha + sum(cache["counts"])):
            # Generate a new value
            result = self.proc(*args)
            cache["values"].append(result)
            cache["counts"].append(1)
        else:
            # Choose an existing value with probability proportional to its count
            probs = np.array(cache["counts"]) / sum(cache["counts"])
            idx = np.random.choice(len(cache["values"]), p=probs)
            result = cache["values"][idx]
            cache["counts"][idx] += 1

        return result

    def _make_hashable(self, args):
        """Convert arguments to a hashable representation for caching."""
        hashable_args = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                hashable_args.append(tuple(self._make_hashable(arg)))
            elif isinstance(arg, dict):
                hashable_args.append(tuple(sorted((k, self._make_hashable((v,))[0]) for k, v in arg.items())))
            elif isinstance(arg, np.ndarray):
                hashable_args.append(arg.tobytes())
            else:
                hashable_args.append(arg)
        return tuple(hashable_args)


def mem(proc):
    """Memoize a procedure, caching results for each set of arguments."""
    return MemoizedProcedure(proc)


def DPmem(alpha, proc):
    """Create a Dirichlet Process memoized procedure.

    Args:
        alpha: Concentration parameter (0 = always reuse cached values,
              infinity = always generate new values)
        proc: Procedure to memoize
    """
    return DPMemoizedProcedure(alpha, proc)


def pick_a_stick(sticks_fn, j):
    """Helper function for the DP implementation."""
    if np.random.random() < sticks_fn(j):
        return j
    else:
        return pick_a_stick(sticks_fn, j + 1)


def DP(alpha, proc):
    """Create a Dirichlet Process with base measure defined by proc.

    This implements the DP helper function as shown in the paper.
    """
    sticks = MemoizedProcedure(lambda x: beta(1.0, alpha))
    atoms = MemoizedProcedure(lambda x: proc())

    def sample_dp():
        return atoms(pick_a_stick(sticks, 1))

    return sample_dp


def register_memoization_primitives():
    """Register all memoization primitives."""
    register_primitive("mem", mem)
    register_primitive("DPmem", DPmem)
    register_primitive("DP", DP)
    register_primitive("pick-a-stick", pick_a_stick)
