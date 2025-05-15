import hy
from hy.models import Complex, Expression, Float, Integer, List, String, Symbol


class CathedralError(Exception):
    """Base exception for Cathedral errors."""

    pass


class CathedralSyntaxError(CathedralError):
    """Exception for Cathedral syntax errors."""

    pass


class CathedralRuntimeError(CathedralError):
    """Exception for Cathedral runtime errors."""

    pass


class CathedralProcedure:
    """Represents a Cathedral procedure (lambda or elementary random procedure)."""

    def __init__(self, body, args, env, dist_func=None):
        """Initialize a Cathedral procedure.

        Args:
            body: The body of the procedure
            args: List of formal parameter symbols
            env: The environment in which the procedure was defined
            dist_func: For elementary random procedures, a probability distribution function
        """
        self.body = body
        self.args = args
        self.env = env
        self.dist_func = dist_func

    def __call__(self, *arg_values):
        """Apply the procedure to arguments."""
        if len(arg_values) != len(self.args):
            raise CathedralRuntimeError(f"Expected {len(self.args)} arguments, got {len(arg_values)}")

        # Extend the environment with bindings for arguments
        new_env = self.env.copy()
        for arg, value in zip(self.args, arg_values, strict=False):
            new_env[arg] = value

        # Evaluate the body in the new environment
        return eval_church(self.body, new_env)

    def is_elementary_random(self):
        """Check if this is an elementary random procedure."""
        return self.dist_func is not None

    def distribution(self, arg_values):
        """Return the probability distribution function for the given arguments.

        Only valid for elementary random procedures.
        """
        if not self.is_elementary_random():
            raise CathedralRuntimeError("Not an elementary random procedure")
        return self.dist_func(self.env, arg_values)


def parse_church(expr):
    """Parse a Cathedral expression into a Hy expression."""
    try:
        return hy.read(expr)
    except Exception as e:
        raise CathedralSyntaxError(f"Failed to parse expression: {expr}") from e


def eval_church(expr, env):
    """Evaluate a Cathedral expression in the given environment.

    Args:
        expr: The Cathedral expression to evaluate
        env: The environment (a dict mapping symbols to values)

    Returns:
        The result of evaluating the expression
    """
    # Parse the expression if it's a string - but only if it's an actual Python string, not a Hy model
    if isinstance(expr, str) and not isinstance(expr, hy.models.Object):
        expr = parse_church(expr)

    # Constants
    if isinstance(expr, (Integer, Float, Complex)):
        return int(expr) if isinstance(expr, Integer) else float(expr) if isinstance(expr, Float) else complex(expr)
    if expr == Symbol("True"):
        return True
    if expr == Symbol("False"):
        return False
    if expr == Symbol("nil"):
        return None
    
    # String literals
    if isinstance(expr, String):
        return str(expr)

    # Variables
    if isinstance(expr, Symbol):
        symbol_str = str(expr)
        if symbol_str in env:
            return env[symbol_str]
        raise CathedralRuntimeError(f"Undefined variable: {expr}")

    # Special forms
    if isinstance(expr, Expression):
        if not expr:
            return []

        op = expr[0]

        # Quote
        if op == Symbol("quote"):
            if len(expr) != 2:
                raise CathedralSyntaxError("quote requires exactly one argument")
            # Just return the quoted expression without evaluating it
            return expr[1]

        # Lambda expressions
        if op == Symbol("lambda"):
            if len(expr) < 3:
                raise CathedralSyntaxError("lambda requires at least 2 arguments: params and body")
            params = expr[1]
            if not isinstance(params, (List, Expression)):
                raise CathedralSyntaxError("lambda parameters must be a list")
            params = [str(p) for p in params]
            body = expr[2]
            return CathedralProcedure(body, params, env)

        # If expressions
        if op == Symbol("if"):
            if len(expr) != 4:
                raise CathedralSyntaxError("if requires exactly 3 arguments: condition, then-expr, else-expr")
            cond = eval_church(expr[1], env)
            if cond:
                return eval_church(expr[2], env)
            else:
                return eval_church(expr[3], env)

        # Define expressions
        if op == Symbol("define"):
            if len(expr) != 3:
                raise CathedralSyntaxError("define requires exactly 2 arguments: variable and value")
            var = expr[1]
            if not isinstance(var, Symbol):
                raise CathedralSyntaxError("define's first argument must be a symbol")
            val = eval_church(expr[2], env)
            env[str(var)] = val
            return env

        # Function application
        else:
            fn = eval_church(op, env)
            if not callable(fn):
                raise CathedralRuntimeError(f"Cannot apply non-procedure: {fn}")
            args = [eval_church(arg, env) for arg in expr[1:]]
            return fn(*args)

    # Lists, dicts, etc.
    return expr


# Global primitives registry
_PRIMITIVES = {}


def register_primitive(name, func, dist_func=None):
    """Register a primitive function.

    Args:
        name: The name of the primitive
        func: The Python function implementing the primitive
        dist_func: For random primitives, a function that returns the probability distribution
    """
    global _PRIMITIVES
    _PRIMITIVES[name] = (func, dist_func)


def get_primitive(name):
    """Get a primitive function by name."""
    if name in _PRIMITIVES:
        func, dist_func = _PRIMITIVES[name]
        return func, dist_func
    return None, None


# Register some basic primitives
register_primitive("+", lambda *args: sum(args))
register_primitive("-", lambda a, *args: a - sum(args) if args else -a)
register_primitive("*", lambda *args: 1 if not args else args[0] * eval("*(*args[1:])") if len(args) > 1 else args[0])
register_primitive("/", lambda a, *args: a / eval("*(*args)") if args else 1 / a)
register_primitive("=", lambda *args: all(args[0] == arg for arg in args[1:]))
register_primitive("<", lambda *args: all(args[i] < args[i + 1] for i in range(len(args) - 1)))
register_primitive(">", lambda *args: all(args[i] > args[i + 1] for i in range(len(args) - 1)))
register_primitive("<=", lambda *args: all(args[i] <= args[i + 1] for i in range(len(args) - 1)))
register_primitive(">=", lambda *args: all(args[i] >= args[i + 1] for i in range(len(args) - 1)))
register_primitive("not", lambda x: not x)
register_primitive("and", lambda *args: all(args))
register_primitive("or", lambda *args: any(args))

register_primitive("list", lambda *args: list(args))
register_primitive("pair", lambda a, b: [a, b])
register_primitive("first", lambda x: x[0])
register_primitive("rest", lambda x: x[1:])
register_primitive("cons", lambda x, y: [x] + (y if isinstance(y, list) else [y]))
register_primitive("append", lambda *args: sum((arg if isinstance(arg, list) else [arg] for arg in args), []))
register_primitive("length", lambda x: len(x))
register_primitive("null?", lambda x: len(x) == 0 if isinstance(x, list) else False)


# Initialize the global environment with primitives
def create_global_env():
    """Create and return the global environment."""
    env = {}
    for name, (func, dist_func) in _PRIMITIVES.items():
        if dist_func:
            # Elementary random procedure
            env[str(name)] = CathedralProcedure(None, [], {}, dist_func)
        else:
            # Regular primitive
            env[str(name)] = func
    return env
