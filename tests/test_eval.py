import pytest
import hy.models

from cathedral.core.eval import eval_church, create_global_env
from cathedral.primitives.random import register_random_primitives


@pytest.fixture
def global_env():
    """Create a global environment with all primitives registered."""
    env = create_global_env()
    register_random_primitives()
    return env


def test_eval_constants(global_env):
    """Test evaluation of constants."""
    assert eval_church("42", global_env) == 42
    assert eval_church("True", global_env) == True
    assert eval_church("False", global_env) == False
    assert eval_church("nil", global_env) == None


def test_eval_variable_lookup(global_env):
    """Test evaluation of variables from the environment."""
    env = global_env.copy()
    env["x"] = 42
    env["y"] = "hello"
    assert eval_church("x", env) == 42
    assert eval_church("y", env) == "hello"

    # Variable not in environment should raise an error
    with pytest.raises(Exception):
        eval_church("z", env)


def test_eval_if_expressions(global_env):
    """Test evaluation of if expressions."""
    assert eval_church("(if True 1 2)", global_env) == 1
    assert eval_church("(if False 1 2)", global_env) == 2

    # The condition should be evaluated
    result = eval_church('(if (= 1 1) "yes" "no")', global_env)
    assert result == "yes"
    
    result = eval_church('(if (= 1 2) "yes" "no")', global_env)
    assert result == "no"


def test_eval_lambda_expressions(global_env):
    """Test evaluation of lambda expressions."""
    # Lambda should return a procedure
    proc = eval_church("(lambda (x) x)", global_env)
    assert callable(proc)

    # Lambda should capture its environment
    env = global_env.copy()
    env["y"] = 10
    proc = eval_church("(lambda (x) (+ x y))", env)
    assert proc(5) == 15


def test_eval_procedure_application(global_env):
    """Test evaluation of procedure applications."""
    # Simple application
    assert eval_church("((lambda (x) (+ x 1)) 41)", global_env) == 42

    # Nested applications
    assert eval_church("((lambda (x) ((lambda (y) (+ x y)) 2)) 3)", global_env) == 5

    # Application with multiple arguments
    assert eval_church("((lambda (x y) (+ x y)) 3 4)", global_env) == 7


def test_eval_define(global_env):
    """Test evaluation of define expressions."""
    env = global_env.copy()
    result = eval_church("(define x 42)", env)
    assert result == env
    assert env["x"] == 42

    # Define should allow using previous definitions
    eval_church("(define y (+ x 10))", env)
    assert env["y"] == 52


def test_eval_quote(global_env):
    """Test evaluation of quote expressions."""
    # For quote, we expect to get back the actual Hy model object, not a string
    quoted_42 = eval_church("'42", global_env)
    assert isinstance(quoted_42, hy.models.Integer)
    
    quoted_list = eval_church("'(1 2 3)", global_env)
    assert isinstance(quoted_list, hy.models.Expression)
    
    quoted_expr = eval_church("'(+ 1 2)", global_env)
    assert isinstance(quoted_expr, hy.models.Expression)

    # quote should prevent evaluation
    quoted_if = eval_church("'(if True 1 2)", global_env)
    assert isinstance(quoted_if, hy.models.Expression)
