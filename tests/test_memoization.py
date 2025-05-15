from cathedral.core.eval import eval_church
from cathedral.primitives.memoization import register_memoization_primitives
from cathedral.primitives.random import register_random_primitives

# Register primitives before testing
register_random_primitives()
register_memoization_primitives()


def test_mem_basic():
    """Test basic memoization behavior."""
    # Define a memoized function that returns a random value
    church_program = """
    (define f (mem (lambda (x) (random-integer 1000))))
    (= (f 1) (f 1))
    """
    # The function should return the same value when called with the same argument
    assert eval_church(church_program, {}) == True

    # Different arguments should (probably) yield different values
    church_program = """
    (define f (mem (lambda (x) (random-integer 1000))))
    (list (f 1) (f 2))
    """
    result = eval_church(church_program, {})
    # Small chance they're equal, but very unlikely
    assert len(set(result)) > 1


def test_mem_persistence():
    """Test that memoized values persist across evaluations."""
    # Create an environment and define a memoized function in it
    env = {}
    eval_church("(define f (mem (lambda (x) (random-integer 1000))))", env)

    # Get the value for f(1)
    val1 = eval_church("(f 1)", env)

    # Call it again, should get the same value
    val2 = eval_church("(f 1)", env)
    assert val1 == val2

    # Call with a different argument, should get a different value
    val3 = eval_church("(f 2)", env)
    assert val1 != val3  # Again, small chance they're equal but unlikely


def test_nested_mem():
    """Test nested memoization."""
    church_program = """
    (define f (mem (lambda (x) (mem (lambda (y) (random-integer 1000))))))
    (define g1 (f 1))
    (define g2 (f 2))
    (list (= (g1 3) (g1 3))   ; Should be True - same inner function, same arg
          (= (g2 3) (g2 3))   ; Should be True - same inner function, same arg
          (= (g1 3) (g2 3)))  ; Should be False - different inner functions
    """
    result = eval_church(church_program, {})
    assert result[0] == True
    assert result[1] == True
    # The third one should usually be False, but there's a small chance they're equal
    # So we don't assert on it


def test_recursive_mem():
    """Test recursive memoized function."""
    # Define a recursive memoized function (simple recursive Fibonacci)
    church_program = """
    (define fib 
      (mem (lambda (n)
             (if (< n 2)
                 n
                 (+ (fib (- n 1)) (fib (- n 2)))))))
    (fib 10)
    """
    # Should compute efficiently due to memoization
    result = eval_church(church_program, {})
    assert result == 55


def test_dpmem_basic():
    """Test basic DPmem behavior."""
    # DPmem with alpha=0 should behave like regular mem
    church_program = """
    (define f (DPmem 0 (lambda (x) (random-integer 1000))))
    (= (f 1) (f 1))
    """
    assert eval_church(church_program, {}) == True

    # DPmem with very high alpha should behave like no memoization
    # We use alpha=1000 here, which is high enough for testing
    church_program = """
    (define f (DPmem 1000 (lambda (x) (random-integer 10))))
    ; Call f(1) multiple times and count unique values
    (define results (list (f 1) (f 1) (f 1) (f 1) (f 1)))
    (> (length (unique results)) 1)  ; Should have more than one unique value
    """
    # Define unique function for this test
    env = {}
    eval_church(
        "(define unique (lambda (lst) (if (null? lst) '() (cons (first lst) (filter (lambda (x) (not (= x (first lst)))) (unique (rest lst)))))))",
        env,
    )

    assert eval_church(church_program, env) == True
