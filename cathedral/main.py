import argparse
import sys

from cathedral.core.eval import create_global_env, eval_church
from cathedral.core.query import register_query_primitives
from cathedral.primitives.memoization import register_memoization_primitives
from cathedral.primitives.random import register_random_primitives


def initialize_environment():
    """Initialize Cathedral environment with all primitives."""
    # Create global environment
    env = create_global_env()

    # Register all primitives
    register_random_primitives()
    register_memoization_primitives()
    register_query_primitives()

    return env


def read_cathedral_file(filename):
    """Read a Cathedral program from a file."""
    with open(filename) as f:
        return f.read()


def cathedral_repl(env=None):
    """Run an interactive Cathedral REPL."""
    if env is None:
        env = initialize_environment()

    print("Cathedral REPL. Press Ctrl+D to exit.")

    while True:
        try:
            expr = input("cathedral> ")
            if not expr.strip():
                continue
            result = eval_church(expr, env)
            print(result)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
        except EOFError:
            print("\nExiting Cathedral REPL")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point for the Cathedral CLI."""
    parser = argparse.ArgumentParser(description="Cathedral probabilistic programming language")
    parser.add_argument("file", nargs="?", help="Cathedral file to execute")
    parser.add_argument("-e", "--expression", help="Cathedral expression to evaluate")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive REPL after executing file")
    args = parser.parse_args()

    # Initialize environment
    env = initialize_environment()

    # Execute file if provided
    if args.file:
        try:
            program = read_cathedral_file(args.file)
            result = eval_church(program, env)
            print(result)
        except Exception as e:
            print(f"Error executing file: {e}")
            return 1

    # Evaluate expression if provided
    if args.expression:
        try:
            result = eval_church(args.expression, env)
            print(result)
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return 1

    # Start REPL if requested or if no file or expression provided
    if args.interactive or (not args.file and not args.expression):
        cathedral_repl(env)

    return 0


if __name__ == "__main__":
    sys.exit(main())
