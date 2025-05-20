# Church Core Dependencies

This document describes the core dependencies of the MIT Church codebase, focusing on the foundational modules and their relationships. This is the starting point for porting to Hy.

## Core Files

- `church.ss` (root): Entry point, imports `(church church)`
- `church/church.ss`: Main implementation, imports:
  - `church standard-env`
  - `church utils rnrs`
  - `church register-primitives`
  - `church readable-scheme`
  - `church standard-env`
  - `church church-eval/church-eval`
  - `church mcmc/mcmc-core`
  - `church constraint-propagation/constraints`
  - `church adis/bounds`
  - `church xrp-lib/CRP`

## Key Relationships

- **Standard Environment**: Provides core language features and standard library functions.
- **Utils**: Utility functions and compatibility layers.
- **Register Primitives**: Mechanism for registering built-in functions and ERPs.
- **Readable Scheme**: Syntax and utility helpers for more readable code.
- **Church Eval**: Core interpreter and evaluation logic.
- **MCMC Core**: Probabilistic inference engine.
- **Constraint Propagation**: Logic for constraint-based inference.
- **XRPs**: Exchangeable Random Procedures (e.g., CRP, Dirichlet, Beta-Binomial).

## Next Steps

- Use this as a reference for porting the core logic to Hy.
- See related docs for details on each subsystem.
