# MCMC Dependencies

This document describes the dependencies and structure of the Markov Chain Monte Carlo (MCMC) subsystem in MIT Church.

## Main Files

- `church/mcmc/mcmc-core.ss`: Core MCMC logic
- `church/mcmc/queries/base-queries.ss`: Base query types
- `church/mcmc/queries/mh-query.ss`: Metropolis-Hastings
- `church/mcmc/queries/mtm-query.ss`: Multiple-Try Metropolis
- `church/mcmc/queries/annealed-importance.ss`: Annealed Importance Sampling
- `church/mcmc/queries/gradient-mh-query.ss`: Gradient-based MH
- `church/mcmc/queries/tempered-transitions.ss`: Tempered transitions
- `church/mcmc/queries/emc-games.ss`: Evolutionary Monte Carlo
- `church/mcmc/queries/temperature-games.ss`: Parallel tempering, etc.

## Key Relationships

- Imports utility modules, standard environment, and core Church evaluation logic.
- Each query type may have its own sub-dependencies.

## Next Steps

- Port MCMC core and queries to Hy, leveraging Python's scientific libraries where possible.
- Ensure probabilistic inference logic is preserved.
