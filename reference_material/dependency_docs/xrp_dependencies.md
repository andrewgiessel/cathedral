# XRPs (Exchangeable Random Procedures) Dependencies

This document describes the dependencies and structure of the XRPs subsystem in MIT Church.

## Main Files

- `church/xrp-lib/CRP.ss`: Chinese Restaurant Process
- `church/xrp-lib/beta-binomial.ss`: Beta-Binomial distribution
- `church/xrp-lib/dirichlet-discrete.ss`: Dirichlet-Discrete distribution
- `church/xrp-lib/dirichlet-discrete-nocache.ss`: Dirichlet-Discrete (no cache)
- `church/xrp-lib/gensym-xrp.ss`: Gensym XRP
- `church/xrp-lib/ms-dirichlet-discrete.ss`: Multi-sample Dirichlet-Discrete
- `church/xrp-lib/normal-normal-gamma.ss`: Normal-Normal-Gamma model

## Key Relationships

- Each XRP implements a probabilistic primitive for use in Church programs.
- Imports utility modules and standard environment.

## Next Steps

- Port XRPs to Hy, leveraging Python's statistical libraries where possible.
- Ensure correct probabilistic semantics.
