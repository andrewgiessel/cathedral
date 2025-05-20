# Standard Environment Dependencies

This document describes the dependencies and structure of the Standard Environment subsystem in MIT Church.

## Main File

- `church/standard-env.ss`: Provides core language features, standard library functions, and imports many SRFIs and utility modules.

## Key Imports

- SRFI modules (e.g., SRFI-1, SRFI-6, SRFI-8, SRFI-9, SRFI-19, SRFI-23, SRFI-27, SRFI-39, SRFI-43, SRFI-69)
- `church/utils/rnrs.ss`: Compatibility and utility functions
- `church/utils/utils.mzscheme.ss`: Additional utilities
- `church/utils/mega-comparator.ss`: Data comparison utilities
- `church/utils/serializer.mzscheme.ss`: Serialization helpers

## SRFI Dependencies

- Many SRFIs are implemented in the `_srfi/` directory, often with compatibility layers for different Scheme implementations.
- Each SRFI may have its own sub-dependencies (see SRFI doc for details).

## Next Steps

- Port the standard environment and its SRFI dependencies to Hy, ensuring compatibility with Hy's standard library and Python ecosystem.
- See the SRFI dependency doc for more details on each SRFI.
