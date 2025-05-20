# SRFI Dependencies

This document describes the dependencies and structure of the SRFI (Scheme Requests for Implementation) modules used in MIT Church.

## Main SRFIs Used

- SRFI-1: List library
- SRFI-6: Basic string ports
- SRFI-8: Receive multiple values
- SRFI-9: Defining record types
- SRFI-19: Time data types and procedures
- SRFI-23: Error reporting mechanism
- SRFI-27: Sources of random bits
- SRFI-39: Parameter objects
- SRFI-43: Vector library
- SRFI-69: Basic hash tables

## Structure

- Implemented in `mit-church/_srfi/` directory
- Many have compatibility layers for different Scheme implementations (e.g., `compat.mzscheme.sls`)
- Some SRFIs import each other or standard R6RS libraries

## Next Steps

- Port required SRFI functionality to Hy, using Python/Hy built-ins or third-party libraries where possible.
- Only port the SRFIs actually used by Church code.
