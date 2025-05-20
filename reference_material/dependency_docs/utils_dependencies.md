# Utils Dependencies

This document describes the dependencies and structure of the utility modules in MIT Church.

## Main Files

- `church/utils/rnrs.ss`: Compatibility and re-exports from R6RS
- `church/utils/utils.mzscheme.ss`: General utility functions
- `church/utils/mega-comparator.ss`: Data comparison utilities
- `church/utils/serializer.mzscheme.ss`: Serialization/deserialization helpers
- `church/utils/AD.ss`: Automatic differentiation utilities

## Key Relationships

- Used throughout the codebase for compatibility, data handling, and utility operations.
- Many utility modules import each other and standard libraries.

## Next Steps

- Port utility modules to Hy, mapping Scheme idioms to Python/Hy idioms.
- Ensure compatibility with Hy's and Python's standard libraries.
