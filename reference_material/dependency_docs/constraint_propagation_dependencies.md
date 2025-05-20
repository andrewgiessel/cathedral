# Constraint Propagation Dependencies

This document describes the dependencies and structure of the constraint propagation subsystem in MIT Church.

## Main Files

- `church/constraint-propagation/constraints.ss`: Core constraint logic
- `church/constraint-propagation/primitive-inverses.ss`: Inverse operations for primitives
- `church/constraint-propagation/trace-constraint-propagation.ss`: Trace-based constraint propagation

## Key Relationships

- Imports utility modules, standard environment, and core Church logic.
- Used for constraint-based probabilistic inference.

## Next Steps

- Port constraint propagation logic to Hy, mapping Scheme constraint logic to Python/Hy idioms.
