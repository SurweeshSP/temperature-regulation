"""
Phase 2 EntropyTarget — thin re-export of Phase 1 EntropyTarget so Phase 2
scripts can import from mctr.control consistently.
"""
from mctr.controller.controller import EntropyTarget

Phase2EntropyTarget = EntropyTarget

__all__ = ["Phase2EntropyTarget", "EntropyTarget"]
