"""
Phase 2 Temperature Controller.

Re-uses the Phase 1 MCTRController as the temperature head.
This module exists so Phase 2 components can import from mctr.control
without coupling to the Phase 1 controller location.
"""
from mctr.controller.controller import MCTRController, EntropyTarget, ControlState

# Alias for clarity inside Phase 2 code
Phase2TemperatureController = MCTRController

__all__ = ["Phase2TemperatureController", "EntropyTarget", "ControlState"]
