"""
Base Simulation Engine - Minimal interface for simulation classes.
"""
from typing import Dict, Any, List

class BaseSimulationEngine:
    """Base class for simulation engines."""
    
    def get_agents(self) -> List[Any]:
        """Return list of agents in the simulation."""
        return []
    
    def step(self, year: int) -> None:
        """Execute one simulation step."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Return current simulation state."""
        return {}

