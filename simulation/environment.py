"""
Tiered Environment for Scientific World Modeling.

Distinguishes between:
1. Global State (e.g., Inflation, Sea Level)
2. Regional/Local State (e.g., Tract Paving Density, Market Sector Demand)
3. Institutional State (e.g., Government Budget)

Serves as the single source of truth for "Non-Personal" state.
"""
from typing import Dict, Any, Optional, List

class TieredEnvironment:
    def __init__(self, global_state: Optional[Dict[str, Any]] = None):
        # Layer 1: Global (Sim-wide)
        self.global_state: Dict[str, Any] = global_state or {}
        
        # Layer 2: Spatial/Local (Tracts, Neighborhoods)
        self.local_states: Dict[str, Dict[str, Any]] = {}
        
        # Layer 3: Institutional (Government, Companies)
        self.institutions: Dict[str, Dict[str, Any]] = {}

    def set_global(self, key: str, value: Any):
        """Set a global variable."""
        self.global_state[key] = value

    def set_local(self, location_id: str, key: str, value: Any):
        """Set a variable for a specific location (tract)."""
        if location_id not in self.local_states:
            self.local_states[location_id] = {}
        self.local_states[location_id][key] = value

    def get_observable(self, path: str, default: Any = None) -> Any:
        """
        Safe retrieval using dot-notation path.
        
        Examples:
        - "global.inflation"
        - "local.T001.paving_density"
        - "institutions.fema.budget"
        """
        parts = path.split('.')
        
        if not parts:
            return default

        scope = parts[0]
        
        try:
            if scope == "global":
                if len(parts) == 2:
                    return self.global_state.get(parts[1], default)
                    
            elif scope == "local":
                if len(parts) == 3:
                    loc_id, key = parts[1], parts[2]
                    return self.local_states.get(loc_id, {}).get(key, default)
                    
            elif scope == "institutions":
                 if len(parts) == 3:
                    inst_id, key = parts[1], parts[2]
                    return self.institutions.get(inst_id, {}).get(key, default)
                    
        except Exception:
            return default
            
        return default

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for audit/logging."""
        return {
            "global": self.global_state,
            "local": self.local_states,
            "institutions": self.institutions
        }
