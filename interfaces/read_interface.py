"""
Read Interface - Read-only access to system state.

This is the ONLY way for LLM agents to observe state.
All reads are bounded and logged.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ReadInterface(ABC):
    """
    Read-only interface for state observation.
    
    LLM agents can ONLY interact with state through this interface.
    The broker ensures all reads are bounded and audited.
    """
    
    @abstractmethod
    def get_observable_signals(self, agent_id: str) -> Dict[str, Any]:
        """
        Get observable signals for an agent.
        
        Returns:
            Dict with ONLY observable fields.
            Hidden state MUST NOT be included.
        """
        pass
    
    @abstractmethod
    def get_environment_signals(self) -> Dict[str, Any]:
        """
        Get environment-level signals.
        
        Returns:
            Dict with observable environment state.
        """
        pass
    
    @abstractmethod
    def get_memory(self, agent_id: str, window_size: int = 5) -> List[str]:
        """
        Get agent memory (bounded by window).
        
        Args:
            agent_id: Agent identifier
            window_size: Maximum memory entries to return
            
        Returns:
            List of memory entries (most recent first)
        """
        pass


class SimpleReadInterface(ReadInterface):
    """Simple implementation for basic domains."""
    
    def __init__(self, state_provider: Any, observable_fields: List[str]):
        self.state_provider = state_provider
        self.observable_fields = observable_fields
    
    def get_observable_signals(self, agent_id: str) -> Dict[str, Any]:
        state = self.state_provider.get_agent_state(agent_id)
        return {k: v for k, v in state.items() if k in self.observable_fields}
    
    def get_environment_signals(self) -> Dict[str, Any]:
        return self.state_provider.get_environment_state()
    
    def get_memory(self, agent_id: str, window_size: int = 5) -> List[str]:
        state = self.state_provider.get_agent_state(agent_id)
        memory = state.get("memory", [])
        return memory[-window_size:]
