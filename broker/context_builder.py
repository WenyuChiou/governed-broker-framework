"""
Context Builder - Builds bounded context for LLM.

Responsibilities:
- Fetch observable signals (READ-ONLY)
- Format prompt with bounded information
- Enforce information boundaries
"""
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class ContextBuilder(ABC):
    """
    Abstract base class for building LLM context.
    
    Subclass this for domain-specific context building.
    """
    
    @abstractmethod
    def build(self, agent_id: str) -> Dict[str, Any]:
        """
        Build bounded context for an agent.
        
        Returns:
            Dict with observable signals only.
            Must NOT include hidden state variables.
        """
        pass
    
    @abstractmethod
    def format_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format context into LLM prompt.
        
        Args:
            context: Bounded context from build()
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def get_memory(self, agent_id: str) -> List[str]:
        """
        Get current memory state for agent.
        
        Used for audit trail (memory_post).
        """
        pass


class SimpleContextBuilder(ContextBuilder):
    """
    Simple implementation for toy domains.
    """
    
    def __init__(
        self,
        state_provider: Any,
        prompt_template: str,
        observable_fields: List[str]
    ):
        self.state_provider = state_provider
        self.prompt_template = prompt_template
        self.observable_fields = observable_fields
    
    def build(self, agent_id: str) -> Dict[str, Any]:
        """Build context from observable fields only."""
        full_state = self.state_provider.get_agent_state(agent_id)
        
        # Filter to observable fields only
        context = {
            k: v for k, v in full_state.items() 
            if k in self.observable_fields
        }
        context["agent_id"] = agent_id
        
        return context
    
    def format_prompt(self, context: Dict[str, Any]) -> str:
        """Format using template."""
        return self.prompt_template.format(**context)
    
    def get_memory(self, agent_id: str) -> List[str]:
        """Get agent memory."""
        state = self.state_provider.get_agent_state(agent_id)
        return state.get("memory", [])
