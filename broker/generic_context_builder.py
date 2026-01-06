"""
Generic Context Builder - Builds bounded context for any agent type.

Works with BaseAgent to provide 0-1 normalized state in prompts.
Supports multi-agent scenarios where agents observe each other.
"""
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod


class ContextBuilder(ABC):
    """
    Abstract base class for building LLM context.
    
    All values in context should be 0-1 normalized when possible.
    """
    
    @abstractmethod
    def build(
        self, 
        agent_id: str,
        observable: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build bounded context for an agent.
        
        Args:
            agent_id: The agent to build context for
            observable: Optional list of observable categories
        
        Returns:
            Dict with observable signals only (0-1 normalized where applicable).
        """
        pass
    
    @abstractmethod
    def format_prompt(self, context: Dict[str, Any]) -> str:
        """Format context into LLM prompt."""
        pass


class BaseAgentContextBuilder(ContextBuilder):
    """
    Context builder that works with BaseAgent instances.
    
    Features:
    - Automatically uses 0-1 normalized state
    - Supports multi-agent observation
    - Works with any user-defined agent type
    """
    
    def __init__(
        self,
        agents: Dict[str, Any],  # Dict[str, BaseAgent]
        environment: Dict[str, float] = None,
        prompt_templates: Dict[str, str] = None
    ):
        """
        Args:
            agents: Dict mapping agent names to BaseAgent instances
            environment: Shared environment state (0-1 normalized)
            prompt_templates: Dict mapping agent_type to prompt template
        """
        self.agents = agents
        self.environment = environment or {}
        self.prompt_templates = prompt_templates or {}
    
    def build(
        self, 
        agent_id: str,
        observable: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Build context from agent's normalized state and perception."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"agent_id": agent_id, "error": "Agent not found"}
        
        # Get agent's normalized state
        context = {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "agent_type": agent.agent_type,
            "state": agent.get_all_state(),  # Already 0-1 normalized
        }
        
        # Add perception from environment and other agents
        perception = agent.observe(self.environment, self.agents)
        context["perception"] = perception
        
        # Add objectives evaluation
        context["objectives"] = agent.evaluate_objectives()
        
        # Add available skills
        context["available_skills"] = agent.get_available_skills()
        
        return context
    
    def format_prompt(self, context: Dict[str, Any]) -> str:
        """Format using agent-type-specific template."""
        agent_type = context.get("agent_type", "default")
        template = self.prompt_templates.get(agent_type, DEFAULT_PROMPT_TEMPLATE)
        
        # Build formatted sections
        state_str = self._format_state(context.get("state", {}))
        perception_str = self._format_perception(context.get("perception", {}))
        objectives_str = self._format_objectives(context.get("objectives", {}))
        skills_str = ", ".join(context.get("available_skills", []))
        
        return template.format(
            agent_name=context.get("agent_name", "Agent"),
            agent_type=agent_type,
            state=state_str,
            perception=perception_str,
            objectives=objectives_str,
            skills=skills_str
        )
    
    def _format_state(self, state: Dict[str, float]) -> str:
        """Format normalized state for prompt."""
        lines = []
        for param, value in state.items():
            # Show as percentage or decimal
            lines.append(f"- {param}: {value:.2f}")
        return "\n".join(lines) if lines else "No state"
    
    def _format_perception(self, perception: Dict[str, float]) -> str:
        """Format perception signals."""
        lines = []
        for signal, value in perception.items():
            lines.append(f"- {signal}: {value:.2f}")
        return "\n".join(lines) if lines else "No external signals"
    
    def _format_objectives(self, objectives: Dict[str, Dict]) -> str:
        """Format objectives with status."""
        lines = []
        for name, info in objectives.items():
            status = "✓" if info.get("in_range") else "✗"
            current = info.get("current", 0)
            target = info.get("target", (0, 1))
            lines.append(f"- {name}: {current:.2f} (target: {target[0]:.2f}-{target[1]:.2f}) {status}")
        return "\n".join(lines) if lines else "No objectives defined"


# Default prompt template for any agent type
DEFAULT_PROMPT_TEMPLATE = """You are {agent_name}, a {agent_type} agent.

=== CURRENT STATE (0-1 scale) ===
{state}

=== OBSERVATIONS ===
{perception}

=== OBJECTIVES ===
{objectives}

=== AVAILABLE ACTIONS ===
{skills}

=== YOUR TASK ===
Based on your current state and objectives, decide your next action.

=== OUTPUT FORMAT ===
Decision: [one of the available actions]
Adjustment: [0.00-0.15 if applicable]
Justification: [brief explanation]
"""


# Convenience function to create context builder from agents
def create_context_builder(
    agents: Dict[str, Any],
    environment: Dict[str, float] = None,
    custom_templates: Dict[str, str] = None
) -> BaseAgentContextBuilder:
    """
    Create a context builder for a set of agents.
    
    Args:
        agents: Dict of agent_name -> BaseAgent instances
        environment: Shared environment state
        custom_templates: Optional custom prompt templates per agent_type
    
    Returns:
        Configured BaseAgentContextBuilder
    """
    return BaseAgentContextBuilder(
        agents=agents,
        environment=environment,
        prompt_templates=custom_templates or {}
    )
