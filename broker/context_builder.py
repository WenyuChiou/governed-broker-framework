"""
Generic Context Builder - Builds bounded context for any agent type.

Works with BaseAgent to provide 0-1 normalized state in prompts.
Supports multi-agent scenarios where agents observe each other.
"""
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from .memory_engine import MemoryEngine


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
        prompt_templates: Dict[str, str] = None,
        memory_engine: Optional[MemoryEngine] = None
    ):
        """
        Args:
            agents: Dict mapping agent names to BaseAgent instances
            environment: Shared environment state (0-1 normalized)
            prompt_templates: Dict mapping agent_type to prompt template
            memory_engine: Engine for retrieving agent memory
        """
        self.agents = agents
        self.environment = environment or {}
        self.prompt_templates = prompt_templates or {}
        self.memory_engine = memory_engine
    
    def build(
        self, 
        agent_id: str,
        observable: Optional[List[str]] = None,
        include_memory: bool = True,
        include_raw: bool = False
    ) -> Dict[str, Any]:
        """Build context from agent's normalized state and perception.
        
        Args:
            agent_id: Agent to build context for
            observable: Optional categories to include
            include_memory: Include formatted memory
            include_raw: Include raw (denormalized) values alongside 0-1
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return {"agent_id": agent_id, "error": "Agent not found"}
        
        # Get agent's normalized state
        context = {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "agent_type": agent.agent_type,
            "state": agent.get_all_state(),  # 0-1 normalized
        }
        
        # Optionally add raw values for readability
        if include_raw:
            context["state_raw"] = agent.get_all_state_raw()
        
        # Add perception from environment and other agents
        perception = agent.observe(self.environment, self.agents)
        context["perception"] = perception
        
        # Add objectives evaluation
        context["objectives"] = agent.evaluate_objectives()
        
        # Add available skills
        context["available_skills"] = agent.get_available_skills()
        
        # Add memory if available
        if include_memory:
            context["memory"] = self._get_memory(agent)
        
        # Add neighbor summary if observable
        observable = observable or []
        if "neighbors" in observable:
            context["neighbors"] = self._get_neighbor_summary(agent_id)
            
        # Add flood-specific attributes to context if present on agent
        for attr in ['elevated', 'has_insurance', 'relocated', 'flood_threshold', 'trust_in_insurance', 'trust_in_neighbors', 'flood']:
            if hasattr(agent, attr):
                context[attr] = getattr(agent, attr)
                
        # Add custom attributes if present
        if hasattr(agent, 'custom_attributes') and agent.custom_attributes:
            context.update(agent.custom_attributes)
        
        return context
    
    def _get_memory(self, agent) -> List[str]:
        """Get formatted memory via MemoryEngine."""
        if self.memory_engine:
            return self.memory_engine.retrieve(agent, top_k=3)
        
        # Fallback to direct attribute if engine not set (for standalone tests)
        if hasattr(agent, 'memory') and isinstance(agent.memory, list):
            return agent.memory[-3:]
            
        return []
    
    def _get_neighbor_summary(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get summary of neighbor agents' observable state."""
        summaries = []
        for name, agent in self.agents.items():
            if name != agent_id:
                summaries.append({
                    "agent_name": name,
                    "agent_type": agent.agent_type,
                    "state_summary": {
                        k: round(v, 2) for k, v in list(agent.get_all_state().items())[:3]
                    }
                })
        return summaries[:5]  # Limit to 5 neighbors
    
    def format_prompt(self, context: Dict[str, Any]) -> str:
        """Format using agent-type-specific template."""
        agent_type = context.get("agent_type", "default")
        template = self.prompt_templates.get(agent_type, DEFAULT_PROMPT_TEMPLATE)
        
        # Build formatted sections
        state = context.get("state", {})
        state_str = self._format_state(state)
        perception_str = self._format_perception(context.get("perception", {}))
        objectives_str = self._format_objectives(context.get("objectives", {}))
        skills_str = ", ".join(context.get("available_skills", []))
        memory_val = context.get("memory", [])
        if isinstance(memory_val, list):
            # Baseline uses: "- {item}" prefix for each memory line
            memory_str = "\n".join(f"- {m}" for m in memory_val) if memory_val else "No memory available"
        else:
            memory_str = str(memory_val)
        
        # Prepare template variables
        # Base variables
        template_vars = {
            "agent_name": context.get("agent_name", "Agent"),
            "agent_type": agent_type,
            "state": state_str,
            "perception": perception_str,
            "objectives": objectives_str,
            "skills": skills_str,
            "memory": memory_str
        }
        
        # Add individual state variables (for custom templates)
        # e.g. {loss_ratio}
        template_vars.update(state)
        
        # Add raw state variables if present (prioritize over normalized?)
        # Usually we want normalized in prompt, but maybe raw for specific metrics
        if "state_raw" in context:
            template_vars.update(context["state_raw"])
            
        # Add perception variables
        if "perception" in context:
            template_vars.update(context["perception"])
            
        # Add extra context keys (like 'budget_remaining' if passed in context root)
        template_vars.update({k: v for k, v in context.items() 
                            if k not in template_vars and isinstance(v, (str, int, float))})
            
        return template.format(**template_vars)
    
    def _format_state(self, state: Dict[str, float], compact: bool = True) -> str:
        """Format normalized state. Compact: inline, Verbose: multiline."""
        if compact:
            # Add semantic labels: LOW(<0.3), MED(0.3-0.7), HIGH(>0.7)
            # Filter distinct numeric values only
            return " ".join(
                f"{k}={v:.2f}({self._semantic(v)})" 
                for k, v in state.items() 
                if isinstance(v, (int, float))
            )
        return "\n".join(f"- {k}: {v:.2f}" for k, v in state.items()) or "No state"
    
    def _semantic(self, v: float) -> str:
        """Convert 0-1 value to semantic label."""
        if v < 0.3: return "L"
        if v > 0.7: return "H"
        return "M"
    
    def _format_perception(self, perception: Dict[str, float], compact: bool = True) -> str:
        """Format perception signals."""
        if compact:
            return " ".join(f"{k}={v:.2f}" for k, v in perception.items()) or "-"
        return "\n".join(f"- {k}: {v:.2f}" for k, v in perception.items()) or "No signals"
    
    def _format_objectives(self, objectives: Dict[str, Dict], compact: bool = True) -> str:
        """Format objectives with status."""
        if compact:
            # Show: name=current(status) where status is ✓=in_range, ✗=out
            return " ".join(
                f"{n}={info.get('current',0):.2f}{'✓' if info.get('in_range') else '✗'}"
                for n, info in objectives.items()
            ) or "-"
        # Verbose format
        lines = []
        for name, info in objectives.items():
            status = "✓" if info.get("in_range") else "✗"
            current = info.get("current", 0)
            target = info.get("target", (0, 1))
            lines.append(f"- {name}: {current:.2f} ({target[0]:.2f}-{target[1]:.2f}) {status}")
        return "\n".join(lines) if lines else "No objectives"

from .interaction_hub import InteractionHub

class TieredContextBuilder(BaseAgentContextBuilder):
    """
    PR 2: Hierarchical Context Builder (T0, T1, T2).
    
    Tiers:
    - Tier 0: Personal (Internal state/memory)
    - Tier 1: Local (Social Gossip & Spatial Observation)
    - Tier 2: Global (Institutional/Public News)
    """
    def __init__(
        self,
        agents: Dict[str, Any],
        hub: InteractionHub,
        global_news: List[str] = None,
        prompt_templates: Dict[str, str] = None,
        memory_engine: Optional[MemoryEngine] = None
    ):
        super().__init__(agents=agents, prompt_templates=prompt_templates, memory_engine=memory_engine)
        self.hub = hub
        self.global_news = global_news or []

    def build(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """Build context using the InteractionHub's tiered logic."""
        return self.hub.build_tiered_context(agent_id, self.agents, self.global_news)

    def format_prompt(self, context: Dict[str, Any]) -> str:
        """Format the tiered context into a grounded prompt."""
        agent_type = context['personal'].get('agent_type', 'household')
        # Use a tiered template if available, otherwise use a generic structured one
        template = self.prompt_templates.get(agent_type, "{personal_section}\n\n{local_section}\n\n{global_section}\n\n{options_text}")
        
        # 1. Personal Section (T0)
        p = context['personal']
        mem_items = p.get('memory', [])
        mem_str = "\n".join([f"- {m}" for m in mem_items]) if mem_items else "No private memories."
        personal_section = f"### [MY STATUS & HISTORY]\n{mem_str}"
        
        # 2. Local Section (T1)
        l = context['local']
        spatial = l['spatial']
        spatial_str = f"- Neighborhood Observation: {spatial['elevated_pct']}% of my direct neighbors have elevated homes."
        
        social = l['social']
        social_str = "\n".join([f"- {g}" for g in social]) if social else "- No recent gossip from neighbors."
        local_section = f"### [LOCAL NEIGHBORHOOD]\n{spatial_str}\n{social_str}"
        
        # 3. Global Section (T2)
        g = context['global']
        global_str = "\n".join([f"- {news}" for news in g]) if g else "- No major public news today."
        global_section = f"### [CITY-WIDE NEWS]\n{global_str}"
        
        # Combine
        return template.format(
            personal_section=personal_section,
            local_section=local_section,
            global_section=global_section,
            options_text="[Now, consider your available options and make a decision.]"
        )


# Compact token-efficient prompt template with LLM interpretation
DEFAULT_PROMPT_TEMPLATE = """[{agent_type}:{agent_name}]

STATE:{state}
OBS:{perception}
MEM:{memory}
OBJ:{objectives}
ACT:{skills}

OUTPUT:
INTERPRET:[1-line summary of your situation]
DECIDE:[action] ADJ:[0-0.15] REASON:[1-line]"""


# Verbose template for debugging/analysis
VERBOSE_PROMPT_TEMPLATE = """You are {agent_name}, a {agent_type} agent.

=== STATE (0-1) ===
{state}

=== OBSERVATIONS ===
{perception}

=== MEMORY ===
{memory}

=== OBJECTIVES ===
{objectives}

=== ACTIONS ===
{skills}

DECIDE: action, adjustment(0.00-0.15), justification
"""


# Convenience function to create context builder from agents
def create_context_builder(
    agents: Dict[str, Any],
    environment: Dict[str, float] = None,
    custom_templates: Dict[str, str] = None,
    load_yaml: bool = True,
    yaml_path: str = None,
    memory_engine: Optional[MemoryEngine] = None
) -> BaseAgentContextBuilder:
    """
    Create a context builder for a set of agents.
    
    Args:
        agents: Dict of agent_name -> BaseAgent instances
        environment: Shared environment state
        custom_templates: Optional custom prompt templates per agent_type
        load_yaml: If True, load templates from YAML file
        yaml_path: Optional path to YAML file (default: broker/prompt_templates.yaml)
        memory_engine: Engine for retrieving agent memory
    
    Returns:
        Configured BaseAgentContextBuilder
    """
    templates = {}
    
    # Load from YAML if requested
    if load_yaml:
        templates = load_prompt_templates(yaml_path)
    
    # Override with custom templates
    if custom_templates:
        templates.update(custom_templates)
    
    return BaseAgentContextBuilder(
        agents=agents,
        environment=environment,
        prompt_templates=templates,
        memory_engine=memory_engine
    )


def load_prompt_templates(
    yaml_path: str = None
) -> Dict[str, str]:
    """
    Load prompt templates from YAML file.
    
    Args:
        yaml_path: Path to YAML file (default: broker/prompt_templates.yaml)
    
    Returns:
        Dict mapping agent_type to template string
    """
    import yaml
    from pathlib import Path
    
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "prompt_templates.yaml"
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Extract template strings from YAML structure
        templates = {}
        for key, value in data.items():
            if isinstance(value, dict) and 'template' in value:
                templates[key] = value['template']
            elif isinstance(value, str):
                templates[key] = value
        
        return templates
    except FileNotFoundError:
        return {}

