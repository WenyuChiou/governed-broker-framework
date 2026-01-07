"""
Generic Agent Framework

Provides base classes for building domain-agnostic agents with:
- 0-1 normalized state
- YAML-defined configuration
- Literature-backed objectives and constraints

Usage:
    # Load from user-defined YAML
    agents = load_agents("my_agents.yaml")
    
    # Or create programmatically
    from agents import BaseAgent, AgentConfig
"""

from agents.base_agent import (
    # Normalization utilities
    normalize,
    denormalize,
    
    # Configuration classes
    StateParam,
    Objective,
    Constraint,
    PerceptionSource,
    Skill,
    AgentConfig,
    
    # Agent base class
    BaseAgent,
)

from agents.loader import (
    load_agent_configs,
    load_agents,
)


__all__ = [
    # Utils
    'normalize',
    'denormalize',
    
    # Config
    'StateParam',
    'Objective', 
    'Constraint',
    'PerceptionSource',
    'Skill',
    'AgentConfig',
    
    # Agent
    'BaseAgent',
    
    # Loader
    'load_agent_configs',
    'load_agents',
]
