"""
Generic Agent Framework

Provides base classes for building domain-agnostic agents with:
- 0-1 normalized state
- YAML-defined configuration
- Literature-backed objectives and constraints
"""

from agents.institutional_base import (
    # Normalization utilities
    normalize,
    denormalize,
    
    # Configuration classes
    StateParam,
    Objective,
    Constraint,
    PerceptionSource,
    Skill,
    InstitutionalAgentConfig,
    
    # Agent base class
    InstitutionalAgent,
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
    'InstitutionalAgentConfig',
    
    # Agent
    'InstitutionalAgent',
    
    # Loader
    'load_agent_configs',
    'load_agents',
]
