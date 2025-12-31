"""
Governed Broker Framework - Core Package

Governance middleware for LLM-driven Agent-Based Models.
Version 0.2.0 - Skill-Governed Architecture
"""
# Legacy action-based engine (preserved for backwards compatibility)
from .engine import BrokerEngine
from .types import BrokerResult, DecisionRequest, ValidationResult

# New skill-governed architecture
from .skill_types import (
    SkillProposal, SkillDefinition, ApprovedSkill,
    SkillBrokerResult, SkillOutcome, ExecutionResult
)
from .skill_registry import SkillRegistry, create_flood_adaptation_registry
from .skill_broker_engine import SkillBrokerEngine
from .model_adapter import ModelAdapter, OllamaAdapter, OpenAIAdapter, get_adapter

__all__ = [
    # Legacy
    "BrokerEngine", "BrokerResult", "DecisionRequest", "ValidationResult",
    # Skill-Governed
    "SkillProposal", "SkillDefinition", "ApprovedSkill", "SkillBrokerResult", "SkillOutcome",
    "SkillRegistry", "create_flood_adaptation_registry",
    "SkillBrokerEngine",
    "ModelAdapter", "OllamaAdapter", "OpenAIAdapter", "get_adapter",
    "ExecutionResult"
]
__version__ = "0.2.0"
