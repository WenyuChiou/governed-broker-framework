"""
Model Adapter - Thin layer for multi-LLM support.

The Model Adapter has ONLY two responsibilities:
1. Parse LLM output → SkillProposal
2. Format rejection/retry → LLM prompt

NO domain logic should exist in adapters!
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import re

from skill_types import SkillProposal


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Each LLM type (Ollama, OpenAI, Anthropic) has its own adapter
    that normalizes output into SkillProposal format.
    """
    
    @abstractmethod
    def parse_output(self, raw_output: str, context: Dict[str, Any]) -> Optional[SkillProposal]:
        """
        Parse LLM output into SkillProposal.
        
        Args:
            raw_output: Raw text output from LLM
            context: Current context (for skill mapping)
            
        Returns:
            SkillProposal or None if parsing fails
        """
        pass
    
    @abstractmethod
    def format_retry_prompt(self, original_prompt: str, errors: List[str]) -> str:
        """
        Format retry prompt with validation errors.
        
        Args:
            original_prompt: The original prompt sent to LLM
            errors: List of validation errors
            
        Returns:
            Formatted retry prompt
        """
        pass


class OllamaAdapter(ModelAdapter):
    """Adapter for Ollama models (Llama, Gemma, DeepSeek, etc.)."""
    
    # Skill name mappings from decision codes
    SKILL_MAP_NON_ELEVATED = {
        "1": "buy_insurance",
        "2": "elevate_house",
        "3": "relocate",
        "4": "do_nothing"
    }
    
    SKILL_MAP_ELEVATED = {
        "1": "buy_insurance",
        "2": "relocate",
        "3": "do_nothing"
    }
    
    def parse_output(self, raw_output: str, context: Dict[str, Any]) -> Optional[SkillProposal]:
        """Parse Ollama model output into SkillProposal."""
        agent_id = context.get("agent_id", "unknown")
        is_elevated = context.get("is_elevated", False)
        
        # Extract reasoning
        threat_appraisal = ""
        coping_appraisal = ""
        decision_code = ""
        
        lines = raw_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("threat appraisal:"):
                threat_appraisal = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.lower().startswith("coping appraisal:"):
                coping_appraisal = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.lower().startswith("final decision:"):
                decision_text = line.split(":", 1)[1].strip() if ":" in line else ""
                # Extract digit
                for char in decision_text:
                    if char.isdigit():
                        decision_code = char
                        break
        
        # Map decision code to skill name
        skill_map = self.SKILL_MAP_ELEVATED if is_elevated else self.SKILL_MAP_NON_ELEVATED
        skill_name = skill_map.get(decision_code, "do_nothing")
        
        return SkillProposal(
            skill_name=skill_name,
            agent_id=agent_id,
            reasoning={
                "threat": threat_appraisal,
                "coping": coping_appraisal
            },
            confidence=1.0,
            raw_output=raw_output
        )
    
    def format_retry_prompt(self, original_prompt: str, errors: List[str]) -> str:
        """Format retry prompt for Ollama models."""
        error_text = ", ".join(errors)
        return f"""Your previous response was flagged for the following issues:
{error_text}

Please reconsider your decision and respond again.

{original_prompt}"""


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models (GPT-4, etc.)."""
    
    SKILL_MAP_NON_ELEVATED = {
        "1": "buy_insurance",
        "2": "elevate_house",
        "3": "relocate",
        "4": "do_nothing"
    }
    
    SKILL_MAP_ELEVATED = {
        "1": "buy_insurance",
        "2": "relocate",
        "3": "do_nothing"
    }
    
    def parse_output(self, raw_output: str, context: Dict[str, Any]) -> Optional[SkillProposal]:
        """Parse OpenAI model output into SkillProposal."""
        # Same parsing logic as Ollama for now
        # Can be extended for JSON mode outputs
        agent_id = context.get("agent_id", "unknown")
        is_elevated = context.get("is_elevated", False)
        
        threat_appraisal = ""
        coping_appraisal = ""
        decision_code = ""
        
        lines = raw_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("threat appraisal:"):
                threat_appraisal = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.lower().startswith("coping appraisal:"):
                coping_appraisal = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.lower().startswith("final decision:"):
                decision_text = line.split(":", 1)[1].strip() if ":" in line else ""
                for char in decision_text:
                    if char.isdigit():
                        decision_code = char
                        break
        
        skill_map = self.SKILL_MAP_ELEVATED if is_elevated else self.SKILL_MAP_NON_ELEVATED
        skill_name = skill_map.get(decision_code, "do_nothing")
        
        return SkillProposal(
            skill_name=skill_name,
            agent_id=agent_id,
            reasoning={"threat": threat_appraisal, "coping": coping_appraisal},
            confidence=1.0,
            raw_output=raw_output
        )
    
    def format_retry_prompt(self, original_prompt: str, errors: List[str]) -> str:
        """Format retry prompt for OpenAI models."""
        return f"""Your previous response was flagged:
{', '.join(errors)}

Please reconsider and respond again.

{original_prompt}"""


def get_adapter(model_name: str) -> ModelAdapter:
    """Get the appropriate adapter for a model."""
    model_lower = model_name.lower()
    
    if any(x in model_lower for x in ['gpt', 'openai']):
        return OpenAIAdapter()
    else:
        # Default to Ollama adapter for local models
        return OllamaAdapter()
