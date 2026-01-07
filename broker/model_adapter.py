"""
Model Adapter - Thin layer for multi-LLM support.

The Model Adapter has ONLY two responsibilities:
1. Parse LLM output → SkillProposal
2. Format rejection/retry → LLM prompt

NO domain logic should exist in adapters!

v0.3: Unified adapter with optional preprocessor for model-specific quirks.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import re

from .skill_types import SkillProposal


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


class UnifiedAdapter(ModelAdapter):
    """
    Unified adapter supporting all models AND all agent types.
    
    Agent-type-specific parsing is configured via AGENT_TYPE_CONFIG.
    Model-specific quirks (like DeepSeek's <think> tags) are handled via preprocessor.
    
    Usage:
        # Standard household agent
        adapter = UnifiedAdapter(agent_type="household")
        
        # Insurance agent
        adapter = UnifiedAdapter(agent_type="insurance")
        
        # DeepSeek with <think> tag removal
        adapter = UnifiedAdapter(agent_type="household", preprocessor=deepseek_preprocessor)
    """
    
    # Agent-type configuration: skills, decision keywords, output format
    AGENT_TYPE_CONFIG = {
        "household": {
            "skills": {"buy_insurance", "elevate_house", "relocate", "do_nothing", 
                       "FI", "HE", "EH", "BP", "RL", "DN"},
            "decision_keywords": ["skill:", "decision:", "final decision:"],
            "skill_map_non_elevated": {
                "1": "buy_insurance", "2": "elevate_house", "3": "relocate", "4": "do_nothing"
            },
            "skill_map_elevated": {
                "1": "buy_insurance", "2": "relocate", "3": "do_nothing"
            },
            "default_skill": "do_nothing"
        },
        "insurance": {
            "skills": {"RAISE", "LOWER", "MAINTAIN", "raise_premium", "lower_premium", "maintain_premium"},
            "decision_keywords": ["decide:", "decision:", "action:"],
            "output_fields": ["interpret:", "decide:", "adj:", "reason:"],
            "default_skill": "maintain_premium"
        },
        "government": {
            "skills": {"INCREASE", "DECREASE", "MAINTAIN", "OUTREACH",
                       "increase_subsidy", "decrease_subsidy", "maintain_subsidy", "target_mg_outreach"},
            "decision_keywords": ["decide:", "decision:", "action:"],
            "output_fields": ["interpret:", "decide:", "adj:", "priority:", "reason:"],
            "default_skill": "maintain_subsidy"
        }
    }
    
    def __init__(
        self,
        agent_type: str = "household",
        preprocessor: Optional[Callable[[str], str]] = None,
        valid_skills: Optional[set] = None
    ):
        """
        Initialize unified adapter.
        
        Args:
            agent_type: Type of agent (household, insurance, government)
            preprocessor: Optional function to preprocess raw output
            valid_skills: Optional set of valid skill names (overrides agent_type config)
        """
        self.agent_type = agent_type
        self.config = self.AGENT_TYPE_CONFIG.get(agent_type, self.AGENT_TYPE_CONFIG["household"])
        self.preprocessor = preprocessor or (lambda x: x)
        self.valid_skills = valid_skills or self.config.get("skills", set())
    
    def parse_output(self, raw_output: str, context: Dict[str, Any]) -> Optional[SkillProposal]:
        """
        Parse LLM output into SkillProposal.
        
        Supports all agent types via AGENT_TYPE_CONFIG:
        - household: Skill/Decision/Final Decision format
        - insurance: DECIDE: RAISE/LOWER/MAINTAIN format
        - government: DECIDE: INCREASE/DECREASE/MAINTAIN format
        """
        # Apply preprocessor (e.g., remove <think> tags)
        cleaned_output = self.preprocessor(raw_output)
        
        agent_id = context.get("agent_id", "unknown")
        is_elevated = context.get("is_elevated", False)
        
        # Initialize results
        skill_name = None
        reasoning = {}
        adjustment = None
        
        lines = cleaned_output.strip().split('\n')
        keywords = self.config.get("decision_keywords", ["decide:", "decision:"])
        
        for line in lines:
            line_lower = line.strip().lower()
            
            # Check for decision keywords from config
            for keyword in keywords:
                if line_lower.startswith(keyword):
                    decision_text = line.split(":", 1)[1].strip() if ":" in line else ""
                    decision_lower = decision_text.lower()
                    
                    # Try to match a skill from config
                    for skill in self.valid_skills:
                        if skill.lower() in decision_lower:
                            skill_name = skill
                            break
                    break
            
            # Parse INTERPRET: for reasoning
            if line_lower.startswith("interpret:"):
                reasoning["interpret"] = line.split(":", 1)[1].strip() if ":" in line else ""
            
            # Parse ADJ: for adjustment percentage
            elif line_lower.startswith("adj:"):
                adj_text = line.split(":", 1)[1].strip() if ":" in line else ""
                try:
                    # Parse "5%" or "0.05" or "5"
                    adj_clean = adj_text.replace("%", "").strip()
                    adj_val = float(adj_clean)
                    adjustment = adj_val / 100 if adj_val > 1 else adj_val
                except ValueError:
                    adjustment = 0.0
            
            # Parse REASON: or JUSTIFICATION:
            elif line_lower.startswith("reason:") or line_lower.startswith("justification:"):
                reasoning["reason"] = line.split(":", 1)[1].strip() if ":" in line else ""
            
            # Parse PRIORITY: for government
            elif line_lower.startswith("priority:"):
                reasoning["priority"] = line.split(":", 1)[1].strip() if ":" in line else ""
            
            # Legacy household parsing
            elif line_lower.startswith("threat appraisal:"):
                reasoning["threat"] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_lower.startswith("coping appraisal:"):
                reasoning["coping"] = line.split(":", 1)[1].strip() if ":" in line else ""
            
            # Legacy: "Final Decision:" for household
            elif line_lower.startswith("final decision:") and not skill_name:
                decision_text = line.split(":", 1)[1].strip()
                decision_lower = decision_text.lower()
                
                for skill in self.valid_skills:
                    if skill.lower() in decision_lower:
                        skill_name = skill
                        break
                
                # Fallback: digit mapping for household
                if not skill_name and self.agent_type == "household":
                    for char in decision_text:
                        if char.isdigit():
                            skill_map = self.config.get("skill_map_elevated" if is_elevated else "skill_map_non_elevated", {})
                            skill_name = skill_map.get(char, self.config.get("default_skill", "do_nothing"))
                            break
        
        # Default skill from config
        if not skill_name:
            skill_name = self.config.get("default_skill", "do_nothing")
        
        # Add adjustment to reasoning if present
        if adjustment is not None:
            reasoning["adjustment"] = adjustment
        
        return SkillProposal(
            skill_name=skill_name,
            agent_id=agent_id,
            reasoning=reasoning,
            confidence=1.0,
            raw_output=raw_output
        )
    
    def format_retry_prompt(self, original_prompt: str, errors: List[str]) -> str:
        """Format retry prompt with validation errors."""
        error_text = ", ".join(errors)
        return f"""Your previous response was flagged for the following issues:
{error_text}

Please reconsider your decision and respond again.

{original_prompt}"""


# =============================================================================
# PREPROCESSORS for model-specific quirks
# =============================================================================

def deepseek_preprocessor(text: str) -> str:
    """
    Preprocessor for DeepSeek models.
    Removes <think>...</think> reasoning tags.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def json_preprocessor(text: str) -> str:
    """
    Preprocessor for models that may return JSON.
    Extracts text content from JSON if present.
    """
    import json
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Look for common fields
            for key in ['response', 'output', 'text', 'content']:
                if key in data:
                    return str(data[key])
        return text
    except json.JSONDecodeError:
        return text


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_adapter(model_name: str) -> ModelAdapter:
    """
    Get the appropriate adapter for a model.
    
    Uses UnifiedAdapter with model-specific preprocessor if needed.
    """
    model_lower = model_name.lower()
    
    # DeepSeek models use <think> tags
    if 'deepseek' in model_lower:
        return UnifiedAdapter(preprocessor=deepseek_preprocessor)
    
    # All other models use standard adapter
    # (Llama, Gemma, GPT-OSS, OpenAI, Anthropic, etc.)
    return UnifiedAdapter()


# =============================================================================
# LEGACY ALIASES (for backward compatibility)
# =============================================================================

class OllamaAdapter(UnifiedAdapter):
    """Alias for UnifiedAdapter (backward compatibility)."""
    pass


class OpenAIAdapter(UnifiedAdapter):
    """Alias for UnifiedAdapter (backward compatibility)."""
    pass
