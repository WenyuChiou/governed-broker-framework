"""
Action Request Interface - Submit action intents (Step ④).

This interface accepts action proposals but DOES NOT execute them.
Execution is done only by the ExecutionInterface.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ActionIntent:
    """An action proposal (intent only, not executed)."""
    agent_id: str
    action_name: str
    parameters: Dict[str, Any]
    timestamp: str


class ActionRequestInterface(ABC):
    """
    Interface for submitting action requests.
    
    Key distinction:
    - This interface accepts INTENT only
    - It does NOT execute actions
    - Execution happens through ExecutionInterface
    """
    
    @abstractmethod
    def submit_request(self, intent: ActionIntent) -> Dict[str, Any]:
        """
        Submit an action request.
        
        Args:
            intent: Action intent from LLM decision
            
        Returns:
            Receipt containing:
            - request_id
            - status: "submitted" | "rejected"
            - reason (if rejected)
        """
        pass
    
    @abstractmethod
    def get_available_actions(self, agent_id: str) -> Dict[str, Any]:
        """
        Get available actions for an agent.
        
        Returns:
            Dict with action catalog + constraints
        """
        pass


class SimpleActionRequestInterface(ActionRequestInterface):
    """Simple implementation for basic domains."""
    
    def __init__(self, action_catalog: Dict[str, Any]):
        self.action_catalog = action_catalog
        self.request_counter = 0
    
    def submit_request(self, intent: ActionIntent) -> Dict[str, Any]:
        self.request_counter += 1
        
        # Basic validation
        if intent.action_name not in self.action_catalog:
            return {
                "request_id": self.request_counter,
                "status": "rejected",
                "reason": f"Unknown action: {intent.action_name}"
            }
        
        return {
            "request_id": self.request_counter,
            "status": "submitted",
            "intent": intent
        }
    
    def get_available_actions(self, agent_id: str) -> Dict[str, Any]:
        return self.action_catalog
