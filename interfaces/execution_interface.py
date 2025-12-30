"""
Execution Interface - System-only execution (Step ⑥).

This interface is ONLY accessible by the simulation engine.
LLM agents CANNOT call this interface directly.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class AdmissibleCommand:
    """A validated command ready for execution (Step ⑤)."""
    agent_id: str
    action_name: str
    parameters: Dict[str, Any]
    admissibility_check: str = "PASSED"


@dataclass
class ExecutionResult:
    """Result from action execution (Step ⑥)."""
    success: bool
    state_changes: Dict[str, Any]
    memory_updates: list
    error: str = None


class ExecutionInterface(ABC):
    """
    System-only interface for action execution.
    
    CRITICAL: This interface MUST NOT be accessible to LLM agents.
    Only the simulation engine should call these methods.
    
    All state mutations happen ONLY through this interface.
    """
    
    @abstractmethod
    def check_admissibility(self, agent_id: str, action_name: str, 
                            parameters: Dict[str, Any]) -> AdmissibleCommand:
        """
        Check if action is admissible (Step ⑤).
        
        Args:
            agent_id: Agent to act
            action_name: Action to perform
            parameters: Action parameters
            
        Returns:
            AdmissibleCommand with PASSED or FAILED status
        """
        pass
    
    @abstractmethod
    def execute(self, command: AdmissibleCommand) -> ExecutionResult:
        """
        Execute an admissible command (Step ⑥).
        
        This is the ONLY method that mutates state.
        
        Args:
            command: Validated admissible command
            
        Returns:
            ExecutionResult with state changes
        """
        pass
    
    @abstractmethod
    def update_memory(self, agent_id: str, event: str) -> None:
        """
        Update agent memory (system-only).
        
        Memory updates are deterministic based on execution outcomes.
        LLM agents CANNOT call this directly.
        """
        pass


class SimpleExecutionInterface(ExecutionInterface):
    """Simple implementation for basic domains."""
    
    def __init__(self, state_manager: Any, action_handlers: Dict[str, callable]):
        self.state_manager = state_manager
        self.action_handlers = action_handlers
    
    def check_admissibility(self, agent_id: str, action_name: str,
                            parameters: Dict[str, Any]) -> AdmissibleCommand:
        # Basic admissibility check
        agent = self.state_manager.get_agent(agent_id)
        if agent is None:
            return AdmissibleCommand(
                agent_id=agent_id,
                action_name=action_name,
                parameters=parameters,
                admissibility_check="FAILED: Agent not found"
            )
        
        if action_name not in self.action_handlers:
            return AdmissibleCommand(
                agent_id=agent_id,
                action_name=action_name,
                parameters=parameters,
                admissibility_check="FAILED: Unknown action"
            )
        
        return AdmissibleCommand(
            agent_id=agent_id,
            action_name=action_name,
            parameters=parameters,
            admissibility_check="PASSED"
        )
    
    def execute(self, command: AdmissibleCommand) -> ExecutionResult:
        if command.admissibility_check != "PASSED":
            return ExecutionResult(
                success=False,
                state_changes={},
                memory_updates=[],
                error=command.admissibility_check
            )
        
        handler = self.action_handlers.get(command.action_name)
        if handler:
            state_changes = handler(command.agent_id, command.parameters)
            return ExecutionResult(
                success=True,
                state_changes=state_changes,
                memory_updates=[]
            )
        
        return ExecutionResult(
            success=False,
            state_changes={},
            memory_updates=[],
            error="No handler found"
        )
    
    def update_memory(self, agent_id: str, event: str) -> None:
        agent = self.state_manager.get_agent(agent_id)
        if agent and hasattr(agent, 'memory'):
            agent.memory.append(event)
            # Enforce window size
            if len(agent.memory) > 5:
                agent.memory = agent.memory[-5:]
