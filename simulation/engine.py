"""
Toy Domain Simulation Engine

A minimal simulation engine for testing the framework.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import random

from interfaces.execution_interface import ExecutionInterface, AdmissibleCommand, ExecutionResult


@dataclass
class ToyAgent:
    """A simple agent for toy domain."""
    id: str
    resources: float = 100.0
    threat_perception: float = 0.5
    vulnerability: float = 1.0
    last_action: str = "none"
    memory: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class ToyEnvironment:
    """Environment state for toy domain."""
    step: int = 0
    risk_level: float = 0.3
    shock_occurred: bool = False
    
    def advance(self, seed: int) -> None:
        """Advance environment by one step."""
        self.step += 1
        random.seed(seed + self.step)
        self.risk_level = random.uniform(0.1, 0.9)
        self.shock_occurred = random.random() < 0.2


class ToySimulationEngine(ExecutionInterface):
    """
    Minimal simulation engine implementing ExecutionInterface.
    
    Actions:
    - do_nothing: No cost, no benefit
    - adapt: Reduces vulnerability but costs resources
    - buy_insurance: Moderate cost, partial protection
    """
    
    def __init__(self, num_agents: int = 10, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Initialize agents
        self.agents: Dict[str, ToyAgent] = {}
        for i in range(1, num_agents + 1):
            self.agents[f"Agent_{i}"] = ToyAgent(
                id=f"Agent_{i}",
                resources=random.uniform(50, 150),
                threat_perception=random.uniform(0.2, 0.8)
            )
        
        # Initialize environment
        self.environment = ToyEnvironment()
        
        # Action handlers
        self.action_handlers = {
            "do_nothing": self._do_nothing,
            "adapt": self._adapt,
            "buy_insurance": self._buy_insurance
        }
    
    def get_agent(self, agent_id: str) -> Optional[ToyAgent]:
        return self.agents.get(agent_id)
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_id)
        if agent:
            return {
                "resources": agent.resources,
                "threat_perception": agent.threat_perception,
                "vulnerability": agent.vulnerability,
                "last_action": agent.last_action,
                "memory": agent.memory.copy(),
                "is_active": agent.is_active
            }
        return {}
    
    def get_environment_state(self) -> Dict[str, Any]:
        return {
            "step": self.environment.step,
            "risk_level": self.environment.risk_level,
            "shock_occurred": self.environment.shock_occurred
        }
    
    def advance_step(self) -> None:
        """Advance simulation by one step."""
        self.environment.advance(self.seed)
        
        # Update memory based on events
        for agent in self.agents.values():
            if agent.is_active:
                if self.environment.shock_occurred:
                    agent.memory.append(f"Step {self.environment.step}: Shock occurred!")
                    # Damage calculation
                    damage = agent.vulnerability * 20.0
                    agent.resources = max(0, agent.resources - damage)
                else:
                    agent.memory.append(f"Step {self.environment.step}: No shock.")
                
                # Decay threat perception
                agent.threat_perception *= 0.95
                
                # Keep memory bounded
                agent.memory = agent.memory[-5:]
    
    # ExecutionInterface implementation
    
    def check_admissibility(self, agent_id: str, action_name: str,
                            parameters: Dict[str, Any]) -> AdmissibleCommand:
        agent = self.get_agent(agent_id)
        
        if not agent:
            return AdmissibleCommand(
                agent_id=agent_id,
                action_name=action_name,
                parameters=parameters,
                admissibility_check="FAILED: Agent not found"
            )
        
        if not agent.is_active:
            return AdmissibleCommand(
                agent_id=agent_id,
                action_name=action_name,
                parameters=parameters,
                admissibility_check="FAILED: Agent inactive"
            )
        
        if action_name not in self.action_handlers:
            return AdmissibleCommand(
                agent_id=agent_id,
                action_name=action_name,
                parameters=parameters,
                admissibility_check="FAILED: Unknown action"
            )
        
        # Check resource constraints
        if action_name == "adapt" and agent.resources < 30:
            return AdmissibleCommand(
                agent_id=agent_id,
                action_name=action_name,
                parameters=parameters,
                admissibility_check="FAILED: Insufficient resources for adapt"
            )
        
        if action_name == "buy_insurance" and agent.resources < 15:
            return AdmissibleCommand(
                agent_id=agent_id,
                action_name=action_name,
                parameters=parameters,
                admissibility_check="FAILED: Insufficient resources for insurance"
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
            state_changes = handler(command.agent_id)
            return ExecutionResult(
                success=True,
                state_changes=state_changes,
                memory_updates=[f"Executed: {command.action_name}"]
            )
        
        return ExecutionResult(
            success=False,
            state_changes={},
            memory_updates=[],
            error="No handler"
        )
    
    def update_memory(self, agent_id: str, event: str) -> None:
        agent = self.get_agent(agent_id)
        if agent:
            agent.memory.append(event)
            agent.memory = agent.memory[-5:]
    
    # Action handlers
    
    def _do_nothing(self, agent_id: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_id)
        agent.last_action = "do_nothing"
        return {"action": "do_nothing"}
    
    def _adapt(self, agent_id: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_id)
        agent.resources -= 30.0
        agent.vulnerability = max(0.1, agent.vulnerability * 0.5)
        agent.last_action = "adapt"
        return {
            "action": "adapt",
            "resources_spent": 30.0,
            "new_vulnerability": agent.vulnerability
        }
    
    def _buy_insurance(self, agent_id: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_id)
        agent.resources -= 15.0
        agent.vulnerability = max(0.3, agent.vulnerability * 0.7)
        agent.last_action = "buy_insurance"
        return {
            "action": "buy_insurance",
            "resources_spent": 15.0,
            "new_vulnerability": agent.vulnerability
        }
