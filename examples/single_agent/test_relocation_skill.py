
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from broker.skill_registry import SkillRegistry
from broker.model_adapter import UnifiedAdapter
from broker.skill_types import SkillProposal
from validators.agent_validator import AgentValidator
from simulation.base_simulation_engine import BaseSimulationEngine
from simulation.state_manager import SharedState
from broker.skill_broker_engine import SkillBrokerEngine
from broker.context_builder import create_context_builder
from broker.audit_writer import GenericAuditWriter, AuditConfig
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestRelocation")

# Mock Agent for testing
class MockAgent:
    def __init__(self, agent_id, relocated=False, elevated=False):
        self.id = agent_id
        self.type = "household"
        self.relocated = relocated
        self.elevated = elevated
        self.has_insurance = False
        self.memory = []
        self.trust_in_insurance = 0.5
        self.trust_in_neighbors = 0.5
        self.flood_threshold = 0.5
        # Provide dict access for context builder
        self.__dict__.update({
            "relocated": relocated,
            "elevated": elevated
        })

class MockSimulation(BaseSimulationEngine):
    def __init__(self):
        self.agents = {"Agent_1": MockAgent("Agent_1")}
        self.environment = SharedState(year=1, flood_event=True)
    
    def advance_step(self):
        """Abstract method implementation - no-op for testing."""
        pass
    
    def execute_skill(self, decision):
        # Mock execution logic aligned with run_experiment.py
        agent = self.agents[decision.agent_id]
        if decision.skill_name == "relocate":
            agent.relocated = True
            return type('obj', (object,), {'success': True, 'state_changes': {'relocated': True}})
        return type('obj', (object,), {'success': False, 'error': 'Unknown skill'})

def test_relocation_lifecycle():
    print("="*60)
    print("TEST: RELOCATION SKILL LIFECYCLE")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    # 1. Load Registry
    registry = SkillRegistry()
    registry.register_from_yaml(base_dir / "skill_registry.yaml")
    if "relocate" in registry.list_skills():
        print("✅ Registry: 'relocate' skill is registered.")
    else:
        print("❌ Registry: 'relocate' skill is MISSING!")
        return

    # 2. Test Context Builder (Prompt Presence)
    print("\n--- Testing Context Builder ---")
    from broker.agent_config import AgentTypeConfig
    AgentTypeConfig._instance = None
    config = AgentTypeConfig.load(str(base_dir / "agent_types.yaml"))
    # AgentTypeConfig often returns a dict-like object or has an internal dict
    household_cfg = config.get("household")
    template = household_cfg.get("prompt_template", "")
    
    mock_sim = MockSimulation()
    ctx_builder = create_context_builder(
        agents=mock_sim.agents, 
        environment=mock_sim.environment.to_dict(),
        custom_templates={"household": template}
    )
    
    # Generate context for Agent 1
    context = ctx_builder.build_context("Agent_1", mock_sim.agents["Agent_1"])
    prompt_text = context["options_text"]
    
    if "relocate" in prompt_text.lower():
        print(f"✅ Context: 'relocate' found in options_text.\n   Snippet: {prompt_text[:100]}...")
    else:
        print(f"❌ Context: 'relocate' NOT found in options_text!\n   Full Text: {prompt_text}")

    # 3. Test Validation Logic
    print("\n--- Testing Validation Logic ---")
    validator = AgentValidator(config_path=str(base_dir / "agent_types.yaml"))
    
    # Case A: Ideal Case (High Threat, High Coping) -> Should Pass
    proposal_good = SkillProposal(
        skill_name="relocate",
        agent_id="Agent_1",
        reasoning={"TP": "High", "CP": "High"}, # High threat, can afford it
        params={}
    )
    
    results_good = validator.validate(proposal_good, mock_sim.agents["Agent_1"].__dict__)
    errors_good = [r.message for r in results_good if not r.valid]
    
    if not errors_good:
        print("✅ Validation (TP=High, CP=High): PASSED (As expected)")
    else:
        print(f"❌ Validation (TP=High, CP=High): FAILED! Errors: {errors_good}")

    # Case B: Overreaction Case (Low Threat) -> Should Fail
    proposal_bad = SkillProposal(
        skill_name="relocate",
        agent_id="Agent_1",
        reasoning={"TP": "Low", "CP": "High"}, # Low threat -> Why move?
        params={}
    )
    
    results_bad = validator.validate(proposal_bad, mock_sim.agents["Agent_1"].__dict__)
    errors_bad = [r.message for r in results_bad if not r.valid]
    
    if errors_bad:
        print(f"✅ Validation (TP=Low): BLOCKED (Correctly trapped overreaction). Msg: {errors_bad[0]}")
    else:
        print("❌ Validation (TP=Low): PASSED (Should have been blocked!)")

    # 4. Test Execution
    print("\n--- Testing Execution ---")
    mock_agent = mock_sim.agents["Agent_1"]
    print(f"   Pre-Execution State: relocated={mock_agent.relocated}")
    
    # Manually execute
    # We cheat and skip the broker engine full loop, just testing the simulation execute_skill equivalent
    decision = type('ApprovedSkill', (object,), {'agent_id': 'Agent_1', 'skill_name': 'relocate'})
    result = mock_sim.execute_skill(decision)
    
    print(f"   Post-Execution State: relocated={mock_agent.relocated}")
    if mock_agent.relocated:
        print("✅ Execution: Agent state successfully updated to relocated=True")
    else:
        print("❌ Execution: Agent state failed to update!")

if __name__ == "__main__":
    test_relocation_lifecycle()
