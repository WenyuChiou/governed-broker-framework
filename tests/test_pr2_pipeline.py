import sys
from pathlib import Path
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from broker.social_graph import RandomGraph
from broker.interaction_hub import InteractionHub
from broker.context_builder import TieredContextBuilder
from broker.experiment import ExperimentBuilder
from agents.base_agent import BaseAgent, AgentConfig
from broker.skill_types import ExecutionResult

class PR2MockSim:
    def __init__(self):
        self.year = 0
    def advance_year(self):
        self.year += 1
        return {"flood_event": True if self.year == 2 else False, "news": ["Gov Policy: Aid Increased"] if self.year == 1 else []}
    def execute_skill(self, approved_skill):
        # SYSTEM-ONLY ACTING
        sid = approved_skill.skill_name
        return ExecutionResult(success=True, state_changes={"has_insurance": True if sid == "buy_insurance" else False})

def mock_llm_behavior(prompt):
    # If a flood is in memory, buy insurance
    if "major flood" in prompt:
        return "Threat Appraisal: High\nDecision: buy_insurance"
    return "Threat Appraisal: Low\nDecision: do_nothing"

def run_pipeline_test():
    print("Verifying PR 2 Multi-Layer Pipeline...")
    
    # 1. Setup Agents
    agent_ids = ["A", "B", "C"]
    agents = {}
    for aid in agent_ids:
        config = AgentConfig(name=aid, agent_type="household", 
                             state_params=[], objectives=[], constraints=[], 
                             skills=["buy_insurance", "do_nothing"])
        agent = BaseAgent(config)
        agent.id = aid
        agent.memory = []
        agents[aid] = agent

    # 2. Setup Social Layer (Everyone connected)
    graph = RandomGraph(agent_ids, p=1.0)
    hub = InteractionHub(graph)
    ctx_builder = TieredContextBuilder(agents, hub, global_news=["Initial City Plan"])
    
    # 3. Build Experiment
    builder = ExperimentBuilder()
    runner = (
        builder
        .with_model("pr2-test")
        .with_years(2)
        .with_agents(agents)
        .with_simulation(PR2MockSim())
        .with_context_builder(ctx_builder)
        .with_governance("default", "examples/single_agent/agent_types.yaml")
        .with_output("results/pr2_test")
        .build()
    )
    
    # 4. Run
    captured_prompts = []
    def intercept_llm(p):
        captured_prompts.append(p)
        return mock_llm_behavior(p)

    runner.run(llm_invoke=intercept_llm)
    
    # 5. Pipeline Verification
    print("\n--- Pipeline Check ---")
    
    # Year 1 Prompts should have Global News
    assert "Initial City Plan" in captured_prompts[0]
    
    # Year 2 Prompts should contain Gossip from Year 1 decisions
    # All agents chose 'do_nothing' in Year 1 because no flood
    y2_prompt = captured_prompts[3] # Index for Year 2, Agent A
    print(f"Year 2 Gossip Check: {y2_prompt[y2_prompt.find('### [LOCAL NEIGHBORHOOD]'):y2_prompt.find('### [CITY-WIDE NEWS]')]}")
    assert "Decided to: Do nothing" in y2_prompt, "Social Tier Failure: Gossip about Year 1 decision missing!"
    
    # Memory Check: Agent A should have 1 item in memory at start of Year 2
    assert len(agents["A"].memory) >= 2 # 1 simulation event + 1 decision
    
    print("\nPR 2 Pipeline Test PASSED!")

if __name__ == "__main__":
    run_pipeline_test()
