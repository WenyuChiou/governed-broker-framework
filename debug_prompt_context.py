import sys
from pathlib import Path

# Add core to path
FRAMEWORK_PATH = Path(__file__).parent
sys.path.insert(0, str(FRAMEWORK_PATH))
sys.path.insert(0, str(FRAMEWORK_PATH / "examples" / "single_agent"))

from examples.single_agent.run_experiment import FloodSimulation, FloodContextBuilder, setup_governance, create_llm_invoke
from broker.skill_registry import SkillRegistry
from broker.agent_config import AgentTypeConfig

def debug_decision():
    # 1. Setup minimal simulation
    sim = FloodSimulation(num_agents=1, seed=42)
    # Force a flood year context
    sim.environment['year'] = 3
    sim.environment['flood_event'] = True
    sim.environment['flood_severity'] = 0.8
    sim.grant_available = True
    
    # 2. Setup Context Builder
    skill_registry = SkillRegistry()
    registry_path = Path("examples/single_agent/skill_registry.yaml")
    skill_registry.register_from_yaml(registry_path)
    
    config_path = Path("examples/single_agent/agent_types.yaml")
    full_config = AgentTypeConfig.load(str(config_path))
    prompt_template = full_config.get("household").get("prompt_template")
    
    context_builder = FloodContextBuilder(
        skill_registry=skill_registry,
        agents=sim.agents,
        environment=sim.environment,
        prompt_templates={"household": prompt_template}
    )
    
    # 3. Build Context for Agent_1
    agent_id = "Agent_1"
    agent = sim.agents[agent_id]
    
    # Update agent memory for the debug case
    agent.memory = [
        "Year 3: A severe flood is occurring right now!",
        "Year 3: I observe deep water in my yard.",
        "Year 3: Elevation grants are available for 90% cost coverage."
    ]
    
    context = context_builder.build(agent_id)
    prompt = context_builder.format_prompt(context)
    
    print("\n" + "="*80)
    print("DEBUG: CONTEXT KEYS")
    print("="*80)
    print(list(context.keys()))
    
    print("\n" + "="*80)
    print("DEBUG: PERCEPTION VALUE")
    print("="*80)
    print(context.get('perception'))
    
    print("\n" + "="*80)
    print("DEBUG: FORMATTED PROMPT")
    print("="*80)
    print(prompt)
    
    # 4. Test Parsing with UnifiedAdapter
    from broker.model_adapter import UnifiedAdapter
    adapter = UnifiedAdapter(agent_type="household", config_path=str(config_path))
    
    # Mock some responses
    mock_responses = [
        "Threat Appraisal: High because water is everywhere.\nCoping Appraisal: Medium because I can buy insurance.\nFinal Decision: buy_insurance",
        "Threat Appraisal: Low because I'm brave.\nCoping Appraisal: High.\nFinal Decision: do_nothing"
    ]
    
    print("\n" + "="*80)
    print("DEBUG: PARSING TEST")
    print("="*80)
    for resp in mock_responses:
        proposal = adapter.parse_output(resp, context)
        print(f"Raw: {resp.replace('\\n', ' ')}")
        print(f"Parsed Skill: {proposal.skill_name}")
        print("-" * 40)

if __name__ == "__main__":
    debug_decision()
