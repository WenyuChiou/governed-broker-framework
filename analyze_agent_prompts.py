
import json
import argparse
from pathlib import Path

def analyze_prompts(audit_file, agent_ids):
    """
    Extracts and displays the prompt evolution for specific agents.
    Analysis includes: Memory, Trust Statements, Elevation Status, and Options.
    """
    file_path = Path(audit_file)
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    print(f"Analyzing prompts for Agents: {agent_ids} in {file_path}")
    
    agent_data = {aid: {} for aid in agent_ids}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                aid = entry.get('agent_id')
                year = entry.get('step') # Assuming step maps to year in single_agent logic 1-10
                
                # In single_agent run_experiment, step_id increments for every agent action.
                # But typically we want to group by 'year' context.
                # The 'prompt' is what we want.
                
                if aid in agent_ids:
                    # Parse the prompt text to extracting key sections
                    prompt_text = entry.get('prompt', '')
                    
                    # Extract roughly by sections
                    # This is brittle but effective for quick inspection
                    
                    data = {}
                    
                    # Status
                    if "Your house is already elevated" in prompt_text:
                        data['status'] = "Elevated"
                    elif "You have not elevated" in prompt_text:
                        data['status'] = "Not Elevated"
                    else:
                        data['status'] = "Unknown"
                        
                    if "You have flood insurance" in prompt_text:
                         data['insurance'] = "Yes"
                    elif "You do not have flood insurance" in prompt_text:
                         data['insurance'] = "No"
                    
                    # Trust
                    # Extract line containing "You [trust info] your neighbors..."
                    lines = prompt_text.split('\n')
                    trust_line = next((l for l in lines if "your neighbors" in l and "insurance companies" in l), "Not Found")
                    data['trust_line'] = trust_line.strip()
                    
                    # Memory
                    # Extract content between Memory: and Options:
                    try:
                        mem_start = prompt_text.index("Memory:") + len("Memory:")
                        opts_start = prompt_text.index("Options:")
                        memory_block = prompt_text[mem_start:opts_start].strip()
                        data['memory'] = memory_block.replace('\n', ' | ')
                    except ValueError:
                        data['memory'] = "Parse Error"
                        
                    agent_data[aid][year] = data
                    
            except json.JSONDecodeError:
                continue

    # Print Report
    for aid in agent_ids:
        print(f"\n{'='*40}")
        print(f"AGENT: {aid}")
        print(f"{'='*40}")
        
        # Sort by step/year (keys might be string or int)
        # Filter valid keys
        valid_keys = [k for k in agent_data[aid].keys() if k is not None]
        try:
            steps = sorted(valid_keys, key=lambda x: float(x))
        except ValueError:
            steps = sorted(valid_keys)
        
        for step in steps:
            info = agent_data[aid][step]
            print(f"Step {step}:")
            print(f"  Status: {info.get('status')} | Insurance: {info.get('insurance')}")
            print(f"  Trust: {info.get('trust_line')}")
            print(f"  Memory: {info.get('memory')[:200]}..." if len(info.get('memory', '')) > 200 else f"  Memory: {info.get('memory')}")
            print("-" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_file", help="Path to default_audit.jsonl")
    parser.add_argument("--agents", nargs="+", default=["Agent_1", "Agent_5"], help="Agent IDs to analyze")
    args = parser.parse_args()
    
    analyze_prompts(args.audit_file, args.agents)
