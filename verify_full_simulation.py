
import json
import argparse
from pathlib import Path
import re
from collections import defaultdict

def verify_simulation(audit_file, report_file):
    """
    Detailed verification of Context (Input) and Output for all agents over 10 years.
    """
    file_path = Path(audit_file)
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    print(f"Verifying {file_path}...")
    
    agents = defaultdict(dict)
    validation_stats = {
        "total_steps": 0,
        "valid_json": 0,
        "memory_correct": 0,
        "trust_verbalized": 0,
        "skills_formatted": 0,
        "attachment_phrase": 0,
        "high_threat_blocked": 0,
        "high_threat_do_nothing_allowed": 0 # Should be low/zero if blocking works
    }
    
    # Flood Years: 3, 4, 9 (from previous info)
    PROMPT_PARITY_PHRASE = "strong attachment to your community"
    TRUST_PARITY_PHRASES = ["strongly trust", "moderately trust", "have slight doubts about", "deeply distrust"]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                aid = entry.get('agent_id')
                year = entry.get('step') # Assuming step matches year 1-10
                prompt = entry.get('prompt', '')
                response_str = entry.get('response', '')
                
                validation_stats["total_steps"] += 1
                
                # --- 1. CONTEXT VERIFICATION ---
                
                # A. Community Attachment
                if PROMPT_PARITY_PHRASE in prompt:
                    validation_stats["attachment_phrase"] += 1
                
                # B. Trust Verbalization
                if any(phrase in prompt for phrase in TRUST_PARITY_PHRASES):
                    validation_stats["trust_verbalized"] += 1
                    
                # C. Skill Formatting (1-indexed)
                if "1. Buy flood insurance" in prompt: # Check for the specific baseline string start
                    validation_stats["skills_formatted"] += 1
                
                # D. Memory Logic
                memory_block_match = re.search(r"Memory:\s*(.*?)\s*Options:", prompt, re.DOTALL)
                memory_content = memory_block_match.group(1) if memory_block_match else ""
                
                mem_check = False
                if year == 1:
                    # Year 1 memory should be effectively empty or just static past events, NO simulation history
                    if "Year" not in memory_content: 
                        mem_check = True
                elif year > 3: # Flood occurred in Y3
                    # Should mention flood
                    if "flood" in memory_content.lower():
                        mem_check = True
                else:
                    mem_check = True # Loose check for other years
                
                if mem_check:
                    validation_stats["memory_correct"] += 1

                # --- 2. OUTPUT VERIFICATION ---
                
                # E. JSON Structure
                try:
                    # Clean markdown code blocks if present
                    clean_response = response_str.replace("```json", "").replace("```", "").strip()
                    response_json = json.loads(clean_response)
                    validation_stats["valid_json"] += 1
                    
                    # F. Logic Check (Urgency Blocking)
                    # We need to see if TP=High led to Do Anything
                    # Note: Response might be nested or flat depending on model adapter
                    # Standard Framework expectation: {"pmt_eval": {"threat_appraisal": {"level": "..."}}}
                    
                    tp_level = "Unknown"
                    decision = "Unknown"
                    
                    if "pmt_eval" in response_json:
                        tp_level = response_json["pmt_eval"]["threat_appraisal"].get("level", "Unknown")
                        decision = response_json.get("decision", {}).get("id", "Unknown")
                    elif "Threat Appraisal" in response_json: # Fallback/Flat
                        tp_level = response_json.get("Threat Appraisal")
                        decision = response_json.get("Final Decision")
                        
                    # Normalize High
                    is_high_threat = tp_level in ["High", "H", "HIGH"]
                    is_do_nothing = str(decision) in ["4", "do_nothing", "Do Nothing"]
                    
                    if is_high_threat and is_do_nothing:
                        validation_stats["high_threat_do_nothing_allowed"] += 1
                    
                    # Store for report
                    agents[aid][year] = {
                        "TP": tp_level,
                        "Dec": decision,
                        "Mem_Valid": mem_check
                    }
                    
                except json.JSONDecodeError:
                    pass # Invalid JSON count remains 0 for this entry
                    
            except Exception as e:
                print(f"Error processing line: {e}")

    # Generate Report
    with open(report_file, 'w', encoding='utf-8') as r:
        r.write("# Detailed Simulation Verification Report\n\n")
        r.write(f"**Total Steps Analyzed**: {validation_stats['total_steps']}\n\n")
        
        r.write("## 1. Context (Input) Integrity\n")
        r.write(f"- **Prompt Parity Phrase ('Attachment')**: {validation_stats['attachment_phrase']}/{validation_stats['total_steps']} ({(validation_stats['attachment_phrase']/validation_stats['total_steps'])*100:.1f}%)\n")
        r.write(f"- **Trust Verbalization**: {validation_stats['trust_verbalized']}/{validation_stats['total_steps']} ({(validation_stats['trust_verbalized']/validation_stats['total_steps'])*100:.1f}%)\n")
        r.write(f"- **Skill Formatting (1. Buy...)**: {validation_stats['skills_formatted']}/{validation_stats['total_steps']} ({(validation_stats['skills_formatted']/validation_stats['total_steps'])*100:.1f}%)\n")
        r.write(f"- **Memory Logic**: {validation_stats['memory_correct']}/{validation_stats['total_steps']} ({(validation_stats['memory_correct']/validation_stats['total_steps'])*100:.1f}%)\n\n")
        
        r.write("## 2. Output Integrity\n")
        r.write(f"- **Valid JSON Responses**: {validation_stats['valid_json']}/{validation_stats['total_steps']} ({(validation_stats['valid_json']/validation_stats['total_steps'])*100:.1f}%)\n")
        r.write(f"- **High Threat 'Do Nothing' Events**: {validation_stats['high_threat_do_nothing_allowed']} (Potential Validator Failures or Justified Exceptions)\n\n")
        
        r.write("## 3. Agent Sample Traces\n")
        sample_agents = list(agents.keys())[:3] # First 3 agents
        for aid in sample_agents:
            r.write(f"### {aid}\n")
            r.write("| Year | Threat Appraisal | Decision | Memory Valid? |\n")
            r.write("| :--- | :--- | :--- | :--- |\n")
            # Sort years
            years = sorted(agents[aid].keys(), key=lambda x: float(x))
            for y in years:
                d = agents[aid][y]
                r.write(f"| {y} | {d['TP']} | {d['Dec']} | {d['Mem_Valid']} |\n")
            r.write("\n")

    print(f"Report generated at {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_file", help="Path to default_audit.jsonl")
    parser.add_argument("--report-file", default="verification_report.md", help="Output report file")
    args = parser.parse_args()
    
    verify_simulation(args.audit_file, args.report_file)
