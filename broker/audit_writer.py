"""
Audit Writer - Writes structured audit traces.

Features:
- JSONL format
- Step-level traces
- Rotation support
- Summarization
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AuditConfig:
    """Configuration for audit writing."""
    output_dir: str
    log_level: str = "full"  # full, summary, errors_only
    max_entries_per_file: int = 10000
    rotate: bool = True


class AuditWriter:
    """
    Writes audit traces for decision steps.
    
    Trace fields (per specification):
    - run_id, step_id, timestamp, seed
    - agent_id
    - context_hash
    - memory_pre, memory_post
    - llm_output
    - validator_results
    - action_request (④)
    - admissible_command (⑤)
    - execution_result (⑥)
    - state_diff
    - outcome, retry_count
    """
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_file = self.output_dir / "audit_trace.jsonl"
        self.entry_count = 0
        self.file_index = 0
        
        # Summary statistics
        self.summary = {
            "total_steps": 0,
            "executed": 0,
            "retry_success": 0,
            "uncertain": 0,
            "aborted": 0
        }
    
    def write_trace(self, trace: Dict[str, Any]) -> None:
        """Write a single audit trace entry."""
        self.summary["total_steps"] += 1
        
        outcome = trace.get("outcome", "")
        if outcome == "EXECUTED":
            self.summary["executed"] += 1
        elif outcome == "RETRY_SUCCESS":
            self.summary["retry_success"] += 1
        elif outcome == "UNCERTAIN":
            self.summary["uncertain"] += 1
        elif outcome == "ABORTED":
            self.summary["aborted"] += 1
        
        # Filter based on log level
        should_write = self._should_write(trace)
        if not should_write:
            return
        
        # Check rotation
        if self.config.rotate and self.entry_count >= self.config.max_entries_per_file:
            self._rotate()
        
        # Write entry
        with open(self.current_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace, default=str) + '\n')
        
        self.entry_count += 1
    
    def _should_write(self, trace: Dict) -> bool:
        """Determine if trace should be written based on log level."""
        if self.config.log_level == "full":
            return True
        elif self.config.log_level == "summary":
            # Only write if there were issues
            return trace.get("outcome") in ["UNCERTAIN", "RETRY_SUCCESS", "ABORTED"]
        elif self.config.log_level == "errors_only":
            return trace.get("outcome") in ["UNCERTAIN", "ABORTED"]
        return True
    
    def _rotate(self) -> None:
        """Rotate to new log file."""
        self.file_index += 1
        self.current_file = self.output_dir / f"audit_trace_{self.file_index}.jsonl"
        self.entry_count = 0
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize audit and write summary."""
        summary_path = self.output_dir / "audit_summary.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                **self.summary,
                "consistency_rate": f"{self.summary['executed']/self.summary['total_steps']*100:.1f}%" 
                    if self.summary['total_steps'] > 0 else "N/A",
                "finalized_at": datetime.now().isoformat()
            }, f, indent=2)
        
        return self.summary
