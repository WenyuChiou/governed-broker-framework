"""
Generic Data Loader

Load agent data from CSV/Excel files without type-specific dependencies.
Returns raw Dict data that can be used to create any agent type.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_agents_csv(
    filepath: str,
    required_columns: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Load agents from CSV file as list of dicts.
    
    Args:
        filepath: Path to CSV file
        required_columns: Optional list of required column names
    
    Returns:
        List of agent data dicts
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Convert to list of dicts, handling NaN
    agents = df.to_dict('records')
    
    # Clean up values
    for agent in agents:
        for key, val in agent.items():
            # Convert NaN to None
            if pd.isna(val):
                agent[key] = None
            # Parse boolean strings
            elif isinstance(val, str) and val.upper() in ['TRUE', 'FALSE']:
                agent[key] = val.upper() == 'TRUE'
    
    print(f"[DataLoader] Loaded {len(agents)} agents from {filepath}")
    return agents


def load_agents_excel(
    filepath: str,
    sheet_name: str = "Agents"
) -> List[Dict[str, Any]]:
    """
    Load agents from Excel file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent data file not found: {filepath}")
    
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    agents = df.to_dict('records')
    
    print(f"[DataLoader] Loaded {len(agents)} agents from {filepath} (sheet: {sheet_name})")
    return agents


def export_agents_csv(
    agents: List[Dict[str, Any]],
    filepath: str,
    columns: Optional[List[str]] = None
) -> None:
    """
    Export agent data to CSV.
    
    Args:
        agents: List of agent dicts
        filepath: Output path
        columns: Optional columns to include (default: all)
    """
    df = pd.DataFrame(agents)
    
    if columns:
        df = df[columns]
    
    df.to_csv(filepath, index=False)
    print(f"[DataLoader] Exported {len(agents)} agents to {filepath}")


def create_sample_config(output_path: str = "agent_data_template.csv") -> None:
    """
    Create a sample CSV template for agent data.
    """
    sample = pd.DataFrame([
        {
            "agent_id": "H001",
            "agent_type": "household",
            "mg": True,
            "tenure": "Owner",
            "region_id": "NJ",
            "income": 45000,
            "property_value": 280000,
            "trust_gov": 0.5,
            "trust_ins": 0.5,
            "trust_neighbors": 0.5
        },
        {
            "agent_id": "H002",
            "agent_type": "household",
            "mg": False,
            "tenure": "Renter",
            "region_id": "NY",
            "income": 65000,
            "property_value": 0,
            "trust_gov": 0.6,
            "trust_ins": 0.7,
            "trust_neighbors": 0.4
        }
    ])
    sample.to_csv(output_path, index=False)
    print(f"[DataLoader] Created template at {output_path}")


# Helper for dynamic loading
def get_column_type_mapping(data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Dynamically infer basic types from data."""
    if not data:
        return {}
    mapping = {}
    first = data[0]
    for k, v in first.items():
        mapping[k] = type(v).__name__
    return mapping

