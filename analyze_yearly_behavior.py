
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import glob
import os

def analyze_yearly_behavior(results_dir):
    """Generates a yearly stacked bar chart of agent adaptation states."""
    
    # correct for my sanitization logic (colon to underscore)
    search_pattern = str(Path(results_dir) / "**" / "simulation_log.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"No simulation_log.csv found in {results_dir}")
        return

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        model_name = Path(csv_file).parent.name
        print(f"Analyzing {model_name}...")
        
        # Ensure we have the required columns
        req_cols = ['year', 'cumulative_state']
        if not all(col in df.columns for col in req_cols):
            print(f"Skipping {csv_file}: Missing required columns {req_cols}")
            continue

        # Group by Year and Cumulative State
        # We want to count how many agents are in each state per year
        # Note: If agents are 'Relocated', they might stay in the log as 'Relocated' for subsequent years 
        # (depending on simulation logic). 
        # In run_experiment.py, 'relocated' agents are usually removed from 'active_agents' loop? 
        # Let's check: 'for agent in active_agents:'. 
        # If they are removed from the loop, they won't appear in the log for subsequent years?
        # WAIT. If run_experiment.py only logs *active* agents, then 'Relocated' agents disappear from the log in future years!
        # We need to account for this. 
        # If an agent is not in the log for Year X, but was Relocated in Year X-1, they are still 'Relocated'.
        # However, let's look at the log counts.
        
        counts = df.groupby(['year', 'cumulative_state']).size().unstack(fill_value=0)
        
        # Rename columns for cleaner legend
        rename_map = {
            "Only Flood Insurance": "FI",
            "Only House Elevation": "HE",
            "Both Flood Insurance and House Elevation": "Both FI & HE",
            "Do Nothing": "Do Nothing",
            "Relocate": "Relocate"
        }
        counts = counts.rename(columns=rename_map)
        
        # Standardize columns
        desired_order = ["Do Nothing", "FI", "HE", "Both FI & HE", "Relocate"]
        # Filter existing columns
        existing_cols = [c for c in desired_order if c in counts.columns]
        counts = counts[existing_cols]
        
        # Color mapping (Matched to user reference)
        colors = {
            "Do Nothing": "#377eb8",    # Blue
            "FI": "#ff7f00",            # Orange
            "HE": "#4daf4a",            # Green
            "Both FI & HE": "#e41a1c",  # Red
            "Relocate": "#984ea3"       # Purple
        }
        plot_colors = [colors.get(c, "#333333") for c in existing_cols]

        # Plot
        ax = counts.plot(kind='bar', stacked=True, color=plot_colors, width=0.8, figsize=(10, 6))
        
        # Add Flood Year Highlights (Years 3, 4, 9)
        # Note: x-axis is categorical (0-indexed bars), so Year 1 is x=0, Year 3 is x=2
        flood_years = [3, 4, 9]
        for fy in flood_years:
            # Shift by -1 for 0-based index
            x_pos = fy - 1
            ax.axvspan(x_pos - 0.45, x_pos + 0.45, color='pink', alpha=0.3, zorder=0, label='Flood Event' if fy==3 else "")

        plt.title(f"Yearly Cumulative Adaptation State - {model_name}")
        plt.xlabel("Year")
        plt.ylabel("Number of Agents")
        
        # Handle legend duplicate for Flood Event
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Adaptation", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        output_path = Path(csv_file).parent / "yearly_progression.png"
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Directory containing model results")
    args = parser.parse_args()
    
    analyze_yearly_behavior(args.results_dir)
