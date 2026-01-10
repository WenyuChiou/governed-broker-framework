import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib

# Define paths
BASE_DIR = pathlib.Path(__file__).parent.absolute()
BASELINE_PATH = BASE_DIR / "ref/flood_adaptation_simulation_log.csv"
FRAMEWORK_PATH = BASE_DIR / "results/llama3.2_3b/simulation_log.csv"

# Standardized Colors
COLORS = {
    'Do Nothing': 'lightgray',
    'Only Flood Insurance': 'skyblue', 
    'Only House Elevation': 'orange',
    'Relocated': 'red',
    'Both': 'purple'
}

def derive_state(row):
    """
    Derives the effective state of the agent based on state columns.
    This normalizes differences in how 'decision' might be logged.
    """
    decision = str(row.get('decision', '')).lower()
    
    # Handle both boolean and string "True"/"False" if necessary
    relocated_col = str(row.get('relocated', False)).lower() == 'true'
    elevated = str(row.get('elevated', False)).lower() == 'true'
    insurance = str(row.get('has_insurance', False)).lower() == 'true'
    
    # Baseline logs "Already relocated" with missing 'relocated' column (NaN), so we must check decision text too
    if relocated_col or 'relocate' in decision or 'already relocated' in decision:
        return 'Relocated'
    elif elevated and insurance:
        return 'Both'
    elif elevated:
        return 'Only House Elevation'
    elif insurance:
        return 'Only Flood Insurance'
    else:
        return 'Do Nothing'

def get_yearly_distribution(df):
    """
    Returns a DataFrame where index is Year and columns are States, values are counts.
    """
    # Ensure distinct agent-year rows (take last if duplicates)
    df = df.sort_values(['year', 'agent_id'])
    
    # Derive state for every row
    df['derived_state'] = df.apply(derive_state, axis=1)
    
    # Filter out 'Relocated' to show population deduction (User Request)
    # This visualizes "Active Agents" only.
    df = df[df['derived_state'] != 'Relocated']
    
    # Pivot for stacked bar chart
    pivot_df = df.groupby(['year', 'derived_state']).size().unstack(fill_value=0)
    
    # Ensure all columns exist
    for state in COLORS.keys():
        if state not in pivot_df.columns:
            pivot_df[state] = 0
            
    # Reorder columns
    return pivot_df[list(COLORS.keys())]

def main():
    if not os.path.exists(BASELINE_PATH):
        print(f"Error: Baseline file not found: {BASELINE_PATH}")
        return
    if not os.path.exists(FRAMEWORK_PATH):
        print(f"Error: Framework file not found: {FRAMEWORK_PATH}")
        return
        
    print("Loading datasets...")
    df_base = pd.read_csv(BASELINE_PATH)
    df_frame = pd.read_csv(FRAMEWORK_PATH)
    
    print("Processing yearly distributions...")
    dist_base = get_yearly_distribution(df_base)
    dist_frame = get_yearly_distribution(df_frame)
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    # Plot Baseline
    dist_base.plot(kind='bar', stacked=True, ax=axes[0], color=[COLORS[c] for c in dist_base.columns], width=0.8)
    axes[0].set_title("BASELINE (Llama 3.2:3B)\nCumulative Adaptation by Year", fontsize=14)
    axes[0].set_ylabel("Number of Agents", fontsize=12)
    axes[0].set_xlabel("Year", fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].legend(loc='lower left', title="State")
    
    # Plot Framework
    dist_frame.plot(kind='bar', stacked=True, ax=axes[1], color=[COLORS[c] for c in dist_frame.columns], width=0.8)
    axes[1].set_title("FRAMEWORK (Llama 3.2:3B)\nCumulative Adaptation by Year", fontsize=14)
    axes[1].set_xlabel("Year", fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].legend(loc='lower left', title="State")
    
    plt.tight_layout()
    output_path = "comparison_cumulative_yearly.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ Yearly cumulative comparison chart saved to {output_path}")
    
    # Print final year stats for text response
    print("\n--- Final Year (Year 10) Stats ---")
    print("Baseline:")
    print(dist_base.iloc[-1])
    print("\nFramework:")
    print(dist_frame.iloc[-1])

if __name__ == "__main__":
    main()
