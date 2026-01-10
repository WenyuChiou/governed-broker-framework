import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib
import glob

# Setup Paths
BASE_DIR = pathlib.Path(__file__).parent.absolute()
BASELINE_PATH = BASE_DIR / "ref/flood_adaptation_simulation_log.csv"
RESULTS_DIR = BASE_DIR / "results"

# Standardized Colors
COLORS = {
    'Do Nothing': '#d62728',          # Red
    'Only Flood Insurance': '#ff7f0e', # Orange
    'Only House Elevation': '#2ca02c', # Green
    'Relocated': '#1f77b4',           # Blue (Excluded from Active if filtered)
    'Both': '#9467bd'                 # Purple
}

def derive_state(row):
    """Derives effective state."""
    decision = str(row.get('decision', '')).lower()
    relocated_col = str(row.get('relocated', False)).lower() == 'true'
    elevated = str(row.get('elevated', False)).lower() == 'true'
    insurance = str(row.get('has_insurance', False)).lower() == 'true'
    
    # Check relocation first
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

def process_data(file_path):
    """Loads and processes a single log file."""
    if not os.path.exists(file_path):
        return None
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    # Standardize columns
    cols = [c for c in df.columns if c.strip()]
    df = df[cols]
    
    # Process
    df['derived_state'] = df.apply(derive_state, axis=1)
    
    # EXCLUDE Relocated to show active population
    df = df[df['derived_state'] != 'Relocated']
    
    # Group
    pivot = df.groupby(['year', 'derived_state']).size().unstack(fill_value=0)
    
    # Fill missing columns
    for c in COLORS.keys():
        if c not in pivot.columns:
            pivot[c] = 0
            
    # Remove Relocated column if it exists (should be empty/0 now)
    if 'Relocated' in pivot.columns:
        pivot = pivot.drop(columns=['Relocated'])
        
    return pivot

def main():
    # 1. Load Baseline (Reference)
    print("Loading Baseline...")
    baseline_data = process_data(BASELINE_PATH)
    
    # 2. Define Model Paths (Explicit)
    # Using specific paths to ensure we get the right runs
    models = [
        {
            "name": "Llama 3.2 3B",
            "path": RESULTS_DIR / "llama3.2_3b/simulation_log.csv"
        },
        {
            "name": "Gemma 3 4B",
            "path": RESULTS_DIR / "gemma3_4b/simulation_log.csv"
        },
        {
            "name": "DeepSeek R1 1.5B",
            "path": RESULTS_DIR / "deepseek-r1_1.5b/simulation_log.csv"
        },
        {
            "name": "GPT-OSS",
            # Handle nested path for GPT-OSS retry
            "path": RESULTS_DIR / "gpt-oss_retry/gpt-oss/simulation_log.csv"
        }
    ]

    # 3. Setup Plot (2 Rows x 4 Columns)
    # Row 1: Baseline (Repeated for comparison)
    # Row 2: Framework Results
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharey=True)
    
    # Add Row Labels
    pad = 5
    for ax, col in zip(axes[0], models):
        ax.annotate(col["name"], xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline', weight='bold')

    rows = ['Baseline (Original)', 'Framework (Governed)']
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90, weight='bold')

    # 4. Plot Loops
    for i, model_info in enumerate(models):
        # --- Top Row: Baseline ---
        ax_top = axes[0, i]
        if baseline_data is not None:
             plot_colors = [COLORS[c] for c in baseline_data.columns]
             baseline_data.plot(kind='bar', stacked=True, ax=ax_top, color=plot_colors, width=0.9, legend=False)
             ax_top.set_title("") # Title handled by col label
             if i == 0: ax_top.set_ylabel("Active Agents")
             ax_top.set_xlabel("")
             ax_top.tick_params(labelbottom=False)

        # --- Bottom Row: Framework ---
        ax_bot = axes[1, i]
        print(f"Processing {model_info['name']} at {model_info['path']}...")
        model_data = process_data(model_info['path'])
        
        if model_data is not None:
             plot_colors = [COLORS[c] for c in model_data.columns]
             model_data.plot(kind='bar', stacked=True, ax=ax_bot, color=plot_colors, width=0.9, legend=False)
             if i == 0: ax_bot.set_ylabel("Active Agents")
             ax_bot.set_xlabel("Year")
        else:
            ax_bot.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
    
    # Shared Legend (Use Baseline's handle)
    if baseline_data is not None:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    plt.tight_layout()
    # Adjust for legend at bottom
    plt.subplots_adjust(bottom=0.1) 
    
    output_file = BASE_DIR / "multi_model_comparison_2x4.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved comparison to {output_file}")

if __name__ == "__main__":
    main()
