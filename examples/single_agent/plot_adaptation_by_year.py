"""
Generate Stacked Bar Chart: Adaptation States by Year
Matches the style from original LLMABMPMT-Final.py experiments
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def plot_adaptation_by_year(csv_path: str, output_path: str = None):
    """Create stacked bar chart showing adaptation states by year."""
    df = pd.read_csv(csv_path)
    
    # Define categories and colors (matching original)
    categories = [
        'Do Nothing',
        'Only Flood Insurance', 
        'Only House Elevation',
        'Both Flood Insurance and House Elevation',
        'Relocate'
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Count decisions by year and category
    years = sorted(df['year'].unique())
    
    data = {cat: [] for cat in categories}
    for year in years:
        year_df = df[df['year'] == year]
        for cat in categories:
            count = len(year_df[year_df['decision'] == cat])
            data[cat].append(count)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(years))
    width = 0.7
    
    bottom = np.zeros(len(years))
    for cat, color in zip(categories, colors):
        values = data[cat]
        ax.bar(x, values, width, label=cat, bottom=bottom, color=color)
        bottom += np.array(values)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Agents', fontsize=12)
    ax.set_title('Adaptation States by Year', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(title='Adaptation State', loc='upper right', fontsize=9)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = Path(csv_path).parent / 'adaptation_states_by_year.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved chart to: {output_path}")
    plt.close()
    
    # Print summary
    print("\n=== Decision Summary ===")
    for cat in categories:
        total = sum(data[cat])
        print(f"{cat}: {total}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "results_seed42/llama3.2_3b/simulation_log.csv"
    
    plot_adaptation_by_year(csv_path)
