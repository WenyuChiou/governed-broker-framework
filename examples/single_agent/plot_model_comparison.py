"""
Generate Cumulative Behavior Comparison Chart (2 models x 4 subplots)
Matches old experiment README style with stacked bar charts per year
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def plot_cumulative_comparison(model_dirs: dict, output_path: str = None):
    """Create cumulative behavior comparison for multiple models."""
    
    # Define categories and colors (matching original experiment style)
    categories = [
        'Do Nothing',
        'Only Flood Insurance', 
        'Only House Elevation',
        'Both Flood Insurance and House Elevation',
        'Relocate'
    ]
    # Original color scheme: Blue=DN, Orange=FI, Green=HE, Red=Both, Purple=RL
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    n_models = len(model_dirs)
    fig, axes = plt.subplots(2, n_models, figsize=(7*n_models, 12))
    
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    for col, (model_name, model_dir) in enumerate(model_dirs.items()):
        csv_path = Path(model_dir) / 'simulation_log.csv'
        if not csv_path.exists():
            # Try without nested folder
            csv_path = Path(model_dir).parent / model_dir.split('/')[-1] / 'simulation_log.csv'
            if not csv_path.exists():
                print(f"Warning: {model_dir} not found")
                continue
            
        df = pd.read_csv(csv_path)
        years = sorted(df['year'].unique())
        
        # --- Row 1: Stacked Bar Chart by Year ---
        ax1 = axes[0, col]
        
        data = {cat: [] for cat in categories}
        for year in years:
            year_df = df[df['year'] == year]
            for cat in categories:
                count = len(year_df[year_df['decision'] == cat])
                data[cat].append(count)
        
        x = np.arange(len(years))
        width = 0.7
        bottom = np.zeros(len(years))
        
        for cat, color in zip(categories, colors):
            values = data[cat]
            ax1.bar(x, values, width, label=cat, bottom=bottom, color=color)
            bottom += np.array(values)
        
        ax1.set_xlabel('Year', fontsize=11)
        ax1.set_ylabel('Number of Agents', fontsize=11)
        ax1.set_title(f'{model_name}\nAdaptation States by Year', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend(title='Adaptation State', loc='upper right', fontsize=8)
        ax1.set_ylim(0, max(bottom) * 1.1)
        
        # --- Row 2: Cumulative Distribution ---
        ax2 = axes[1, col]
        
        # Calculate cumulative percentages by year
        cumulative_data = {cat: [] for cat in categories}
        for year in years:
            year_df = df[df['year'] <= year]
            total = len(year_df)
            for cat in categories:
                pct = len(year_df[year_df['decision'] == cat]) / total * 100 if total > 0 else 0
                cumulative_data[cat].append(pct)
        
        # Stacked area chart
        bottom = np.zeros(len(years))
        for cat, color in zip(categories, colors):
            values = cumulative_data[cat]
            ax2.fill_between(years, bottom, bottom + values, label=cat, color=color, alpha=0.8)
            bottom += np.array(values)
        
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Cumulative %', fontsize=11)
        ax2.set_title(f'{model_name}\nCumulative Distribution', fontsize=13, fontweight='bold')
        ax2.set_xlim(min(years), max(years))
        ax2.set_ylim(0, 100)
        ax2.legend(title='Adaptation State', loc='upper right', fontsize=8)
    
    plt.suptitle('Model Comparison: Flood Adaptation Behavior', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = 'model_comparison.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved comparison plot to: {output_path}")
    plt.close()

if __name__ == "__main__":
    model_dirs = {
        'Gemma 3 (4B)': 'results_comparison/gemma3_4b',
        'Llama 3.2 (3B)': 'results_comparison/llama3.2_3b'
    }
    plot_cumulative_comparison(model_dirs, 'results_comparison/model_comparison.png')

