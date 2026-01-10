import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read baseline simulation log
df = pd.read_csv('ref/flood_adaptation_simulation_log.csv')

# The 'decision' column in baseline CSV already contains cumulative state!
print("=== BASELINE CUMULATIVE BEHAVIOR ANALYSIS ===\n")
print("Total Records:", len(df))
print("Unique cumulative states:", df['decision'].unique())
print("\n=== Year-by-Year Cumulative State Distribution ===\n")

# Analyze each year
yearly_data = []
for year in range(1, 11):
    year_df = df[df['year'] == year]
    print(f"\n--- Year {year} ---")
    state_counts = year_df['decision'].value_counts()
    print(state_counts)
    print(f"Total: {len(year_df)} agents")
    
    # Store for plotting
    yearly_data.append({
        'year': year,
        'Do Nothing': state_counts.get('Do Nothing', 0),
        'Only Flood Insurance': state_counts.get('Only Flood Insurance', 0),
        'Only House Elevation': state_counts.get('Only House Elevation', 0),
        'Both': state_counts.get('Both Flood Insurance and House Elevation', 0),
        'Relocate': state_counts.get('Relocate', 0)
    })

# Create visualization
plot_df = pd.DataFrame(yearly_data)

fig, ax = plt.subplots(figsize=(14, 8))
width = 0.6
x = np.arange(len(plot_df))

# Define colors
colors = {
    'Do Nothing': '#e74c3c',
    'Only Flood Insurance': '#3498db',
    'Only House Elevation': '#2ecc71',
    'Both': '#9b59b6',
    'Relocate': '#f39c12'
}

# Stacked bar chart
bottom = np.zeros(len(plot_df))
for label in ['Do Nothing', 'Only Flood Insurance', 'Only House Elevation', 'Both', 'Relocate']:
    ax.bar(x, plot_df[label], width, label=label, bottom=bottom, color=colors[label])
    bottom += plot_df[label].values

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Agents', fontsize=12)
ax.set_title('Baseline: Cumulative Adaptation States by Year\n(100 Agents, Gemma 3 4B)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(plot_df['year'])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(axis='y', alpha=0.3)

# Mark flood years
for fy in [3, 4, 9]:
    ax.axvline(x=fy-1, color='red', linestyle='--', alpha=0.5, label=f'Flood Year {fy}' if fy==3 else '')
ax.text(2.1, 95, '🌊', fontsize=14)
ax.text(3.1, 95, '🌊', fontsize=14)
ax.text(8.1, 95, '🌊', fontsize=14)

plt.tight_layout()
plt.savefig('C:/Users/wenyu/.gemini/antigravity/brain/f2b36be6-b9c6-4a3f-8a0e-010648b12f8f/baseline_cumulative_states.png', dpi=150, bbox_inches='tight')
print("\n✅ Plot saved to baseline_cumulative_states.png")

# Final summary
print("\n=== Year 10 Final Summary ===")
final = df[df['year'] == 10]['decision'].value_counts()
print(final)
total = len(df[df['year'] == 10])
dn = final.get('Do Nothing', 0)
print(f"\nAdaptation Rate: {100 - (dn / total * 100):.1f}%")
