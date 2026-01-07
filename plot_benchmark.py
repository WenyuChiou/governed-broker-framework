import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

# Configuration matching user's request
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_MAP = {
    "Do Nothing": "#1f77b4",        # Blue
    "Only Flood Insurance": "#ff7f0e", # Orange
    "Only House Elevation": "#2ca02c", # Green
    "Both Flood Insurance and House Elevation": "#d62728", # Red
    "Relocate": "#9467bd"           # Purple
}

ORDER = [
    "Do Nothing",
    "Only Flood Insurance",
    "Only House Elevation",
    "Both Flood Insurance and House Elevation",
    "Relocate"
]

def plot_results(log_path, label):
    if not log_path.exists():
        print(f"Skipping {label} - No log found")
        return

    print(f"Plotting for {label}...")
    df = pd.read_csv(log_path)
    
    # Use 'cumulative_state' if it exists, fallback to 'decision'
    target_col = 'cumulative_state' if 'cumulative_state' in df.columns else 'decision'
    
    # Group by Year and the target column
    counts = df.groupby(['year', target_col]).size().unstack(fill_value=0)
    
    # Rename columns for shorter legend and aggregate states
    SHORT_LABELS = {
        "Do Nothing": "Do Nothing",
        "Only Flood Insurance": "Insurance",
        "Only House Elevation": "Elevation",
        "Both Flood Insurance and House Elevation": "Ins + Elev",
        "Relocate": "Relocate",
        "Already relocated": "Relocate" # Aggregate legacy state
    }
    # First, handle missing states in columns to avoid errors during aggregation
    for old_state in SHORT_LABELS.keys():
        if old_state not in counts.columns:
            counts[old_state] = 0
            
    # Aggregate duplicate groups (like Already relocated -> Relocate)
    mapped_counts = pd.DataFrame()
    for short_label in set(SHORT_LABELS.values()):
        relevant_cols = [old for old, short in SHORT_LABELS.items() if short == short_label]
        mapped_counts[short_label] = counts[relevant_cols].sum(axis=1)
    
    counts = mapped_counts
    
    # Update order for plot (using short names now)
    PLOT_ORDER = ["Do Nothing", "Insurance", "Elevation", "Ins + Elev", "Relocate"]
    counts = counts[PLOT_ORDER]
    
    # Update color map to match new labels
    NEW_COLOR_MAP = {
        "Do Nothing": "#1f77b4",
        "Insurance": "#ff7f0e",
        "Elevation": "#2ca02c",
        "Ins + Elev": "#d62728",
        "Relocate": "#9467bd"
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts.plot(kind='bar', stacked=True, ax=ax, color=[NEW_COLOR_MAP.get(c, "#333333") for c in counts.columns], width=0.5)
    
    ax.set_title(f"Adaptation States by Year ({label})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Agents")
    ax.set_ylim(0, 100) # Assuming 100 agents provided by user context
    ax.legend(title="Adaptation State", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    # Use label to distinguish
    output_path = OUTPUT_DIR / f"plot_{label}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help="Directory to scan for log files")
    parser.add_argument("--file", type=str, default=None, help="Direct path to a simulation_log.csv")
    parser.add_argument("--label", type=str, default="custom", help="Label for --file output")
    args = parser.parse_args()
    
    if args.file:
        plot_results(Path(args.file), args.label)
    elif args.dir:
        scan_dir = Path(args.dir)
        if not scan_dir.exists():
            print(f"Directory {scan_dir} not found.")
            return

        # Find all model subdirectories
        for item in scan_dir.iterdir():
            if item.is_dir() and item.name != "plots":
                log_p = item / "simulation_log.csv"
                if log_p.exists():
                    plot_results(log_p, f"{scan_dir.name}_{item.name}")
    else:
        # Default behavior: scan results dir
        scan_dir = Path("results")
        if scan_dir.exists():
            for item in scan_dir.iterdir():
                if item.is_dir() and item.name != "plots":
                    log_p = item / "simulation_log.csv"
                    if log_p.exists():
                        plot_results(log_p, f"results_{item.name}")

if __name__ == "__main__":
    main()
