import os
import glob
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Guidance Norms")
    parser.add_argument("--log-dir", type=str, default="guidance_norms", help="Directory containing JSON logs")
    parser.add_argument("--fig-dir", type=str, default="figures", help="Directory to save figures")
    return parser.parse_args()

def load_data(log_dir):
    data = []
    
    # Load Baseline
    # Pattern: *baseline*guidance_norms.json
    baseline_files = glob.glob(os.path.join(log_dir, "*baseline*guidance_norms.json"))
    print(f"Found {len(baseline_files)} baseline files.")
    
    for fpath in baseline_files:
        try:
            with open(fpath, 'r') as f:
                content = json.load(f)
                # Content is a list of dicts (one per prompt)
                for entry in content:
                    if 'guidance_norm_baseline' in entry and entry['guidance_norm_baseline']:
                        norms = entry['guidance_norm_baseline']
                        # norms is a list of floats (per timestep)
                        for step, val in enumerate(norms):
                            data.append({
                                'step': step,
                                'norm': val,
                                'type': 'Standard Diffusion',
                                'file': os.path.basename(fpath)
                            })
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    # Load Steered
    # Pattern: *steered*guidance_norms.json
    steered_files = glob.glob(os.path.join(log_dir, "*steered*guidance_norms.json"))
    print(f"Found {len(steered_files)} steered files.")

    for fpath in steered_files:
        try:
            with open(fpath, 'r') as f:
                content = json.load(f)
                for entry in content:
                    if 'guidance_norm_steered' in entry and entry['guidance_norm_steered']:
                        norms = entry['guidance_norm_steered']
                        for step, val in enumerate(norms):
                             data.append({
                                'step': step,
                                'norm': val,
                                'type': 'RADS (Ours)',
                                'file': os.path.basename(fpath)
                            })
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return pd.DataFrame(data)

def main():
    args = parse_args()
    
    # Setup plotting style (matching plot.py)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["P052", "Palatino", "serif"],
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })
    
    print(f"Loading data from {args.log_dir}...")
    df = load_data(args.log_dir)
    
    if df.empty:
        print("No data found to plot. Ensure you have run compute_guidance_norms.py first.")
        # Create dummy data just to verify plot loop if needed, or just exit.
        # For now, just exit.
        return

    # Transform steps to follow denoising notation (Start=50 -> End=0)
    # Original: 0 (Start) -> 49 (End)
    # New: 50 (Start) -> 1 (End)
    df['step'] = 50 - df['step']

    os.makedirs(args.fig_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 4))
    
    # Plot using seaborn
    # lineplot automatically calculates mean and confidence interval (default 95% CI)
    # The user asked for "mean and standard deviation". 
    # errorbar='sd' uses standard deviation for the shaded region.
    sns.lineplot(
        data=df, 
        x='step', 
        y='norm', 
        hue='type', 
        errorbar='sd',
        linewidth=1
    )
    
    # Invert x-axis so 50 is on Left (Start) and ~0 is on Right (End)
    plt.gca().invert_xaxis()
    
    plt.xlabel("Denoising Timestep (t)")
    plt.ylabel("L2 Norm of Classifier Guidance")
    plt.title("") # Often paper plots don't have titles, or use caption. Leaving empty for clean look.
    plt.grid(True, alpha=0.3)
    plt.legend(title=None, loc='best', frameon=True)
    
    # Add annotations for Start/End
    # Now: Max step is Start (Noise), Min step is End (Image)
    min_step = df['step'].min()
    max_step = df['step'].max()
    
    ax = plt.gca()
    trans = ax.get_xaxis_transform() # x in data coords, y in axes coords (0=bottom, 1=top)
    
    # y = -0.08 moves it higher than -0.15
    y_pos = -0.08
    plt.text(max_step, y_pos, "Start (Noise)", transform=trans, ha='center', va='top', fontsize=14)
    plt.text(min_step, y_pos, "End (Image)", transform=trans, ha='center', va='top', fontsize=14)
    
    output_path = os.path.join(args.fig_dir, "guidance_norms_plot.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
