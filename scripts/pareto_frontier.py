import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 1. Set the general theme & Font (Palatino)
sns.set_theme(style="whitegrid")

# Explicitly set the font family to serif and prioritize Palatino
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Palatino LT STD", "Book Antiqua", "Georgia"],
    "text.usetex": False  
})

# 2. Comprehensive Data from Table 1a (Webster Main) and Table 1c (MemBench)
methods = ["No mitigation", "Wen '24", "Ren '24", "Hintersdorf '24", "Jain '25", "RADS (Ours)"]
colors = ["#95a5a6", "#3498db", "#9b59b6", "#f1c40f", "#e74c3c", "#2ecc71"]

full_data = {
    "Webster": {
        "sscd_target": [0.6364, 0.4187, 0.3781, 0.4149, 0.1816, 0.2261],
        "sscd_target_err": [0.2626, 0.2796, 0.2420, 0.2657, 0.1144, 0.1090],
        "clip": [0.3129, 0.3056, 0.2940, 0.3119, 0.2266, 0.2909],
        "clip_err": [0.0279, 0.0329, 0.0500, 0.0296, 0.0532, 0.0370],
        "fid": [42.1447, 31.7825, 32.7631, 32.4977, 63.9765, 31.5678],
        "fid_err": [5.9063, 4.8706, 4.7828, 4.6376, 16.1718, 5.8163]
    },
    "MemBench": {
        "sscd_target": [0.5030, 0.2091, 0.2184, np.nan, 0.1775, 0.1449],
        "sscd_target_err": [0.2836, 0.1494, 0.1482, np.nan, 0.1226, 0.0439],
        "clip": [0.3005, 0.2998, 0.2937, np.nan, 0.2520, 0.2540],
        "clip_err": [0.0392, 0.0388, 0.0426, np.nan, 0.0470, 0.0411],
        "fid": [22.0230, 18.7822, 20.1319, np.nan, 38.5641, 26.7537],
        "fid_err": [0.8903, 1.9536, 2.6071, np.nan, 7.2228, 6.5509]
    }
}

# 3. Define Plotting Logic and Generate Separate Plots
def plot_dataset(dataset_name, data, output_filename, title):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    configs = [
        (0, "fid", r"Quality ($-\log_{10}(\mathrm{FID}) \uparrow$)", "o"),
        (1, "clip", r"Alignment (CLIP Score $\uparrow$)", "o"),
    ]
    
    x_label = r"Diversity ($1 - SSCD_{target} \uparrow$)"
    scatter_handles = []
    
    for row, metric, y_label, marker in configs:
        ax = axes[row]
        
        for i in range(len(methods)):
            m_sscd = data["sscd_target"][i]
            if i == 0: # Skip "No mitigation"
                continue
            
            if np.isnan(m_sscd):
                continue

            x = 1 - m_sscd
            x_err = data["sscd_target_err"][i]
            
            if metric == "clip":
                y = data["clip"][i]
                y_err = data["clip_err"][i]
            else: # fid
                m_fid = data["fid"][i]
                y = -np.log10(m_fid)
                y_err = data["fid_err"][i] / (m_fid * np.log(10))

            ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', 
                         ecolor=colors[i], alpha=0.3, capsize=3, elinewidth=1.2)

            # Highlight Ours (RADS) with a border and larger size
            is_ours = "Ours" in methods[i]
            sc = ax.scatter(x, y, color=colors[i], marker=marker,
                            s=250 if is_ours else 120,
                            edgecolor='black' if is_ours else 'none',
                            linewidth=1.5, zorder=10, label=methods[i])
            
            if row == 0:
                scatter_handles.append(sc)

        # Formatting
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Axis Limits
        if dataset_name == "Webster":
            ax.set_xlim(0.3, 0.95)
        else:
            ax.set_xlim(0.6, 0.95)
        
        # Zoom out Y axis slightly
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        
        if metric == "clip":
             ax.set_ylim(0.1, 0.4)
        else:
             ax.set_ylim(ymin - 0.15 * yrange, ymax + 0.15 * yrange)

        # Remove ticks.
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Set the colors to be black.
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    save_name_png = output_filename + ".png"
    save_name_pdf = output_filename + ".pdf"
    plt.savefig(save_name_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_name_pdf, bbox_inches='tight')
    print(f"Saved {save_name_png} and {save_name_pdf}")

# 4. Run for each dataset
plot_dataset("Webster", full_data["Webster"], "pareto_frontier_webster", "Webster Dataset")
plot_dataset("MemBench", full_data["MemBench"], "pareto_frontier_membench", "MemBench Dataset")