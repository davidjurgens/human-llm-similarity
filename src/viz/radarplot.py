import numpy as np
import matplotlib.pyplot as plt
from math import pi

def radar_plot(categories, datasets, errors=None, title="Awesome Metrics"):
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))

    # Color palette - vibrant colors with good contrast
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    # Calculate the offset angle
    offset_angle = 2 * pi / (N * 20)  # Adjust this value to increase/decrease offset


    # Plot each dataset
    for i, (label, values) in enumerate(datasets.items()):
        values += values[:1]  # Repeat the first value to close the polygon
        
        # Calculate offset angles for this dataset
        dataset_angles = [angle + (i - len(datasets) / 2 + 0.5) * offset_angle for angle in angles]
        
        ax.plot(dataset_angles, values, 'o-', linewidth=3, color=colors[i % len(colors)], label=label)
        
        ax.fill(dataset_angles, values, alpha=0.08, color=colors[i % len(colors)])

        # Add error bars if provided
        if errors and label in errors:
            err = errors[label] + errors[label][:1]
            for j in range(len(values)):
                ax.errorbar(dataset_angles[j], values[j], yerr=err[j], fmt='none', 
                            ecolor=colors[i % len(colors)], capsize=6, capthick=4, elinewidth=3.5, alpha=0.8)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')

    # Set y-axis labels
    ax.set_rlabel_position(0)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    y_labels = ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#333333", fontsize=15)
    ax.set_ylim(0, 1.1)

    # Add subtle gridlines with increased opacity and thickness
    ax.grid(color='#C0C0C0', linestyle='--', linewidth=1, alpha=1)

    # Remove spines
    ax.spines['polar'].set_visible(False)

    # Add a legend with a semi-transparent background
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=14)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_facecolor('#F0F0F0')

    # Set background color for the plot area
    ax.set_facecolor('#F7FBFF')  # Very light blue color

    # Add title with adjusted position
    plt.title(title, fontsize=22, fontweight='bold', pad=30, color='#333333')

    # Adjust layout and display
    plt.tight_layout()

    return fig, ax

# Sample data
categories = ['Lexical', 'Syntax', 'Semantic', 'Style', 'Pragmatics', 'End']
datasets = {
    'Llama-3.1 8B': [0.8, 0.7, 0.6, 0.9, 0.8, 0.7],
    'Mixtral': [0.7, 0.9, 0.5, 0.8, 0.6, 0.8],
    'Gemma': [0.6, 0.8, 0.7, 0.7, 0.9, 0.6]
}
errors = {
    'Llama-3.1 8B': [0.08, 0.04, 0.03, 0.03, 0.02, 0.05],
    'Mixtral': [0.02, 0.09, 0.03, 0.02, 0.04, 0.06],
    'Gemma': [0.03, 0.06, 0.05, 0.03, 0.08, 0.04]
}

fig, ax = radar_plot(categories, datasets, errors)
plt.show()