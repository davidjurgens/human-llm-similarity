import numpy as np
import matplotlib.pyplot as plt
from math import pi

def radar_plot(categories, datasets, errors=None, title="Some Title"):
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

    # Color palette
    colors = ['#1E88E5', '#FFC107', '#4CAF50', '#9C27B0', '#FF5722']

    # Plot each dataset
    for i, (label, values) in enumerate(datasets.items()):
        values += values[:1]  # Repeat the first value to close the polygon
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)], label=label)
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

        # Add error bars if provided
        if errors and label in errors:
            err = errors[label] + errors[label][:1]
            for j in range(len(values)):
                ax.errorbar(angles[j], values[j], yerr=err[j], fmt='none', 
                            ecolor=colors[i % len(colors)], capsize=5, capthick=1, elinewidth=1, alpha=0.5)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

    # Set y-axis labels
    ax.set_rlabel_position(0)
    ax.set_rticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="#333333", fontsize=10)
    ax.set_ylim(0, 6)

    # Add subtle gridlines
    ax.grid(color='#CCCCCC', linestyle='--', linewidth=0.5, alpha=0.7)

    # Remove spines
    ax.spines['polar'].set_visible(False)

    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Set background color
    ax.set_facecolor('#F5F5F5')

    # Add title
    plt.title(title, fontsize=20, fontweight='bold', pad=20, color='#333333')

    # Adjust layout and display
    plt.tight_layout()
    return fig, ax

# Sample data
categories = ['Lexical', 'Syntax', 'Semantic', 'Style', 'Pragmatics']
datasets = {
    'Llama-3.1 8B': [4, 3, 2, 5, 4],
    'Mixtral': [3, 5, 1, 4, 2],
    'Gemma': [2, 4, 3, 3, 5]
}
errors = {
    'Llama-3.1 8B': [0.5, 0.4, 0.3, 0.6, 0.5],
    'Mixtral': [0.3, 0.5, 0.2, 0.4, 0.3],
    'Gemma': [0.4, 0.3, 0.5, 0.3, 0.6]
}

fig, ax = radar_plot(categories, datasets, errors)
plt.show()