import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd

def radar_plot(categories, datasets, errors, title="Awesome Metrics"):
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))

    # Color palette - vibrant colors with good contrast
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    # Calculate the offset angle
    offset_angle = 4 * pi / (N * 20)  # Adjust this value to increase/decrease offset

    # Plot each dataset
    for i, (label, values) in enumerate(datasets.items()):
        values += values[:1]  # Repeat the first value to close the polygon
        
        # Calculate offset angles for this dataset
        dataset_angles = [angle + (i - len(datasets) / 2 + 0.5) * offset_angle for angle in angles]
        
        ax.plot(dataset_angles, values, 'o-', linewidth=3, color=colors[i], label=label)
        
        ax.fill(dataset_angles, values, alpha=0.05, color=colors[i])

        # Add error bars
        err = errors[label]
        err += err[:1]  # Repeat the first value to close the polygon
        for j in range(len(values)):
            ax.errorbar(dataset_angles[j], values[j], yerr=[[err[j][0]], [err[j][1]]], fmt='none', 
                        ecolor=colors[i], capsize=5, capthick=3, elinewidth=2.0, alpha=0.5)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([i.capitalize() for i in categories], fontsize=14, fontweight='bold')

    # Set y-axis labels
    ax.set_rlabel_position(1.0)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    y_labels = ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"], color="#333333", fontsize=15)
    ax.set_ylim(0, 0.75)

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

data = pd.read_csv('data/cor_metrics_prompt.csv')
grouped = data.groupby(['prompt', 'category'])['cor'].mean().unstack()
prompt_means = grouped.mean(axis=1)
best_prompt = prompt_means.idxmax()
worst_prompt = prompt_means.idxmin()


human_baseline_df = pd.read_csv('data/cor_metrics_human_baseline.csv')
human_baseline = human_baseline_df['cor'].mean()

def bootstrap_mean_ci(data, n_boot=1000, ci=95):
    boot_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    ci_lower, ci_upper = np.percentile(boot_means, [(100-ci)/2, 100-(100-ci)/2])
    return np.mean(data), ci_lower, ci_upper

def calculate_category_stats(prompt_data):
    categories = prompt_data['category'].unique()
    results = {}
    for category in categories:
        category_data = prompt_data[prompt_data['category'] == category]['cor']
        mean, ci_lower, ci_upper = bootstrap_mean_ci(category_data)
        results[category] = {'mean': mean, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
    return results

best_prompt_data = data[data['prompt'] == best_prompt]
worst_prompt_data = data[data['prompt'] == worst_prompt]

best_prompt_stats = calculate_category_stats(best_prompt_data)
worst_prompt_stats = calculate_category_stats(worst_prompt_data)
human_baseline_stats = calculate_category_stats(human_baseline_df)

f1_prompt_data = pd.read_csv('data/f1_prompt.csv')

best_prompt_f1 = f1_prompt_data[f1_prompt_data['prompt'] == best_prompt]['f1'].values[0]
worst_prompt_f1 = f1_prompt_data[f1_prompt_data['prompt'] == worst_prompt]['f1'].values[0]

human_baseline_f1 = 0.5 

categories = list(best_prompt_stats.keys()) + ['F1']


datasets = {
    'Best Prompt': [best_prompt_stats[cat]['mean'] for cat in categories[:-1]] + [best_prompt_f1],
    'Worst Prompt': [worst_prompt_stats[cat]['mean'] for cat in categories[:-1]] + [worst_prompt_f1],
    'Human Baseline': [human_baseline_stats[cat]['mean'] for cat in categories[:-1]] + [human_baseline_f1]
}

errors = {
    'Best Prompt': [(best_prompt_stats[cat]['mean'] - max(best_prompt_stats[cat]['ci_lower'], 0),
                     best_prompt_stats[cat]['ci_upper'] - best_prompt_stats[cat]['mean']) for cat in categories[:-1]] + [(0, 0)],
    'Worst Prompt': [(worst_prompt_stats[cat]['mean'] - max(worst_prompt_stats[cat]['ci_lower'], 0),
                      worst_prompt_stats[cat]['ci_upper'] - worst_prompt_stats[cat]['mean']) for cat in categories[:-1]] + [(0, 0)],
    'Human Baseline': [(human_baseline_stats[cat]['mean'] - max(human_baseline_stats[cat]['ci_lower'], 0),
                       human_baseline_stats[cat]['ci_upper'] - human_baseline_stats[cat]['mean']) for cat in categories[:-1]] + [(0, 0)]
}

fig, ax = radar_plot(categories, datasets, errors, title=f"Best ({best_prompt}) vs Worst ({worst_prompt}) vs Human Baseline Prompts")
plt.show()