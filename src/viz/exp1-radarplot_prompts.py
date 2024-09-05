import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd

def radar_plot(categories, datasets, errors, title="Awesome Metrics"):
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

    # Color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    offset_angle = 2.5 * pi / (N * 20)

    for i, (label, values) in enumerate(datasets.items()):
        values += values[:1]
        
        dataset_angles = [angle + (i - len(datasets) / 2 + 0.5) * offset_angle for angle in angles]
        
        ax.plot(dataset_angles, values, 'o-', linewidth=3, color=colors[i], label=label)
        
        ax.fill(dataset_angles, values, alpha=0.05, color=colors[i])

        # Add error bars
        err = errors[label]
        err += err[:1]
        for j in range(len(values)):
            ax.errorbar(dataset_angles[j], values[j], yerr=[[err[j][0]], [err[j][1]]], fmt='none', 
                        ecolor=colors[i], capsize=5, capthick=3, elinewidth=2.0, alpha=0.5)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([i.capitalize() for i in categories], fontsize=20, fontweight='bold')

    # Set y-axis labels
    ax.set_rlabel_position(1.0)
    ax.set_rticks([0.1, 0.2, 0.3])
    y_labels = ax.set_yticklabels(["0.1", "0.2", "0.3"], color="#333333", fontsize=25)
    ax.set_ylim(0, 0.4)

    # Add subtle gridlines with increased opacity and thickness
    ax.grid(color='#C0C0C0', linestyle='--', linewidth=1, alpha=1)

    # Remove spines
    ax.spines['polar'].set_visible(False)

    # Add a legend with a semi-transparent background
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=20)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_facecolor('#F0F0F0')

    ax.set_facecolor('#F7FBFF')

    plt.tight_layout()

    return fig, ax

# Read the CSV file and process the data
data = pd.read_csv('data/cor_metrics_prompt.csv')
grouped = data.groupby(['prompt', 'category'])['cor'].mean().unstack()
prompt_means = grouped.mean(axis=1)
best_prompt = prompt_means.idxmax()
worst_prompt = prompt_means.idxmin()

print(best_prompt)
print(worst_prompt)

# Annotator Baseline
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


# Prepare data for radar plot
categories = list(best_prompt_stats.keys())
datasets = {
    'Best Prompt': [best_prompt_stats[cat]['mean'] for cat in categories],
    'Worst Prompt': [worst_prompt_stats[cat]['mean'] for cat in categories],
    'Annotator Baseline': [human_baseline_stats[cat]['mean'] for cat in categories]
}
errors = {
    'Best Prompt': [(best_prompt_stats[cat]['mean'] - max(best_prompt_stats[cat]['ci_lower'], 0),
                     best_prompt_stats[cat]['ci_upper'] - best_prompt_stats[cat]['mean']) for cat in categories],
    'Worst Prompt': [(worst_prompt_stats[cat]['mean'] - max(worst_prompt_stats[cat]['ci_lower'], 0),
                      worst_prompt_stats[cat]['ci_upper'] - worst_prompt_stats[cat]['mean']) for cat in categories],
    'Annotator Baseline': [(human_baseline_stats[cat]['mean'] - max(human_baseline_stats[cat]['ci_lower'], 0),
                       human_baseline_stats[cat]['ci_upper'] - human_baseline_stats[cat]['mean']) for cat in categories]
}

# Create the radar plot
fig, ax = radar_plot(categories, datasets, errors, title=f"Best ({best_prompt}) vs Worst ({worst_prompt}) vs Annotator Baseline Prompts")

# Save the plot to pdf
fig.savefig('radarplot_prompts_test.pdf', bbox_inches='tight')
plt.show()