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

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    offset_angle = 2.5 * pi / (N * 20)
    for i, (label, values) in enumerate(datasets.items()):
        values += values[:1]
        
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
    ax.set_xticklabels([i.capitalize() for i in categories], fontsize=20, fontweight='bold')

    # Set y-axis labels
    ax.set_rlabel_position(1.0)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4])
    y_labels = ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4"], color="#333333", fontsize=25)
    ax.set_ylim(0, 0.4)

    # Add subtle gridlines with increased opacity and thickness
    ax.grid(color='#C0C0C0', linestyle='--', linewidth=1, alpha=1)

    # Remove spines
    ax.spines['polar'].set_visible(False)

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=20)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_facecolor('#F0F0F0')

    # Set background color for the plot area
    ax.set_facecolor('#F7FBFF')

    plt.tight_layout()

    return fig, ax

# Read the CSV file and process the data
data = pd.read_csv('data/cor_metrics_prompt.csv')
grouped = data.groupby(['prompt', 'category'])['cor'].mean().unstack()
prompt_means = grouped.mean(axis=1)
best_prompt = prompt_means.idxmax()
worst_prompt = prompt_means.idxmin()

cot_prompts = [5, 9, 10, 12, 17, 19, 25, 26, 28, 30, 31, 33, 46]
override_prompts = [28, 33, 35, 37, 38, 40, 41, 42, 43, 45, 47, 50, 11, 22, 29, 48, 49]
direct_prompts = [i for i in range(1, 51) 
                 if i not in cot_prompts and i not in override_prompts]

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

# Group prompts by type
prompt_types = {
    'CoT': cot_prompts,
    'Override': override_prompts,
    'Direct': direct_prompts
}

# Calculate stats for all prompts and group by prompt type
all_prompt_stats = {}
for prompt_type, prompt_list in prompt_types.items():
    type_stats = {category: [] for category in ['lexical', 'syntactic', 'semantic', 'style']}
    for prompt in prompt_list:
        prompt_data = data[data['prompt'] == f'Prompt_{prompt}']
        prompt_stats = calculate_category_stats(prompt_data)
        for category, stats in prompt_stats.items():
            type_stats[category].append(stats)
    
    # Average the stats for each category within the prompt type
    all_prompt_stats[prompt_type] = {}
    for category, stats_list in type_stats.items():
        means = [s['mean'] for s in stats_list]
        ci_lowers = [s['ci_lower'] for s in stats_list]
        ci_uppers = [s['ci_upper'] for s in stats_list]
        all_prompt_stats[prompt_type][category] = {
            'mean': np.mean(means),
            'ci_lower': np.mean(ci_lowers),
            'ci_upper': np.mean(ci_uppers)
        }

# Prepare data
categories = ['lexical', 'syntactic', 'semantic', 'style']
datasets = {
    prompt_type: [stats[cat]['mean'] for cat in categories]
    for prompt_type, stats in all_prompt_stats.items()
}
errors = {
    prompt_type: [
        (stats[cat]['mean'] - stats[cat]['ci_lower'],
         stats[cat]['ci_upper'] - stats[cat]['mean'])
        for cat in categories
    ]
    for prompt_type, stats in all_prompt_stats.items()
}

# Create the radar plot
fig, ax = radar_plot(categories, datasets, errors, title="Prompt Class Performance")

# Save the plot to pdf
fig.savefig('radarplot_prompts_class.pdf', bbox_inches='tight')
plt.show()