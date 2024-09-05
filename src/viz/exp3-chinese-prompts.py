import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('data/chinese/cor_metrics_prompt_chinese.csv')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", font_scale=1.2)

fig, ax = plt.subplots(figsize=(18, 10))

category_order = ['lexical', 'style', 'semantic']
metric_order = ['log_word_count', 'perplexity', 'alnum_ratio', 
                'punctuation', 'sentiment', 'toxicity',
                'sbert_embedding']

color_scheme = {
    'Prompt_19': '#FF6B6B',
    'Prompt_11': '#4ECDC4',
    'Prompt_15': '#45B7D1'
}

prompt_mapping = {
    'Prompt_19': 'Top CoT Prompt',
    'Prompt_11': 'Top Override Prompt',
    'Prompt_15': 'Top Direct Prompt'
}

data['prompt'] = data['prompt'].astype(str)
data['prompt'] = data['prompt'].map(prompt_mapping)

prompt_order = ['Top CoT Prompt', 'Top Override Prompt', 'Top Direct Prompt']
color_scheme = {new_name: color for old_name, new_name in prompt_mapping.items() for old_color, color in color_scheme.items() if old_name == old_color}

sns.barplot(x='metric', y='cor', hue='prompt', data=data, ax=ax, order=metric_order,
            hue_order=prompt_order, palette=color_scheme, width=0.6)

ax.set_ylabel('Correlation', fontsize=30, labelpad=10)
ax.set_xlabel('')  # Remove "Metrics" label
ax.tick_params(axis='both', which='major', labelsize=30)

plt.xticks(rotation=45, ha='right')

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

ax.set_ylim(0, 0.29)
ax.yaxis.set_ticks(np.arange(0, 0.26, 0.05))

plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', 
           fontsize=30, title_fontsize=30, ncol=3)

category_positions = {
    'Lexical': 1,
    'Style': 4,
    'Semantic': 6
}

category_bounds = [0, 3, 6, 7]
for bound in category_bounds[1:-1]:
    ax.axvline(x=bound - 0.5, color='gray', linestyle='--', alpha=0.5)

for category, position in category_positions.items():
    ax.text(position, ax.get_ylim()[1]-0.01, category, ha='center', va='bottom', fontweight='bold', fontsize=30)

plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95)

sns.despine()

plt.tight_layout()
plt.savefig('exp3-chinese-prompts.pdf', bbox_inches='tight')
plt.show()