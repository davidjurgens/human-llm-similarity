import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_csv('data/chinese/cor_metrics_chinese.csv')

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", font_scale=1.2)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(19, 10))

# Define the correct order of metrics within each category
category_order = ['lexical', 'style', 'semantic']
metric_order = ['log_word_count', 'perplexity', 'alnum_ratio', 
                'punctuation', 'sentiment', 'toxicity',
                'sbert_embedding']

# Create a mapping for better model names
model_name_mapping = {
    'wildchat_subset_cn_10k_Llama-3.1-70B-Instruct_metrics': 'Llama-3.1-70B',
    'wildchat_subset_cn_10k_Mistral-Large-Instruct_metrics': 'Mistral-Large-123B',
    'wildchat_subset_cn_10k_Mixtral-8x7B-Instruct_metrics': 'Mixtral-8x7B'
}

# Define color scheme
color_scheme = {
    'Llama-3.1-70B': sns.color_palette("Reds_r", 6)[3],
    'Mistral-Large-123B': sns.color_palette("Greens_r", 6)[3],
    'Mixtral-8x7B': sns.color_palette("Greens_r", 6)[2]
}

data['model'] = data['model'].map(model_name_mapping)

data['model'] = data['model'].astype(str)

model_order = ['Llama-3.1-70B', 'Mistral-Large-123B', 'Mixtral-8x7B']

sns.barplot(x='metric', y='cor', hue='model', data=data, ax=ax, order=metric_order,
            hue_order=model_order, palette=color_scheme, width=0.6)

ax.set_ylabel('Correlation', fontsize=36, labelpad=10)
ax.set_xlabel('')  # Remove "Metrics" label
ax.tick_params(axis='both', which='major', labelsize=36)

plt.xticks(rotation=45, ha='right')

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

y_max = data['cor'].max()
ax.set_ylim(0, 0.29)

plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', 
           fontsize=36, title_fontsize=36, ncol=3)

category_positions = {
    'Lexical': 1,
    'Style': 4,
    'Semantic': 6
}

# Add vertical lines to separate categories
category_bounds = [0, 3, 6, 7]
for bound in category_bounds[1:-1]:
    ax.axvline(x=bound - 0.5, color='gray', linestyle='--', alpha=0.5)

# Add category labels with custom positioning
for category, position in category_positions.items():
    ax.text(position, ax.get_ylim()[1]-0.01, category, ha='center', va='bottom', fontweight='bold', fontsize=36)

sns.despine()

plt.tight_layout()
plt.savefig('exp3-chinese-models.pdf', bbox_inches='tight')
plt.show()