import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the data
data = pd.read_csv("data/f1_model.csv")

color_scheme = {
    'meta': sns.color_palette("Reds_r", 6),
    'mistral': sns.color_palette("Greens_r", 6),
    'microsoft': sns.color_palette("Purples_r", 1),
    'qwen': sns.color_palette("Oranges_r", 1),
    'cohere': sns.color_palette("Blues_r", 1)
}

# Shorten model names (if not already done)
def shorten_model_name(name):
    replacements = {
        'Meta-Llama-3-70B-Instruct': 'Llama3-70B',
        'Meta-Llama-3.1-8B-Instruct': 'Llama3.1-8B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3.1-70B',
        'Mistral-7B-Instruct-v0.3': 'Mistral-7B',
        'Mistral-Large-Instruct': 'Mistral-Large-123B',
        'Mixtral-8x7B-Instruct-v0.1': 'Mixtral-8x7B',
        'Phi-3-medium-4k-instruct': 'Phi-3-14B',
        'Qwen2-72B-Instruct': 'Qwen2-72B',
        'c4ai-command-r-v01': 'Command-R-35B'
    }
    for old, new in replacements.items():
        if old in name:
            return new
    return name

data['model'] = data['model'].apply(shorten_model_name)

# Define color assignment function
def assign_color(model):
    if model == 'Random Baseline':
        return 'gray'
    elif 'Llama3' in model:
        return color_scheme['meta'][1 if '3-70B' in model else 2 if '3.1-8B' in model else 3]
    elif 'Mistral' in model or 'Mixtral' in model:
        return color_scheme['mistral'][1 if '8x7B' in model else 2 if 'Large' in model else 3]
    elif 'Phi' in model:
        return color_scheme['microsoft'][0]
    elif 'Qwen' in model:
        return color_scheme['qwen'][0]
    elif 'Command-R' in model:
        return color_scheme['cohere'][0]
    else:
        return 'gray'

# Define model order
model_order = [
    'Llama3.1-8B', 'Llama3.1-70B', 'Llama3-70B',  # Meta models
    'Mistral-7B', 'Mixtral-8x7B', 'Mistral-Large-123B',  # Mistral AI models
    'Phi-3-14B',  # Microsoft model
    'Qwen2-72B',  # Qwen model
    'Command-R-35B',  # Cohere model
]

# Set up the plot style
sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# Create the bar plot
ax = sns.barplot(x='model', y='f1', data=data[data['model'].isin(model_order)],
                 order=model_order, palette=[assign_color(m) for m in model_order],
                 edgecolor='white', linewidth=1.5)



# Customize the plot
plt.xlabel("")
plt.ylabel("Binary F1", fontsize=26)
plt.xticks(rotation=45, ha='right', fontsize=36)

# Set y-axis limits and steps
plt.ylim(0, 0.74)
plt.yticks(np.arange(0, 0.71, 0.1))

# Add value labels on top of each bar
for i, model in enumerate(model_order):
    v = data[data['model'] == model]['f1'].values[0]
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=26)

# Add Random and Human Baseline as horizontal lines with text
random_baseline = data[data['model'] == 'Random Baseline']['f1'].values[0]
human_baseline = data[data['model'] == 'Human Baseline']['f1'].values[0]

plt.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axhline(y=human_baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

plt.text(ax.get_xlim()[1]-0.05, random_baseline-0.1, f'Random Baseline ({random_baseline:.2f})', 
         ha='right', va='bottom', color='gray', fontweight='bold', fontsize=26)
plt.text(ax.get_xlim()[1]-0.05, human_baseline+0.009, f'Annotator Baseline ({human_baseline:.2f})', 
         ha='right', va='bottom', color='gray', fontweight='bold', fontsize=26)

# Customize grid
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Remove top and right spines
sns.despine(top=True, right=True)


# Increase size of y-axis tick font
ax.tick_params(axis='both', which='major', labelsize=26)

legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['meta'][0], edgecolor='none', label='Meta'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['mistral'][0], edgecolor='none', label='Mistral AI'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['microsoft'][0], edgecolor='none', label='Microsoft'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['qwen'][0], edgecolor='none', label='Qwen'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['cohere'][0], edgecolor='none', label='Cohere'),
]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=26, frameon=False)


plt.subplots_adjust(top=0.9, bottom=0.2, left=0.05, right=0.85, hspace=0.2, wspace=0.2)


plt.tight_layout()
plt.savefig('barchart_model_f1_exp1.pdf', bbox_inches='tight')
plt.show()