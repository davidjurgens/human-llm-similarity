import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# TODO: CHANGE THIS PLEASE - THESE ARE JUST SAMPLE FILES FOR DEBUGGING
file_paths = "data/metrics.csv"
data_frames = pd.read_csv(file_paths)

random_baseline = pd.read_csv("data/cor_metrics_random_baseline.csv")

data = pd.concat([data_frames]).reset_index(drop=True)

data['category'] = data['category'].str.capitalize()

data = data[data['cor'] != 0]
data = data.dropna(subset=['cor'])

color_scheme = {
    'meta': sns.color_palette("Reds_r", 6),
    'mistral': sns.color_palette("Greens_r", 6),
    'microsoft': sns.color_palette("Purples_r", 1),
    'qwen': sns.color_palette("Oranges_r", 1),
    'cohere': sns.color_palette("Blues_r", 1)
}

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

data['model'] = data['model'].str.replace('wildchat_subset_en_2k_prompting_', '')
data['model'] = data['model'].apply(shorten_model_name)
data['color'] = data['model'].apply(assign_color)


model_order = [
    'Llama3.1-8B', 'Llama3.1-70B', 'Llama3-70B',  # Meta models
    'Mistral-7B', 'Mixtral-8x7B', 'Mistral-Large-123B',  # Mistral AI models
    'Phi-3-14B',  # Microsoft model
    'Qwen2-72B',  # Qwen model
    'Command-R-35B',  # Cohere model
    'Human Baseline'
]

all_models = data['model'].unique()
model_order = [m for m in model_order if m in all_models] + [m for m in all_models if m not in model_order]

sns.set(style="whitegrid")

print(data['model'].value_counts())
print(data[data['model'] == 'Random Baseline'])

barplot = sns.catplot(
    data=data,
    x="model",
    y="cor",
    col="category",
    kind="bar",
    height=8,
    aspect=1.2,
    palette=data.set_index('model')['color'].to_dict(),
    order=model_order,
    col_wrap=2,
    sharex=False,
    ci="sd",
    n_boot=1000,
    errwidth=1,
    capsize=0.1,
)

barplot.fig.set_size_inches(19.2, 10) 

barplot.set_titles("{col_name} features", fontsize=20)
barplot.set_axis_labels("", "Average Correlation", fontsize=16)
barplot.set_xticklabels(rotation=45, ha="right", fontsize=16)

for ax in barplot.axes.flat:
    current_title = ax.get_title()
    ax.set_title(current_title, fontsize=24)
    
    ax.tick_params(axis='y', labelsize=16)

legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['meta'][0], edgecolor='none', label='Meta'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['mistral'][0], edgecolor='none', label='Mistral AI'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['microsoft'][0], edgecolor='none', label='Microsoft'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['qwen'][0], edgecolor='none', label='Qwen'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['cohere'][0], edgecolor='none', label='Cohere'),
    plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='none', label='Annotator Baseline'),
]
barplot.fig.legend(handles=legend_elements, loc='right', ncol=1, title=None, fontsize=20, frameon=False)

plt.tight_layout()

barplot.fig.subplots_adjust(top=0.95, bottom=0.18, left=0.05, right=0.80, hspace=0.90, wspace=0.05)

plt.savefig('barchart_average_metrics.pdf', bbox_inches='tight')

plt.show()