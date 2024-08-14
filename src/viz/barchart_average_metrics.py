import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# TODO: CHANGE THIS PLEASE - THESE ARE JUST SAMPLE FILES FOR DEBUGGING
file_paths = "data/metrics.csv"
data_frames = pd.read_csv(file_paths)

random_baseline = pd.read_csv("data/cor_metrics_random_baseline.csv")

# Filter for pearson correlation and compute average
random_baseline = random_baseline[random_baseline['corr_method'] == 'pearson']
random_baseline = random_baseline.groupby('metric')['cor'].mean().reset_index()
random_baseline['model'] = 'Random Baseline'
random_baseline['color'] = 'gray'
random_baseline['category'] = random_baseline['metric'].map({
    'word_count': 'Lexical', 'word_length': 'Lexical', 'perplexity': 'Lexical', 'typo': 'Lexical',
    'pos': 'Syntactic', 'dep_dpth': 'Syntactic', 'dep_brth': 'Syntactic', 'dep_dep_dist': 'Syntactic',
    'sbert': 'Semantic', 'liwc': 'Semantic', 'topic': 'Semantic',
    'punctuation': 'Style', 'capitalization': 'Style', 'sentiment': 'Style', 'politeness': 'Style',
    'formality': 'Style', 'toxicity': 'Style', 'readability': 'Style', 'subjectivity': 'Style', 'luar': 'Style'
})

# Ensure data_frames has the same columns as random_baseline
if 'category' not in data_frames.columns:
    data_frames['category'] = data_frames['metric'].map(random_baseline.set_index('metric')['category'])

# Concatenate the dataframes
data = pd.concat([data_frames, random_baseline]).reset_index(drop=True)



data['category'] = data['category'].str.capitalize()

data = data[data['cor'] != 0]
data = data.dropna(subset=['cor'])

color_scheme = {
    'meta': sns.color_palette("Reds_r", 3),
    'mistral': sns.color_palette("Greens_r", 3),
    'microsoft': sns.color_palette("Purples_r", 1),
    'qwen': sns.color_palette("Oranges_r", 1),
    'cohere': sns.color_palette("Blues_r", 1)
}

def assign_color(model):
    if model == 'Random Baseline':
        return 'gray'
    elif 'Llama3' in model:
        return color_scheme['meta'][0 if '3-70B' in model else 1 if '3.1-8B' in model else 2]
    elif 'Mistral' in model or 'Mixtral' in model:
        return color_scheme['mistral'][0 if '8x7B' in model else 1 if 'Large' in model else 2]
    elif 'Phi' in model:
        return color_scheme['microsoft'][0]
    elif 'Qwen' in model:
        return color_scheme['qwen'][0]
    elif 'Command-R' in model:
        return color_scheme['cohere'][0]
    else:
        return 'gray'

# Shorten model names
def shorten_model_name(name):
    replacements = {
        'Meta-Llama-3-70B-Instruct': 'Llama3-70B',
        'Meta-Llama-3.1-8B-Instruct': 'Llama3.1-8B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3.1-70B',
        'Mistral-7B-Instruct-v0.3': 'Mistral-7B',
        'Mistral-Large-Instruct': 'Mistral-Large',
        'Mixtral-8x7B-Instruct-v0.1': 'Mixtral-8x7B',
        'Phi-3-medium-4k-instruct': 'Phi-3',
        'Qwen2-72B-Instruct': 'Qwen2-72B',
        'c4ai-command-r-v01': 'Command-R'
    }
    for old, new in replacements.items():
        if old in name:
            return new
    return name

data['model'] = data['model'].str.replace('wildchat_subset_en_2k_prompting_', '')
data['model'] = data['model'].apply(shorten_model_name)
data['color'] = data['model'].apply(assign_color)


model_order = [
    'Llama3-70B', 'Llama3.1-8B', 'Llama3.1-70B',  # Meta models
    'Mistral-7B', 'Mistral-Large', 'Mixtral-8x7B',  # Mistral AI models
    'Phi-3',  # Microsoft model
    'Qwen2-72B',  # Qwen model
    'Command-R',  # Cohere model
    'Random Baseline'
]

# Ensure all models in the data are included in the order
all_models = data['model'].unique()
model_order = [m for m in model_order if m in all_models] + [m for m in all_models if m not in model_order]

sns.set(style="whitegrid")

print(data['model'].value_counts())
print(data[data['model'] == 'Random Baseline'])

# Adding F1 scores
f1_data = pd.read_csv("data/f1_model.csv")
f1_data['model'] = f1_data['model'].apply(shorten_model_name)
f1_data['color'] = f1_data['model'].apply(assign_color)
f1_data['model_order'] = f1_data['model'].map({model: i for i, model in enumerate(model_order)})
f1_data = f1_data.sort_values('model_order').drop('model_order', axis=1)


# Create the main figure and grid
fig = plt.figure(figsize=(19.2, 10))
gs = fig.add_gridspec(2, 3)

# Create seaborn plots for the four categories
categories = ['Lexical', 'Syntactic', 'Semantic', 'Style']
axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]

for i, (ax, category) in enumerate(zip(axes, categories)):
    sns.barplot(
        data=data[data['category'] == category],
        x="model",
        y="cor",
        ax=ax,
        palette=data.set_index('model')['color'].to_dict(),
        order=model_order[:-1],
        ci="sd",
        capsize=0.1,
    )
    ax.set_title(f"{category} features")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(0, 0.34)  
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}")) 
    ax.set_ylabel("Average Correlation" if i % 2 == 0 else "")

# Create F1 score plot
f1_ax = fig.add_subplot(gs[0, 2])
sns.barplot(
    data=f1_data,
    x="model",
    y="f1",
    ax=f1_ax,
    palette=f1_data.set_index('model')['color'].to_dict(),
    order=model_order[:-1],  # Exclude 'Random Baseline' for F1 scores
)

# Get the Random Baseline and Human Baseline F1 scores
random_baseline_f1 = f1_data[f1_data['model'] == 'Random Baseline']['f1'].values[0]
human_baseline_f1 = f1_data[f1_data['model'] == 'Human Baseline']['f1'].values[0]

# Add vertical line for Random Baseline
f1_ax.axhline(y=random_baseline_f1, color='gray', linestyle='--', linewidth=1.5)

# Add text for Random Baseline
f1_ax.text(len(model_order[:-1]) - 0.5, random_baseline_f1+0.005, f'Random Baseline ({random_baseline_f1:.2f})', 
           ha='right', va='bottom', color='gray', fontsize=9, rotation=0)

# Add vertical line for Human Baseline
f1_ax.axhline(y=human_baseline_f1, color='gray', linestyle='--', linewidth=1.5)

# Add text for Human Baseline
f1_ax.text(len(model_order[:-1]) - 0.5, human_baseline_f1-0.03, f'Human Baseline ({human_baseline_f1:.2f})', 
           ha='right', va='bottom', color='gray', fontsize=9, rotation=0)



f1_ax.set_title("Conversation End Prediction (F1)")
f1_ax.set_xlabel("")
f1_ax.set_ylabel("")
f1_ax.set_xticklabels(f1_ax.get_xticklabels(), rotation=45, ha="right")
f1_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

box = f1_ax.get_position()
rect = plt.Rectangle((box.x0-0.005, box.y0-0.025), box.width+100, box.height+150, fill=True, alpha=0.05, facecolor='blue')
f1_ax.figure.add_artist(rect)

# Apply consistent styling to all plots
for ax in axes + [f1_ax]:
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False)
    ax.set_ylim(0, ax.get_ylim()[1])

# Create legend
legend_ax = fig.add_subplot(gs[1, 2])
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['meta'][0], edgecolor='none', label='Meta'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['mistral'][0], edgecolor='none', label='Mistral AI'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['microsoft'][0], edgecolor='none', label='Microsoft'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['qwen'][0], edgecolor='none', label='Qwen'),
    plt.Rectangle((0,0),1,1, facecolor=color_scheme['cohere'][0], edgecolor='none', label='Cohere'),
    plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='none', label='Random Baseline'),
]
legend_ax.legend(handles=legend_elements, loc='center', ncol=2, fontsize=14, frameon=False)
legend_ax.axis('off')

plt.subplots_adjust(top=0.968, bottom=0.13, left=0.042, right=0.95, hspace=0.331, wspace=0.18)

plt.tight_layout()
plt.show()