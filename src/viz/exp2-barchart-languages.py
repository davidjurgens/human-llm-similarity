import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

corr_all = pd.read_csv('data/cor_metrics_multilingual_agg.csv')
categories = ['Lexical', 'Syntactic', 'Semantic', 'Style']
f1_category = 'Conv End (F1)'

sns.set_theme(style="whitegrid", font_scale=1.2)

plt.rcParams.update({'font.size': 40})  # Increase overall font size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [4, 1]})

new_palette = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# Plot the main categories
sns.barplot(data=corr_all[corr_all['category'].isin(categories)],
            x='category', y='cor', hue='language',
            hue_order=['English', 'Chinese', 'Russian'],
            palette=new_palette,
            ax=ax1)

# Plot the F1 score
sns.barplot(data=corr_all[corr_all['category'] == f1_category],
            x='category', y='cor', hue='language',
            hue_order=['English', 'Chinese', 'Russian'],
            palette=new_palette,
            ax=ax2)

# Customize the plots
for ax in [ax1, ax2]:
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0, labelsize=40)
    ax.tick_params(axis='y', labelsize=40)
    ax.legend_.remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 0.20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))

ax1.set_title('Average Correlation', fontsize=40, fontweight='bold')
ax2.set_title('F1 Score', fontsize=40, fontweight='bold')
ax1.set_ylabel('')
ax2.set_ylabel('')
ax2.set_xticklabels(["Conv End"])  # Remove x-axis text for F1 plot

# Add a common legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.01), 
           fontsize=40, frameon=False)

plt.tight_layout()
fig.subplots_adjust(top=0.78)
plt.savefig('multilingual_corr_new.pdf', bbox_inches='tight')
plt.show()