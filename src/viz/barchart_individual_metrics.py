import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

file_paths = [
    "data/metrics.csv",
]

data_frames = [pd.read_csv(file) for file in file_paths]
data = pd.concat(data_frames).reset_index(drop=True)

data['category'] = data['category'].str.capitalize()

data = data[data['cor'] != 0]
data = data.dropna(subset=['cor'])

data['model'] = data['model'].str.replace('wildchat_subset_en_2k_prompting_', '')

avg_data = data.groupby(['category', 'model'], as_index=False)['cor'].mean()
avg_data['metric'] = 'average'  # Add a metric to identify these rows

data = pd.concat([data, avg_data], ignore_index=True)

sns.set(style="whitegrid")

barplot = sns.catplot(
    data=data,
    x="metric",
    y="cor",
    hue="model",
    col="category",
    kind="bar",
    height=8,
    aspect=1.2,
    palette="husl",
    col_wrap=2,
    sharex=False,
)

for idx, ax in enumerate(barplot.axes.flat):
    if idx == 0:
        ax.axvspan(3.55, 4.45, color='lightblue', alpha=0.25)
        # Add text to the top of the span
        ax.text(4.0, 0.225, 'Average', ha='center', va='center', color='gray', fontsize=10)
    elif idx == 1:
        ax.axvspan(3.55, 4.45, color='lightblue', alpha=0.25)
        ax.text(4.0, 0.225, 'Average', ha='center', va='center', color='gray', fontsize=10)
    elif idx == 2:
        ax.axvspan(2.55, 3.45, color='lightblue', alpha=0.25)
        ax.text(3.0, 0.225, 'Average', ha='center', va='center', color='gray', fontsize=10)
    else:
        ax.axvspan(8.55, 9.45, color='lightblue', alpha=0.25)
        ax.text(9.0, 0.225, 'Average', ha='center', va='center', color='gray', fontsize=10)


barplot.fig.set_size_inches(19.2, 10)

barplot.set_titles("{col_name} features")
barplot.set_axis_labels("", "Correlation")
barplot.set_xticklabels(rotation=45, ha="right")

for ax in barplot.axes.flat:
    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()
    if 'average' in [label.get_text() for label in xticklabels]:
        ax.set_xticks(xticks[:-1])
        ax.set_xticklabels(xticklabels[:-1])

barplot.fig.suptitle("Correlation Metrics by Model and Category", fontsize=24, y=0.95)

# Create a custom legend at the bottom
handles = barplot.legend.legend_handles
labels = [t.get_text() for t in barplot.legend.texts]
legend = barplot.fig.legend(handles, labels, loc='lower center', ncol=9, title=None, fontsize=9, frameon=False)

barplot.legend.remove()

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

barplot.fig.subplots_adjust(top=0.876, bottom=0.15, left=0.05, right=0.95, hspace=0.36, wspace=0.105)

plt.show()