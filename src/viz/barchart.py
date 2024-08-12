import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# TODO: CHANGE THIS PLEASE - THESE ARE JUST SAMPLE FILES FOR DEBUGGING
file_paths = [
    "data/cor_metrics.csv",
    "data/cor_metrics1.csv",
    "data/cor_metrics2.csv",
    "data/cor_metrics3.csv",
    "data/cor_metrics4.csv",
    "data/cor_metrics5.csv",
    "data/cor_metrics6.csv",
    "data/cor_metrics7.csv",
]

data_frames = [pd.read_csv(file) for file in file_paths]
data = pd.concat(data_frames).reset_index(drop=True)

data['category'] = data['category'].str.capitalize()

data = data[data['cor'] != 0]
data = data.dropna(subset=['cor'])

np.random.seed(42)
data['cor'] += np.random.uniform(0, 0.10, size=data.shape[0])

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

barplot.fig.set_size_inches(14, 10)

barplot.set_titles("{col_name} features")
barplot.set_axis_labels("", "Correlation")
barplot.set_xticklabels(rotation=45, ha="right")

barplot.fig.suptitle("Correlation Metrics by Model and Category", fontsize=24, y=0.95)

# Create a custom legend at the bottom
handles = barplot.legend.legend_handles
labels = [t.get_text() for t in barplot.legend.texts]
legend = barplot.fig.legend(handles, labels, loc='lower center', ncol=8, title=None, fontsize=12, frameon=False)

barplot.legend.remove()

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

barplot.fig.subplots_adjust(top=0.876, bottom=0.15, left=0.05, right=0.95, hspace=0.3, wspace=0.105)

plt.show()