import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import METRICS_PATH, PLOTS_PATH, SCALES, MODELS, METRICS, DATASETS

warnings.filterwarnings("ignore", category=FutureWarning)

METRICS_FILE = os.path.join(METRICS_PATH, 'metrics.csv')
df = pd.read_csv(METRICS_FILE)

palette_colors = sns.color_palette("tab10", len(MODELS.keys()))
model_palette = {model: palette_colors[i] for i, model in enumerate(MODELS.keys())}

os.makedirs(PLOTS_PATH, exist_ok=True)

for scale in SCALES:
    os.makedirs(os.path.join(PLOTS_PATH, scale), exist_ok=True)

datasets = list(DATASETS.keys())

# Розподіл метрик по кожному датасету
for dataset in datasets:
    for scale in SCALES:
        subset = df[(df['Dataset'] == dataset) & (df['Scale'] == scale)]
        for metric in METRICS:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Model', y=metric, data=subset, palette=model_palette)
            plt.title(f'{metric} Distribution ({dataset}, {scale})')
            plt.tight_layout()
            save_path = os.path.join(PLOTS_PATH, scale, f'{dataset}_{metric}_distribution.png')
            plt.savefig(save_path)
            plt.close()

# Середні значення по кожному датасету
for dataset in datasets:
    for scale in SCALES:
        subset = df[(df['Dataset'] == dataset) & (df['Scale'] == scale)]
        mean_metrics = subset.groupby('Model')[METRICS].mean().reset_index()
        for metric in METRICS:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Model', y=metric, data=mean_metrics, palette=model_palette)
            plt.title(f'{metric} Mean ({dataset}, {scale})')
            plt.tight_layout()
            save_path = os.path.join(PLOTS_PATH, scale, f'{dataset}_{metric}_mean.png')
            plt.savefig(save_path)
            plt.close()

# Розподіл по масштабах
for scale in SCALES:
    subset = df[df['Scale'] == scale]
    for metric in METRICS:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y=metric, data=subset, palette=model_palette)
        plt.title(f'{metric} Distribution (Scale {scale})')
        plt.tight_layout()
        save_path = os.path.join(PLOTS_PATH, scale, f'ALL_{metric}_distribution.png')
        plt.savefig(save_path)
        plt.close()

# Середні значення по масштабах
for scale in SCALES:
    subset = df[df['Scale'] == scale]
    mean_metrics = subset.groupby('Model')[METRICS].mean().reset_index()
    for metric in METRICS:
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Model', y=metric, data=mean_metrics, palette=model_palette)
        plt.title(f'{metric} Mean (Scale {scale})')
        plt.tight_layout()
        save_path = os.path.join(PLOTS_PATH, scale, f'ALL_{metric}_mean.png')
        plt.savefig(save_path)
        plt.close()

print("Усі графіки побудовано")
