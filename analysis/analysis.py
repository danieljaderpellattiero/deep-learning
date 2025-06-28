from matplotlib.ticker import MaxNLocator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define data for all experiments
data = {
    "Experiment": [
        "Baseline: Zero-shot", "Baseline: TimeSformer",
        "Input: Video", "Input: +Audio", "Input: +Metadata", "Input: +All",
        "Video Aug: 0.01", "Video Aug: 0.05", "Video Aug: 0.1", "Video Aug: 0.5", "Video Aug: 1.0",
        "Meta Aug: +Meta", "Meta Aug: +0.1", "Meta Aug: +0.5", "Meta Aug: +1.0",
        "Arch: MLP-3", "Arch: MLP-4", "Arch: MLP-5", "Arch: Attention"
    ],
    "Category": [
        "Baseline", "Baseline",
        "Input Features", "Input Features", "Input Features", "Input Features",
        "Video Aug", "Video Aug", "Video Aug", "Video Aug", "Video Aug",
        "Meta Aug", "Meta Aug", "Meta Aug", "Meta Aug",
        "Architecture", "Architecture", "Architecture", "Architecture"
    ],
    "Accuracy": [
        55.6, 64.0,
        72.8, 73.9, 81.1, 79.4,
        72.2, 72.8, 73.3, 75.0, 70.0,
        81.1, 80.0, 80.0, 82.2,
        80.6, 81.1, 78.3, 79.4
    ]
}

df = pd.DataFrame(data)

# Create a plot
plt.figure(figsize=(14, 8))
sns.barplot(x="Accuracy", y="Experiment", hue="Category", data=df, dodge=False)
plt.title("Video Classification Validation Accuracy by Experiment", fontsize=16)
plt.xlabel("Accuracy (%)")
plt.ylabel("Experiment")
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xlim(50, 90)
plt.tight_layout()

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig("/home/hwanseok/20251R0136COSE47400/analysis/accuracy_plot.png")
