import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import platform

# Configure font settings (English font + fix for symbol display)
def set_font():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

set_font()

# Load filtered dataset
df = pd.read_csv("merged_vectors_filtered.csv")
X = df.drop(columns=["filename", "label"])
y = df["label"]

# Compute average n-gram counts per group
group_means = X.groupby(y).mean()

# Standardize n-gram values across groups
scaler = StandardScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(group_means.T).T,
    index=group_means.index,
    columns=group_means.columns
)

# Select top 30 n-grams by variance
top_vars = scaled_data.var(axis=0).sort_values(ascending=False).head(30).index
top_data = scaled_data[top_vars]

# Plot heatmap of top n-gram distributions
plt.figure(figsize=(16, 7))
sns.heatmap(top_data, cmap="YlGnBu", annot=False,
            linewidths=0.5, cbar_kws={'label': 'Average n-gram count per sample'})

plt.title("n-gram distribution by APT group (Top 30 by variance)", fontsize=14)
plt.xlabel("n-gram", fontsize=12)
plt.ylabel("APT Group", fontsize=12)

plt.tight_layout()
plt.savefig("ngram_balanced_heatmap_en.png", dpi=300)
plt.show()

print("Saved: ngram_balanced_heatmap_en.png")

