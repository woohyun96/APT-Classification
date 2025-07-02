import pandas as pd, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load data and extract features/labels
df = pd.read_csv("merged_vectors_filtered.csv")
X = df.drop(columns=["filename","label"])
y = df["label"]

# Perform t-SNE dimensionality reduction (3D)
X_emb = TSNE(n_components=3, perplexity=40, random_state=42).fit_transform(X)

# Initialize 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each label group with unique color
colors = plt.cm.tab10.colors
label_list = y.unique()
for i, lab in enumerate(label_list):
    pts = X_emb[y == lab]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               label=f"{lab} ({len(pts)})",
               color=colors[i % len(colors)],
               s=20, alpha=0.7)

# Set plot title
plt.title("t-SNE 3D – Gorgon / Equation / DarkHotel / Winnti / APT29\n(Chi-square Top500 features)",
          fontsize=13, pad=20)

# Add legend to upper right inside the plot area
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)

# Save plot with adjusted layout
plt.tight_layout()
plt.savefig("tsne_3d_final_adjusted.png", dpi=300, bbox_inches='tight')
print("t-SNE 3D plot saved: tsne_3d_final_adjusted.png")
