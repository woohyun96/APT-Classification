import pandas as pd
from collections import Counter, defaultdict

# Set parameters for feature selection
TOP_K = 500
GROUP_THRESHOLD_RATIO = 0.3
COMMON_GROUP_LIMIT = 4

# Merge multiple CSVs containing group-specific vectors
files = {
    "Gorgon": "gorgon_vectors.csv",
    "Equation": "equation_vectors.csv",
    "DarkHotel": "darkhotel_vectors.csv",
    "Winnti": "winnti_vectors.csv",
    "APT29": "apt29_vectors.csv"
}
dfs = [pd.read_csv(f) for f in files.values()]
merged = pd.concat(dfs, ignore_index=True).fillna(0)

X = merged.drop(columns=["filename", "label"])
y = merged["label"]

# Remove features that appear frequently across all groups
group_presence = {col: 0 for col in X.columns}
for group, path in files.items():
    df_group = pd.read_csv(path).fillna(0)
    X_group = df_group.drop(columns=["filename", "label"])
    for col in X.columns:
        if col in X_group.columns:
            ratio = (X_group[col] > 0).mean()
            if ratio >= GROUP_THRESHOLD_RATIO:
                group_presence[col] += 1

final_cols = [col for col in X.columns if group_presence[col] <= COMMON_GROUP_LIMIT]
X_filt = X[final_cols]

# Compute chi-square-like score and select top K features
cls_mean, tot_mean = defaultdict(Counter), Counter()
for lab, row in zip(y, X_filt.values):
    cls_mean[lab].update(dict(zip(X_filt.columns, row)))
for c in cls_mean.values():
    tot_mean += c

chi = Counter()
for g in X_filt.columns:
    for lab in cls_mean:
        chi[g] += abs(cls_mean[lab][g] - tot_mean[g] / len(cls_mean))

top_feats = [g for g, _ in chi.most_common(TOP_K)]
X_sel = X_filt[top_feats]

# Save the filtered dataset
out = pd.concat([merged[["filename", "label"]], X_sel], axis=1)
out.to_csv("merged_vectors_filtered.csv", index=False)
print("Saved: merged_vectors_filtered.csv")

