import pandas as pd
from collections import Counter
import os

# Define input file paths for each APT group
files = {
    "DarkHotel": "darkhotel_vectors.csv",
    "Equation": "equation_vectors.csv",
    "Winnti": "winnti_vectors.csv",
    "APT29": "apt29_vectors.csv",
    "Gorgon": "gorgon_vectors.csv"
}

# Dictionary to store top n-grams per group
top_ngrams_by_group = {}

# Extract top 20 n-grams for each group
for group_name, path in files.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    df = pd.read_csv(path)
    ngram_cols = [col for col in df.columns if col not in ["filename", "label"]]

    total_counts = Counter()
    for _, row in df[ngram_cols].iterrows():
        for ng, count in row.items():
            total_counts[ng] += int(count)

    top_ngrams_by_group[group_name] = total_counts.most_common(20)

# Convert results to a DataFrame
result_df = pd.DataFrame({
    group: [f"{ng} ({count})" for ng, count in top_ngrams]
    for group, top_ngrams in top_ngrams_by_group.items()
})

# Save result to CSV
result_df.to_csv("top20_ngrams_per_group.csv", index=False)
print("Saved: top20_ngrams_per_group.csv")
