import os
import pandas as pd
from collections import Counter

def extract_ngrams(filepath, n=3):
    with open(filepath, "rb") as f:
        data = f.read()
    ngrams = [data[i:i+n] for i in range(len(data)-n+1)]
    return Counter(ngrams)

def main():
    folder = os.path.expanduser("~/malware_extracted/DarkHotel")
    sample_vectors = []
    all_ngrams = Counter()

    # Combine all n-grams from all samples
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        grams = extract_ngrams(path)
        all_ngrams.update(grams)

    # Use only the top 500 most common n-grams as features
    top_ngrams = [ng for ng, _ in all_ngrams.most_common(500)]

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        grams = extract_ngrams(path)
        vector = {ng.hex(): grams.get(ng, 0) for ng in top_ngrams}
        vector["filename"] = filename
        vector["label"] = "DarkHotel"  # APT group label
        sample_vectors.append(vector)

    df = pd.DataFrame(sample_vectors)
    df.to_csv("darkhotel_vectors.csv", index=False)
    print("Saved successfully: darkhotel_vectors.csv")

if __name__ == "__main__":
    main()