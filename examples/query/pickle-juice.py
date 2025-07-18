import argparse
import pickle
import numpy as np
from scipy.stats import ttest_ind
from data_utils import compute_distances, extract_match_nomatch

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


argparser = argparse.ArgumentParser(prog="pickle-juice")
argparser.add_argument("--picklefile", default=None, help="Saved llm results to read from.")
args = argparser.parse_args()

if args.picklefile is None:
    print("Need to provide a pickle file with the results of the llm_filter")

# Load the pickle file
with open(args.picklefile, "rb") as f:
    data = pickle.load(f)  # should be a list of dicts

print(f"Loaded {len(data)} documents from {args.picklefile}")

print(data[0]["properties"])
(e_match, e_no_match, no_batches, elems_processed) = extract_match_nomatch(data)

# Compute average distances
# avg_match_match, std_match_match = average_distance(e_match, e_match, num_pairs=50)
# avg_match_nomatch, std_match_nomatch = average_distance(e_match, e_nomatch, num_pairs=50)
# avg_nomatch_nomatch, std_nomatch_nomatch  = average_distance(e_nomatch, e_nomatch, num_pairs=50)

match_match = compute_distances(e_match, e_match, num_pairs=100)
match_nomatch = compute_distances(e_match, e_no_match, num_pairs=100)
nomatch_nomatch = compute_distances(e_no_match, e_no_match, num_pairs=100)

# Print summary stats
print(f"Match-Match Distance:     mean = {np.mean(match_match):.4f}, std = {np.std(match_match):.4f}")
print(f"Match-NoMatch Distance:   mean = {np.mean(match_nomatch):.4f}, std = {np.std(match_nomatch):.4f}")
print(f"NoMatch-NoMatch Distance: mean = {np.mean(nomatch_nomatch):.4f}, std = {np.std(nomatch_nomatch):.4f}")

# Run t-tests
tt_mm_vs_mnm = ttest_ind(match_match, match_nomatch)
tt_mnm_vs_nmnm = ttest_ind(match_nomatch, nomatch_nomatch)
tt_mm_vs_nmnm = ttest_ind(match_match, nomatch_nomatch)

# Print p-values
print("\nT-test results (p-values):")
print(f"Match-Match-Distance    vs  Match-NoMatch-Distance:   p = {tt_mm_vs_mnm.pvalue:.4f}")
print(f"Match-NoMatch-Distance  vs  NoMatch-NoMatch-Distance: p = {tt_mnm_vs_nmnm.pvalue:.4f}")
print(f"atch-Match-Distance     vs  NoMatch-NoMatch-Distance: p = {tt_mm_vs_nmnm.pvalue:.4f}")


# Combine and create labels
X = np.array(e_match + e_no_match)
y = np.array([1] * len(e_match) + [0] * len(e_no_match))

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
# clf = LogisticRegression(max_iter=1000)
# clf.fit(X_train, y_train)

# Predict and evaluate
# y_pred = clf.predict(X_test)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
# Create dummy 2D embedding data for visualization
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(e_match)


reduced = PCA(n_components=2).fit_transform(e_match)

plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis")
plt.title("Clusters of Embeddings")
plt.show()
