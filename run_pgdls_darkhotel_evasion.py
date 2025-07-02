#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBF-SVM (train/test split + 10-fold CV) + PGD-LS on ALL DarkHotel samples
"""

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from secml.array import CArray
from secml.data import CDataset
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.data.splitter import CDataSplitterKFold
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks.evasion import CAttackEvasionPGDLS

# Load and split the dataset; normalize features
logging.getLogger("secml").setLevel(logging.ERROR)

df = pd.read_csv("merged_vectors_filtered.csv")
X_np = df.drop(columns=["filename", "label"]).values.astype(float)
y_str = df["label"].values

le = LabelEncoder()
y_np = le.fit_transform(y_str)

idx_all = np.arange(len(X_np))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_np, y_np, idx_all, test_size=0.2, random_state=42, stratify=y_np)

train_set = CDataset(X_train, y_train)
test_set = CDataset(X_test, y_test)

norm = CNormalizerMinMax().fit(train_set.X)
train_set.X = norm.transform(train_set.X)
test_set.X = norm.transform(test_set.X)

print(f"Dataset | Train {train_set.num_samples}  Test {test_set.num_samples}  "
      f"Features {train_set.num_features}  Classes {len(le.classes_)}")

# Perform hyperparameter search and train the RBF-SVM classifier
kernel = CKernelRBF()
clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=kernel)

param_grid = {"C": [0.1, 1, 10, 50, 100], "kernel.gamma": [1e-3, 1e-2, 1e-1, 1, 10]}
split = CDataSplitterKFold(num_folds=10, random_state=42)

print("Hyper-parameter search (10-fold CV)…")
best = clf.estimate_parameters(train_set, param_grid, split,
                               metric="accuracy", perf_evaluator="xval")
print("Best params:", best)

clf.fit(train_set.X, train_set.Y)

metric = CMetricAccuracy()
print(f"Train acc {metric.performance_score(train_set.Y, clf.predict(train_set.X)):.3f}  "
      f"Test acc {metric.performance_score(test_set.Y, clf.predict(test_set.X)):.3f}")

# Compute and save confusion matrix from test set predictions
y_pred_test = le.inverse_transform(clf.predict(test_set.X).tondarray())
y_true_test = le.inverse_transform(test_set.Y.tondarray())
cm_test = confusion_matrix(y_true_test, y_pred_test, labels=le.classes_)
cm_test_df = pd.DataFrame(cm_test,
                          index=[f"True_{c}" for c in le.classes_],
                          columns=[f"Pred_{c}" for c in le.classes_])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=300)
plt.close()
print("confusion_matrix_test.png saved")

# Perform PGD-LS adversarial attack on all DarkHotel samples
attack = CAttackEvasionPGDLS(
    classifier=clf, distance="l2", dmax=0.0,
    lb=0.0, ub=1.0, double_init_ds=None, double_init=False)
attack.eta = 0.1
attack.max_iter = 100

BUDGETS = [0.8, 1.2, 1.6, 2.0, 2.5]
dh_class = "DarkHotel"
dh_idx = int(np.where(le.classes_ == dh_class)[0])

dataset_full = CDataset(norm.transform(CArray(X_np)), y_np)
idx_dh_all = np.where(dataset_full.Y.tondarray() == dh_idx)[0]

records, adv_vectors, adv_idx = [], [], []
print(f"PGD-LS on ALL DarkHotel samples ({len(idx_dh_all)})")
for idx in tqdm(idx_dh_all, desc="DarkHotel ➔ PGD-LS"):
    x0, y0 = dataset_full[idx, :].X, dataset_full[idx, :].Y
    success, l2, adv_lab = False, np.nan, -1
    for dmax in BUDGETS:
        attack.dmax = dmax
        try:
            y_adv, _, adv_ds, _ = attack.run(x0, y0)
            l2_tmp = float((adv_ds.X - x0).norm())
            if (y_adv != y0) and (l2_tmp <= dmax):
                success, l2, adv_lab = True, l2_tmp, int(y_adv.item())
                adv_vectors.append(adv_ds.X.tondarray())
                adv_idx.append(int(idx))
                break
        except Exception:
            continue
    records.append(dict(sample=int(idx), success=success, l2=l2,
                        adv_label=le.classes_[adv_lab] if adv_lab >= 0 else "attack_failed"))

pd.DataFrame(records).to_csv("evasion_results.csv", index=False)
succ_rate = np.mean([r['success'] for r in records]) * 100
mean_l2  = np.nanmean([r['l2'] for r in records if r['success']])
print(f"Evasion success {succ_rate:.1f}% | mean L2 {mean_l2:.3f}")

if adv_vectors:
    X_adv = np.vstack(adv_vectors)
    np.save("darkhotel_adv_vectors.npy", X_adv)
    (pd.DataFrame(X_adv)
        .assign(sample_idx=adv_idx)
        .to_csv("darkhotel_adv_vectors.csv", index=False))
    print(f"{len(adv_vectors)} adversarial vectors saved")

# Evaluate impact of adversarial attack on DarkHotel test samples
idx_dh_test = np.where(test_set.Y.tondarray() == dh_idx)[0]
pred_before = clf.predict(test_set.X[idx_dh_test.tolist(), :]).tondarray()
pred_before_lbl = le.inverse_transform(pred_before)

success_dict = {r['sample']: r['adv_label'] for r in records if r['success']}

idx_all = np.arange(len(X_np))
_, _, _, _, _, idx_test = train_test_split(X_np, y_np, idx_all,
                                           test_size=0.2, random_state=42, stratify=y_np)
idx_dh_test_global = idx_test[idx_dh_test]

pred_after_lbl = np.array([success_dict.get(global_idx, lbl)
                           for global_idx, lbl in zip(idx_dh_test_global, pred_before_lbl)])

acc_bef = (pred_before_lbl == dh_class).mean()
acc_aft = (pred_after_lbl == dh_class).mean()

plt.figure(figsize=(7, 5))
sns.countplot(x=pred_before_lbl, order=le.classes_, color='steelblue',
              alpha=0.6, label="Before")
sns.countplot(x=pred_after_lbl,  order=le.classes_, color='tomato',
              alpha=0.6, label="After")
plt.title(f"DarkHotel Test Samples\nAcc before {acc_bef:.2%} | "
          f"Acc after {acc_aft:.2%} | mean L2 {mean_l2:.3f}")
plt.legend(); plt.ylabel("# Samples"); plt.xlabel("Predicted Class")
plt.tight_layout(); plt.savefig("darkhotel_before_after.png", dpi=300); plt.close()
print("darkhotel_before_after.png saved")

# Plot budget-wise success and adversarial target class distribution
res_df = pd.read_csv("evasion_results.csv")
res_df['budget'] = res_df['l2'].apply(
    lambda l2: next((b for b in BUDGETS if l2 <= b), np.nan)
    if not np.isnan(l2) else np.nan)
budget_success = res_df.groupby("budget")["success"].mean() * 100

plt.figure(figsize=(6, 4))
plt.plot(budget_success.index, budget_success.values, marker='o')
plt.title("PGD-LS Success vs L2 Budget"); plt.xlabel("L2 Budget"); plt.ylabel("% success")
plt.grid(True); plt.tight_layout(); plt.savefig("plot_budget_vs_success.png", dpi=300); plt.close()

if res_df['success'].any():
    plt.figure(figsize=(6, 4))
    res_df.loc[res_df.success, 'adv_label'].value_counts().plot(kind='bar', color='tomato')
    plt.title("Attack Target Class Distribution"); plt.ylabel("# Samples"); plt.xlabel("Class")
    plt.tight_layout(); plt.savefig("plot_attack_target_distribution.png", dpi=300); plt.close()

print("All figures & files saved. Script completed.")
