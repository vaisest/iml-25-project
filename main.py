import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

train_path = "data/train.csv"
test_path = "data/test.csv"
sample_path = "data/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("Loaded train, test and sample submission")
print(f"   Train shape: {train.shape}, Test shape: {test.shape}, Sample shape: {sample.shape}")

train["class2"] = (train["class4"] != "nonevent").astype(int)

print("\nTarget distribution (class4):")
print(train["class4"].value_counts())
print("\nTarget distribution (class2):")
print(train["class2"].value_counts())

drop_cols = ["class4", "class2"]
if "date" in train.columns:
    drop_cols.append("date")

feature_cols = [c for c in train.columns if c not in drop_cols]

X = train[feature_cols].values
y_bin = train["class2"].values
y_multi = train["class4"].values
X_test = test[feature_cols].values

print(f"\nUsing {len(feature_cols)} features for modelling")

pipe_bin = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

pipe_multi = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

X_train_multi, X_val_multi, y_train_multi, y_val_multi = train_test_split(
    X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

print("\nTraining binary model (event vs nonevent)...")
pipe_bin.fit(X_train_bin, y_train_bin)
y_val_pred_bin = pipe_bin.predict(X_val_bin)
y_val_proba_bin = pipe_bin.predict_proba(X_val_bin)[:, 1]

acc_bin = accuracy_score(y_val_bin, y_val_pred_bin)
print(f"   Binary validation accuracy: {acc_bin:.4f}")
print("   Binary classification report:")
print(classification_report(y_val_bin, y_val_pred_bin))

print("\nTraining multiclass model (Ia / Ib / II / nonevent)...")
pipe_multi.fit(X_train_multi, y_train_multi)
y_val_pred_multi = pipe_multi.predict(X_val_multi)

acc_multi = accuracy_score(y_val_multi, y_val_pred_multi)
print(f"   Multiclass validation accuracy: {acc_multi:.4f}")
print("   Multiclass classification report:")
print(classification_report(y_val_multi, y_val_pred_multi))

def binary_perplexity(y_true, proba_event):
    y_true = np.array(y_true)
    proba_event = np.array(proba_event)
    correct_proba = np.where(y_true == 1, proba_event, 1.0 - proba_event)
    correct_proba = np.clip(correct_proba, 1e-15, 1.0)
    return np.exp(-np.mean(np.log(correct_proba)))

perpl = binary_perplexity(y_val_bin, y_val_proba_bin)
print(f"\nBinary perplexity on validation: {perpl:.4f}")

print("\n5-fold cross-validation for binary model...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_bin = cross_val_score(pipe_bin, X, y_bin, cv=cv, scoring="accuracy", n_jobs=-1)
print("   CV accuracy (binary):", np.round(cv_scores_bin, 4))
print(f"   Mean CV accuracy (binary): {cv_scores_bin.mean():.4f}")

print("\n5-fold cross-validation for multiclass model...")
cv_scores_multi = cross_val_score(pipe_multi, X, y_multi, cv=cv, scoring="accuracy", n_jobs=-1)
print("   CV accuracy (multiclass):", np.round(cv_scores_multi, 4))
print(f"   Mean CV accuracy (multiclass): {cv_scores_multi.mean():.4f}")

print("\nTraining final models on full data and predicting test set...")

pipe_bin.fit(X, y_bin)
test_proba_bin = pipe_bin.predict_proba(X_test)[:, 1]

pipe_multi.fit(X, y_multi)
test_pred_multi = pipe_multi.predict(X_test)

submission = sample.copy()
if "class4" not in submission.columns:
    submission["class4"] = ""
if "p" not in submission.columns:
    submission["p"] = 0.0

submission["class4"] = test_pred_multi
submission["p"] = test_proba_bin

print("\nPreview of final submission (id, class4, p):")
print(submission.head())

submission_path = "submission_rf_baseline.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")
print("Ready to upload to Kaggle with columns: id, class4, p")
