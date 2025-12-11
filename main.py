import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from kaggle_scoring_metric import score
from analysis import (
    analyse_pca,
    plot_training_data,
    plot_feature_importances,
    plot_classification_report,
    plot_test_set_results,
)


def predict(model, X: pd.DataFrame, label_encoder=None):
    """Returns predictions as a DataFrame.

    class4: Predicted class (nonevent, Ia, Ib or II)
    p: Probability of an event occuring, sum of p(Ia, Ib, II)
    """
    predicted_class = model.predict(X)
    predicted_proba_matrix = model.predict_proba(X)

    # Find which class is 'nonevent' if label_encoder is provided
    if label_encoder is not None:
        # LabelEncoder sorts classes alphabetically: Ia, Ib, II, nonevent
        # So nonevent should be the last class (index 3)
        # Sum probabilities for all classes except nonevent
        nonevent_idx = None
        for idx, label in enumerate(label_encoder.classes_):
            if label == "nonevent":
                nonevent_idx = idx
                break
        if nonevent_idx is not None:
            # Sum all probabilities except nonevent
            event_indices = [
                i for i in range(len(label_encoder.classes_)) if i != nonevent_idx
            ]
            predicted_proba = predicted_proba_matrix[:, event_indices].sum(axis=1)
        else:
            # Fallback: assume first 3 classes are events (for backward compatibility)
            predicted_proba = predicted_proba_matrix[:, 0:3].sum(axis=1)
    else:
        # Fallback: assume first 3 classes are events (Ia, Ib, II)
        predicted_proba = predicted_proba_matrix[:, 0:3].sum(axis=1)

    return pd.DataFrame(
        {
            "class4": predicted_class,
            "p": predicted_proba,
        },
        # Keep indices correct
        index=X.index,
    )


def return_target_df(y: np.ndarray, X):
    """Converts numpy array of targets back to dataframe

    class4: True class (nonevent, Ia, Ib or II)
    """
    return pd.DataFrame(
        {"class4": y},
        # Keep indices correct
        index=X.index,
    )


# ================================
# Data Loading and pre-processing
# ================================

train_path = "data/train.csv"
test_path = "data/test.csv"

# set_index makes sure we don't use the id column for training
train = pd.read_csv(train_path).set_index("id")
test = pd.read_csv(test_path).set_index("id")

print("Loaded train, test and sample submission")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# date is not present in the test data and is only included "solely for
# exploratory data analysis"
# partlybad also seems useless
train_x = train.drop(columns=["class4", "date", "partlybad"])
test_x = test.drop(columns=["date", "partlybad"])

# Optional: Keep only the mean measurements and reduce number of columns to 50
train_x = train_x.loc[:, train_x.columns.str.contains("mean")]
test_x = test_x.loc[:, test_x.columns.str.contains("mean")]

# Target column
train_y = train[["class4"]]

# Standard scaling (needed for PCA and LR/SVM)
cols = train_x.columns
sc = StandardScaler()

# Convert back to dataframe format
train_x_sc = pd.DataFrame(sc.fit_transform(train_x), columns=cols, index=train_x.index)
test_x_sc = pd.DataFrame(sc.transform(test_x), columns=cols, index=test_x.index)

# Fit pca to reduce to ncomponents
num_comp = 14
pca = PCA(n_components=num_comp)
pca.fit(train_x_sc)

# PCA transformation, convert back into dataframe of same format
pca_cols = [f"PCA_{i + 1}" for i in range(num_comp)]
train_x_pca = pd.DataFrame(
    pca.transform(train_x_sc), columns=pca_cols, index=train_x_sc.index
)
test_x_pca = pd.DataFrame(
    pca.transform(test_x_sc), columns=pca_cols, index=test_x_sc.index
)


# ================================
# Model Training
# ================================

svc_hparams = {
    "C": 5,
    "kernel": "rbf",
    "gamma": 0.001,
    "probability": True,
}

X_train = train_x_pca
y_train = np.array(train_y)[:, 0]

# Label encoding (nonevent, Ia, Ib, II -> 0, 1, 2, 3)
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Fit model, SVM classifier in this case
clf = svm.SVC(**svc_hparams).fit(X_train, y_train)

# k-fold CV
cv_acc = np.mean(cross_val_score(clf, X_train, y_train, scoring="accuracy", cv=10))

# Get (train) predictions (np array of encoded classes)
y_pred = clf.predict(X_train)

# Get (train) predictions (dataframe in submission-ready format)
preds = predict(clf, X_train, label_encoder=le)
target_df = return_target_df(le.inverse_transform(y_train), X_train)
preds["class4"] = le.inverse_transform(preds["class4"])

# Accuracy and scoring
train_acc = accuracy_score(y_train, y_pred)
print("\n==================================")
print("Model metrics")
print(f"Hyperparameters: {svc_hparams}")
print(f"Train accuracy: {train_acc:.3f}")
print(f"Cross validation accuracy: {cv_acc}")
print("Score on training set:")
score(target_df, preds, "id")
print("==================================\n")

# ===================================
# Generate pseudo labels
# ===================================

# Predict on test data to produce pseudo labels
test_preds = predict(clf, test_x_pca, label_encoder=le)
test_preds["class4"] = le.inverse_transform(test_preds["class4"])

# Combine train_y and predicted test_y
test_y_pred = test_preds.drop("p", axis=1)
train_y = pd.concat([train_y, test_y_pred])

# Combine train_x and test_x
train_x = pd.concat([train_x, test_x])

# Standard scaling and return as dataframe
train_x_sc = pd.DataFrame(sc.fit_transform(train_x), columns=cols, index=train_x.index)

# PCA fit on train + test data
pca.fit(train_x_sc)
train_x_pca = pd.DataFrame(
    pca.transform(train_x_sc), columns=pca_cols, index=train_x_sc.index
)


test_x_sc = pd.DataFrame(sc.transform(test_x), columns=cols, index=test_x.index)
test_x_pca = pd.DataFrame(
    pca.transform(test_x_sc), columns=pca_cols, index=test_x.index
)


# ====================================
# Model Training (using pseudo labels)
# ====================================

X_train = train_x_pca
y_train = np.array(train_y)[:, 0]

# Label encoding
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Fit model
clf = svm.SVC(**svc_hparams).fit(X_train, y_train)

# k-fold CV
cv_acc = np.mean(cross_val_score(clf, X_train, y_train, scoring="accuracy", cv=10))

# Get (train) predictions (np array of encoded classes)
y_pred = clf.predict(X_train)

# Get (train) predictions (dataframe in submission-ready format)
preds = predict(clf, X_train, label_encoder=le)
target_df = return_target_df(le.inverse_transform(y_train), X_train)
preds["class4"] = le.inverse_transform(preds["class4"])

# Accuracy and scoring
train_acc = accuracy_score(y_train, y_pred)
print("\n==================================")
print("Model metrics using pseudolabels")
print(f"Train accuracy: {train_acc:.3f}")
print(f"Cross validation accuracy: {cv_acc}")
print("Score on training set:")
score(target_df, preds, "id")
print("==================================\n")

# Predict on test data and store to .csv file for submission
test_preds = predict(clf, test_x_pca, label_encoder=le)
test_preds["class4"] = le.inverse_transform(test_preds["class4"])
test_preds.to_csv("submission.csv", index_label="id")
print("Predictions saved to submission.csv.")


# plots
plot_training_data(train)
plot_classification_report(clf, y_train, y_pred, label_encoder=le)
plot_feature_importances(clf, pca=pca)
print(
    "Saved figures: class4_distribution.png, class2_distribution.png, correlation_heatmap.png, feature_importances_top20.png, confusion_matrix_train.png"
)
analyse_pca(pca, train_x_sc.columns)
plot_test_set_results(test_preds, test_x_sc)
