import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def plot_training_data(train: pd.DataFrame):
    print("\nclass4 distribution:")
    print(train["class4"].value_counts())

    plt.figure(figsize=(5, 4))
    train["class4"].value_counts().plot(kind="bar")
    plt.title("class4 distribution")
    plt.tight_layout()
    plt.savefig("class4_distribution.png")
    plt.close()

    train["class2"] = (train["class4"] != "nonevent").astype(int)

    print("\nclass2 distribution (1=event, 0=nonevent):")
    print(train["class2"].value_counts())

    plt.figure(figsize=(5, 4))
    train["class2"].value_counts().plot(kind="bar")
    plt.title("class2 distribution")
    plt.tight_layout()
    plt.savefig("class2_distribution.png")
    plt.close()

    numeric_cols = train.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        cols_for_corr = (
            np.random.choice(
                numeric_cols, size=min(30, len(numeric_cols)), replace=False
            )
            if len(numeric_cols) > 30
            else numeric_cols
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(train[cols_for_corr].corr(), cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (subset of numeric features)")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()


def plot_feature_importances(model, pca=None):
    """
    Plot feature importances.
    For RandomForest: uses feature_importances_
    For other models with PCA: uses explained_variance_ratio_ from PCA
    """
    if hasattr(model, "feature_importances_"):
        # RandomForest or similar
        features_importances = sorted(
            zip(model.feature_importances_, model.feature_names_in_), key=lambda x: x[0]
        )
        print("Feature importances:", features_importances)

        top = features_importances[-20:]
        imp_vals = [v for v, _ in top]
        imp_names = [n for _, n in top]

        plt.figure(figsize=(7, 6))
        sns.barplot(x=imp_vals, y=imp_names)
        plt.title("Top 20 feature importances")
        plt.tight_layout()
        plt.savefig("feature_importances_top20.png")
        plt.close()
    elif pca is not None:
        # Use PCA explained variance ratio
        pca_importances = sorted(
            zip(pca.explained_variance_ratio_, [f"PCA_{i+1}" for i in range(len(pca.explained_variance_ratio_))]),
            key=lambda x: x[0]
        )
        print("PCA component importances (explained variance ratio):", pca_importances)

        top = pca_importances[-20:] if len(pca_importances) > 20 else pca_importances
        imp_vals = [v for v, _ in top]
        imp_names = [n for _, n in top]

        plt.figure(figsize=(7, 6))
        sns.barplot(x=imp_vals, y=imp_names)
        plt.title("Top 20 PCA component importances (explained variance ratio)")
        plt.tight_layout()
        plt.savefig("feature_importances_top20.png")
        plt.close()
    else:
        print("Warning: Model does not support feature importances and no PCA provided. Skipping plot.")


def plot_classification_report(model, train_y, train_pred, label_encoder=None):
    """
    Plot classification report and confusion matrix.
    
    Args:
        model: Trained classifier
        train_y: True labels (can be encoded or original)
        train_pred: Predicted labels (can be encoded or original)
        label_encoder: Optional LabelEncoder to convert encoded labels back to original
    """
    # Convert encoded labels back to original if label_encoder is provided
    if label_encoder is not None:
        train_y_original = label_encoder.inverse_transform(train_y)
        train_pred_original = label_encoder.inverse_transform(train_pred)
        display_labels = label_encoder.classes_
    else:
        train_y_original = train_y
        train_pred_original = train_pred
        # Try to get labels from model
        if hasattr(model, "classes_"):
            display_labels = model.classes_
        else:
            display_labels = sorted(set(train_y_original) | set(train_pred_original))

    print("\nTrain classification report (class4):")
    print(classification_report(train_y_original, train_pred_original, zero_division=0))

    cm = confusion_matrix(
        train_y_original,
        train_pred_original,
        labels=display_labels,
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(xticks_rotation=45)
    plt.title("Train confusion matrix (class4)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_train.png")
    plt.close()

    # score on training set
