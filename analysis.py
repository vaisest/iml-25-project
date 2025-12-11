import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
# sns.set_style("darkgrid")


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
            zip(
                pca.explained_variance_ratio_,
                [f"PCA_{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            ),
            key=lambda x: x[0],
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
        print(
            "Warning: Model does not support feature importances and no PCA provided. Skipping plot."
        )


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


def plot_test_set_results(predictions: pd.DataFrame, x: pd.DataFrame):
    print("####################################")
    print(len(predictions))
    predictions["class2"] = predictions["class4"] != "nonevent"
    print(predictions.head())

    # sns.set_style("darkgrid")

    # # plotting class2 count based on features. too hard to read but could be cool
    # combined = x.copy()
    # combined["class2"] = predictions["class2"]
    # combined["class4"] = predictions["class4"]
    # combined["p"] = predictions["p"]
    # print(combined.head())
    # sns.histplot(data=combined, x="UV_A.mean", y="class2", ax=axs[0])
    # sns.histplot(data=combined, x="UV_B.mean", y="class2", ax=axs[1])

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    p = sns.histplot(data=predictions["p"].to_numpy(), bins=25, ax=axs[0])
    p.set_xlabel("Predicted probabilities")
    p = sns.countplot(data=predictions, x="class2", ax=axs[1])
    p.set_xlabel("Class2 binary predictions")
    p.set_ylabel("Count")

    p = sns.countplot(
        data=predictions,
        x="class4",
        order=predictions["class4"].value_counts().index,  # sort
        ax=axs[2],
    )
    p.set_xlabel("Class4 multi-class predictions")
    p.set_ylabel("Count")
    plt.tight_layout()

    plt.savefig("test_set.png")


def analyse_pca(pca, initial_feature_names):
    # Source - https://stackoverflow.com/a/50845697, but edited
    # Posted by seralouk, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-11, License - CC BY-SA 4.0

    # number of components
    n_pcs = pca.components_.shape[0]

    # get the index of the most important feature on EACH component
    print(pca.components_[2])
    # for each component, we take the component values and sort them to get the
    # columns that contribute the most to the component
    most_important = [
        sorted(enumerate(pca.components_[i]), key=lambda x: abs(x[1]), reverse=True)[
            0:3
        ]
        for i in range(n_pcs)
    ]

    most_important_names = [
        [initial_feature_names[feat_idx] for feat_idx, _ in component]
        for component in most_important
    ]

    # LIST COMPREHENSION HERE AGAIN
    dic = {f"PC{i}": most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    df = pd.DataFrame(dic.items())

    print(df)
