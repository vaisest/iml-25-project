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


def plot_feature_importances(model: RandomForestClassifier):
    features_importances = sorted(
        zip(model.feature_importances_, model.feature_names_in_), key=lambda x: x[0]
    )

    print("importances:", features_importances)

    top = features_importances[-20:]
    imp_vals = [v for v, _ in top]
    imp_names = [n for _, n in top]

    plt.figure(figsize=(7, 6))
    sns.barplot(x=imp_vals, y=imp_names)
    plt.title("Top 20 feature importances")
    plt.tight_layout()
    plt.savefig("feature_importances_top20.png")
    plt.close()


def plot_classification_report(model, train_y, train_pred):
    print("\nTrain classification report (class4):")
    print(classification_report(train_y, train_pred, zero_division=0))

    cm = confusion_matrix(
        train_y,
        train_pred,
        labels=model.classes_,
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(xticks_rotation=45)
    plt.title("Train confusion matrix (class4)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_train.png")
    plt.close()

    # score on training set
