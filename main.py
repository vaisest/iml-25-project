from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from kaggle_scoring_metric import score

train_path = "data/train.csv"
test_path = "data/test.csv"

# set_index makes sure we don't use the id column for training
train = pd.read_csv(train_path).set_index("id")
test = pd.read_csv(test_path).set_index("id")

print("Loaded train, test and sample submission")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# EDA 

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
        np.random.choice(numeric_cols, size=min(30, len(numeric_cols)), replace=False)
        if len(numeric_cols) > 30 else numeric_cols
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[cols_for_corr].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (subset of numeric features)")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

# date is not present in the test data and is only included for "solely for
# exploratory data analysis"
# partlybad also seems useless
train_x = train.drop(columns=["class4", "class2", "date", "partlybad"])
test_x = test.drop(columns=["date", "partlybad"])
train_y = train["class4"]

# model = RandomForestClassifier(n_estimators=300, n_jobs=-1, max_depth=8).fit(
#     train_x, train_y
# )


def predict(model: RandomForestClassifier, X: pd.DataFrame):
    predicted_class = model.predict(X)
    # returns values for probability of each class. "nonevent" is the 4th class
    # encoded so p("event") = sum of first 3 probabilities
    predicted_proba = model.predict_proba(X)[:, 0:3].sum(axis=1)
    return pd.DataFrame(
        {
            "class4": predicted_class,
            "p": predicted_proba,
        },
        # keeps test data index correct
        index=X.index,
    )


# random experimentation, not optimal and might not make any sense
param_grid = [
    {
        "n_estimators": np.linspace(100, 1000, 4, dtype=np.int64),
        "max_depth": [3, 7, 15, None],
        "min_samples_leaf": np.linspace(1, 8, 4, dtype=np.int64),
    }
]
gs = GridSearchCV(RandomForestClassifier(), param_grid, verbose=2, n_jobs=-1).fit(
    train_x, train_y
)
print("\nBest params:", gs.best_params_)
print("Best CV score (GridSearch):", gs.best_score_)

model = RandomForestClassifier(**gs.best_params_).fit(train_x, train_y)
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

train_pred = predict(model, train_x)
print("\nTrain classification report (class4):")
print(classification_report(train_y, train_pred["class4"]))

cm = confusion_matrix(train_y, train_pred["class4"], labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(xticks_rotation=45)
plt.title("Train confusion matrix (class4)")
plt.tight_layout()
plt.savefig("confusion_matrix_train.png")
plt.close()

# score on training set
train_score = score(train[["class4"]].reset_index(), train_pred.reset_index(), "id")
print("\nKaggle-style score on train:", train_score)

test_prediction = predict(model, test_x)
test_prediction.to_csv("submission.csv", index_label="id")
print("\nSaved submission.csv (id, class4, p)")
print("Saved figures: class4_distribution.png, class2_distribution.png, correlation_heatmap.png, feature_importances_top20.png, confusion_matrix_train.png")
