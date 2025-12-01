import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from kaggle_scoring_metric import score
from analysis import (
    plot_training_data,
    plot_feature_importances,
    plot_classification_report,
)


train_path = "data/train.csv"
test_path = "data/test.csv"

# set_index makes sure we don't use the id column for training
train = pd.read_csv(train_path).set_index("id")
test = pd.read_csv(test_path).set_index("id")

print("Loaded train, test and sample submission")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")


# date is not present in the test data and is only included for "solely for
# exploratory data analysis"
# partlybad also seems useless
train_x = train.drop(columns=["class4", "date", "partlybad"])
test_x = test.drop(columns=["date", "partlybad"])
train_y = train["class4"]


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

model = RandomForestClassifier(**gs.best_params_).fit(train_x, train_y)

train_pred = predict(model, train_x)
train_score = score(train[["class4"]].reset_index(), train_pred.reset_index(), "id")
print("\nKaggle-style score on train:", train_score)

test_prediction = predict(model, test_x)
test_prediction.to_csv("submission.csv", index_label="id")
print("\nSaved submission.csv (id, class4, p)")
print(
    "Saved figures: class4_distribution.png, class2_distribution.png, correlation_heatmap.png, feature_importances_top20.png, confusion_matrix_train.png"
)


# plots
plot_training_data(train)
plot_classification_report(model, train_y, train_pred)
plot_feature_importances(model)
