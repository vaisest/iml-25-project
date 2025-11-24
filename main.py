from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
import seaborn as sns

from kaggle_scoring_metric import score

train_path = "data/train.csv"
test_path = "data/test.csv"

# set_index makes sure we don't use the id column for training
train = pd.read_csv(train_path).set_index("id")
test = pd.read_csv(test_path).set_index("id")

print("Loaded train, test and sample submission")
print(f"   Train shape: {train.shape}, Test shape: {test.shape}")


# date is not present in the test data and is only included for "solely for
# exploratory data analysis"
# partlybad also seems useless
train_x = train.drop(columns=["class4", "date", "partlybad"])
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
print(gs.best_params_)
print(gs.best_score_)

model = RandomForestClassifier(**gs.best_params_).fit(train_x, train_y)
features_importances = sorted(
    zip(model.feature_importances_, model.feature_names_in_), key=lambda x: x[0]
)
print("importances:", features_importances)


# score on training set
score(train[["class4"]], predict(model, train_x), "id")

test_prediction = predict(model, test_x)
test_prediction.to_csv("submission.csv", index_label="id")
