import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
train_x = train.drop(columns=["class4", "date"])
test_x = test.drop(columns=["date"])
train_y = train["class4"]

model = RandomForestClassifier(n_estimators=300, n_jobs=-1).fit(train_x, train_y)


def predict(X: pd.DataFrame):
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


# score on training set
score(train[["class4"]], predict(train_x), "id")

test_prediction = predict(test_x)
test_prediction.to_csv("submission.csv", index_label="id")
