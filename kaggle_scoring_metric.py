"""
A custom scoring metric for the Kaggle competition.

The function `score` first validates the submission file and then computes the three
classification metrics: binary classification accuracy, perplexity, and multi-class
classification accuracy. These are then combined into one metric with values in [0, 1].

The class `ParticipantVisibleError` redirects specified raised exceptions
to Kaggle's error stream so that users see error messages; otherwise, exceptions
would only be visible to the host.
"""

# sourced from moodle.helsinki.fi IML25/Materials/Term project

import numpy as np
import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Computes the binary classification accuracy, perplexity, and multi-class classification accuracy
    for a submission and combines them into one score. Additionally handles validation of
    the submission dataframe's contents.

    Args:
        solution (pd.DataFrame): The dataframe generated from the test file.
        submission (pd.DataFrame): The dataframe generated from the submission file.
        row_id_column_name (str): The name of the column used for aligning the dataframe rows.

    Returns:
        float: The combined score computed as:
            (1/3) * (binary accuracy + multi-class accuracy + max(0, min(1, 2 - perplexity)))

    Examples:
        >>> import pandas as pd
        >>> row_id_column_name = "id"
        >>> solution = pd.DataFrame({
        ...     'id': [1, 2, 3, 4],
        ...     'class4': ['Ia', 'nonevent', 'Ib', 'nonevent']
        ... })
        >>> submission = pd.DataFrame({
        ...     'id': [1, 2, 3, 4],
        ...     'class4': ['Ia', 'nonevent', 'nonevent', 'nonevent'],
        ...     'p': [0.9, 0.1, 0.2, 0.8]
        ... })
        >>> score(solution, submission, row_id_column_name)
        Binary Accuracy = 0.75000
        Perplexity = 2.35702
        Multi-Class Accuracy = 0.75000
        Combined Score = 0.50000
        0.5
    """

    # Remove the row ID column if present; ignore if missing
    if row_id_column_name in solution.columns:
        solution = solution.drop(columns=[row_id_column_name])
    if row_id_column_name in submission.columns:
        submission = submission.drop(columns=[row_id_column_name])

    # Assert existence of required columns in submission
    required_cols = ["class4", "p"]
    for col in required_cols:
        if col not in list(submission.columns):
            raise ParticipantVisibleError(
                f"The submission is missing the column {col}. Expected columns are {required_cols}"
            )

    # Validate column types
    if not pandas.api.types.is_object_dtype(submission["class4"]):
        raise ParticipantVisibleError(f"The 'class4' column should be of object type.")
    if not pandas.api.types.is_numeric_dtype(submission["p"]):
        raise ParticipantVisibleError(f"The 'p' column should be of numeric type.")

    # Check submission size matches solution size
    if len(submission) != len(solution):
        raise ParticipantVisibleError(
            f"Invalid number of rows in submission. Found {len(submission)} rows, but expected {len(solution)} rows."
        )

    # Validate probabilities are within [0, 1]
    if not ((submission["p"] >= 0) & (submission["p"] <= 1)).all():
        raise ParticipantVisibleError("The 'p' column contains invalid probabilities. All should be within [0, 1].")

    # Compute binary classification accuracy (event vs nonevent)
    solution_class2 = solution["class4"] != "nonevent"
    submission_class2 = submission["class4"] != "nonevent"
    binary_accuracy = float(np.mean(solution_class2 == submission_class2))

    # Compute perplexity over predicted probabilities for the correct binary class
    p = submission["p"].values
    events = (solution["class4"] != "nonevent").values
    p_i = np.where(events, p, 1 - p)
    p_i = np.clip(p_i, 1e-15, 1 - 1e-15)  # numerical stability
    perplexity = float(np.exp(-np.mean(np.log(p_i))))

    # Normalise perplexity to [0,1]:
    # 1 corresponds to perfect predictions (perplexity = 1),
    # 0 corresponds to random guessing baseline (perplexity >= 2)
    normalized_perplexity = max(0, min(1, 2 - perplexity))

    # Compute multi-class accuracy over all four classes
    multi_class_accuracy = float(np.mean(submission["class4"] == solution["class4"]))

    # Combine metrics with equal weight
    combined_score = (binary_accuracy + multi_class_accuracy + normalized_perplexity) / 3

    print(
        f"Binary Accuracy = {binary_accuracy:.5f}\n"
        f"Perplexity = {perplexity:.5f}\n"
        f"Multi-Class Accuracy = {multi_class_accuracy:.5f}\n"
        f"Combined Score = {combined_score:.5f}"
    )

    return combined_score


if __name__ == "__main__":
    solution = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'class4': ['Ia', 'nonevent', 'Ib', 'nonevent']
    })

    submission = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'class4': ['Ia', 'nonevent', 'nonevent', 'nonevent'],
        'p': [0.9, 0.1, 0.2, 0.8]
    })

    s = score(solution, submission, "id")
    print(s)