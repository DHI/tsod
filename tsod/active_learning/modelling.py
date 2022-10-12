from typing import List
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tsod.active_learning.data_structures import AnnotationState


def get_neighboring_points(
    indices: List, number_neighbors: int, column_for_normalization: str | None = None
):
    """
    Given a list of datetime indices, returns the number_neighbors points before and
    after each point (from df_full).
    If column_for_normalization is set, every neighbor's value is divided by this columns value at the corresponding index.
    """
    df: pd.DataFrame = st.session_state["df_full"].reset_index(names="date")

    if column_for_normalization and (column_for_normalization not in df.columns):
        raise ValueError(f"Column {column_for_normalization} not found.")

    mask = df["date"].isin(indices)

    all_neighbors: List[List] = []

    # Only access df where it matches indices for loop
    for i in df.loc[mask].index:
        sample_neighbors = []
        normalization_value = (
            df.loc[i, column_for_normalization] if column_for_normalization else 1.0
        )
        for i_2 in range(i - number_neighbors, i + number_neighbors + 1):
            try:
                if i_2 != i:
                    # Access full df for neighboring values
                    sample_neighbors.append(df.loc[i_2, "Water Level"] / normalization_value)
            except KeyError:
                sample_neighbors.append(1.0)

        all_neighbors.append(sample_neighbors)

    return all_neighbors


def construct_training_data(window_size: int = 10):
    state: AnnotationState = st.session_state.AS
    outliers = state.df_outlier
    if outliers.empty:
        return
    normal = state.df_normal

    features = []
    labels = []

    features.extend(get_neighboring_points(outliers.index.to_list(), window_size, "Water Level"))
    features.extend(get_neighboring_points(normal.index.to_list(), window_size, "Water Level"))

    labels.extend([1] * len(outliers))
    labels.extend([0] * len(normal))

    class_labels = [f"t-{i}" for i in reversed(range(1, window_size + 1))]
    class_labels.extend([f"t+{i}" for i in range(1, window_size + 1)])

    features = pd.DataFrame(features, columns=class_labels)
    labels = np.array(labels)

    st.session_state["features"] = features
    st.session_state["labels"] = labels


def train_random_forest_classifier():
    if "features" not in st.session_state:
        st.warning("No features were created, not training a model.")
        return
    X = st.session_state["features"]
    y = st.session_state["labels"]

    clf = RandomForestClassifier()
    clf.fit(X, y)

    st.session_state["classifier"] = clf


def show_post_training_info(base_obj=None):
    obj = base_obj or st
    clf = st.session_state["classifier"]

    obj.table(
        sorted(
            [
                {"Feature": feat, "Feature importance": imp}
                for feat, imp in zip(clf.feature_names_in_, clf.feature_importances_)
            ],
            key=lambda x: x["Feature importance"],
            reverse=True,
        )[:10]
    )
