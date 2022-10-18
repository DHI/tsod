import pickle
from typing import Dict, List
import streamlit as st
import pandas as pd
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from tsod.active_learning.data_structures import AnnotationState
from tsod.active_learning.utils import get_as


def get_neighboring_points(
    indices: List,
    number_neighbors: int,
    full_df: pd.DataFrame,
    column_for_normalization: str | None = None,
):
    """
    Given a list of datetime indices, returns the number_neighbors points before and
    after each point (from df_full).
    If column_for_normalization is set, every neighbor's value is divided by this columns value at the corresponding index.
    """
    df: pd.DataFrame = full_df.reset_index(names="date")

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
    state: AnnotationState = get_as()
    outliers = state.df_outlier
    if outliers.empty:
        return
    normal = state.df_normal

    features = []
    labels = []

    features.extend(
        get_neighboring_points(outliers.index.to_list(), window_size, state.df, "Water Level")
    )
    features.extend(
        get_neighboring_points(normal.index.to_list(), window_size, state.df, "Water Level")
    )

    labels.extend([1] * len(outliers))
    labels.extend([0] * len(normal))

    class_labels = [f"t-{i}" for i in reversed(range(1, window_size + 1))]
    class_labels.extend([f"t+{i}" for i in range(1, window_size + 1)])

    features = pd.DataFrame(features, columns=class_labels)
    labels = np.array(labels)

    # with open("data.pcl", "wb") as f:
    # pickle.dump({"features": features, "labels": labels}, f)

    st.session_state["features"] = features
    st.session_state["labels"] = labels


def train_random_forest_classifier(base_obj=None):
    obj = base_obj or st

    if "features" not in st.session_state:
        st.warning("No features were created, not training a model.")
        return
    X = st.session_state["features"]
    y = st.session_state["labels"]

    rfc = RandomForestClassifier()

    forest_params = {
        "max_depth": [int(x) for x in np.linspace(10, 30, num=3)] + [None],
        "max_features": ["sqrt", "log2"],
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        "min_samples_split": [2, 4],
        "bootstrap": [True, False],
    }
    clf = RandomizedSearchCV(
        estimator=rfc, param_distributions=forest_params, cv=3, n_iter=10, n_jobs=-1, verbose=0
    )

    clf.fit(X, y)

    st.session_state["classifier"] = clf.best_estimator_
    st.session_state[
        "last_model_name"
    ] = f"Model ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"
    obj.success("Finished training.")


def post_training_options(base_obj=None):
    obj = base_obj or st
    if "classifier" not in st.session_state:
        return

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

    obj.download_button(
        "Download model", pickle.dumps(clf), f"{st.session_state.last_model_name}.pkl"
    )


def get_model_predictions(
    window_size: int = 10, column_for_normalization: str | None = None, base_obj=None
):
    obj = base_obj or st
    models: Dict[str, RandomForestClassifier] = st.session_state["prediction_models"]
    datasets: Dict[str, pd.DataFrame] = st.session_state["prediction_data"]

    if (not models) or (not datasets):
        obj.error("Please add at least one model and one data file.")
        return

    for dataset_name, ds in datasets.items():
        try:
            features = st.session_state["uploaded_ds_features"][dataset_name]
        except KeyError:
            with st.spinner("Constructing dataset features..."):
                features = get_neighboring_points(
                    ds.index, window_size, ds, column_for_normalization
                )
                st.session_state["uploaded_ds_features"][dataset_name] = features

        with st.spinner("Getting model results..."):
            for model_name, model in models.items():
                st.session_state["models_to_visualize"][dataset_name].update([model_name])
                if model_name not in st.session_state["inference_results"][dataset_name]:
                    results = model.predict(features)
                    st.session_state["inference_results"][dataset_name][model_name] = results
                    st.session_state["number_outliers"][dataset_name][model_name] = len(
                        results.nonzero()[0].tolist()
                    )
