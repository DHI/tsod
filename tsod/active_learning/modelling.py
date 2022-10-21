import pickle
from typing import Dict, List
import streamlit as st
import pandas as pd
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from tsod.active_learning.utils import get_as, recursive_round
from sklearn.metrics import precision_recall_fscore_support


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
    state = get_as()
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


def construct_test_data(window_size: int = 10):
    state = get_as()
    outliers = state.df_test_outlier
    normal = state.df_test_normal

    if outliers.empty and normal.empty:
        return

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

    st.session_state["test_features"] = features
    st.session_state["test_labels"] = labels

    return True


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
    ] = f"RF_Classifier ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"

    # Update Test metrics
    if "current_model_test_metrics" in st.session_state:
        st.session_state["previous_model_test_metrics"] = st.session_state[
            "current_model_test_metrics"
        ]
        st.session_state["previous_model_train_metrics"] = st.session_state[
            "current_model_train_metrics"
        ]

    model = st.session_state["classifier"]
    X_test = st.session_state["test_features"]
    y_test = st.session_state["test_labels"]

    train_preds = model.predict(X)
    preds = model.predict(X_test)

    train_prec, train_rec, train_f1, train_support = precision_recall_fscore_support(y, train_preds)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, preds)

    st.session_state["current_model_test_metrics"] = recursive_round(
        {"precision": prec, "recall": rec, "f1": f1}
    )
    st.session_state["current_model_train_metrics"] = recursive_round(
        {"precision": train_prec, "recall": train_rec, "f1": train_f1}
    )

    if "current_importances" in st.session_state:
        st.session_state["previous_importances"] = st.session_state["current_importances"]

    df_fi = pd.DataFrame(
        [
            {"Feature": feat, "Feature importance": imp.round(3)}
            for feat, imp in zip(model.feature_names_in_, model.feature_importances_)
        ],
    ).sort_values("Feature importance", ascending=False)

    st.session_state["current_importances"] = df_fi

    st.session_state["prediction_models"][st.session_state["last_model_name"]] = model

    st.experimental_rerun()


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
                if model_name not in st.session_state["inference_results"][dataset_name]:
                    st.session_state["models_to_visualize"][dataset_name].update([model_name])
                    results = model.predict(features)
                    st.session_state["inference_results"][dataset_name][model_name] = results
                    st.session_state["number_outliers"][dataset_name][model_name] = len(
                        results.nonzero()[0].tolist()
                    )
