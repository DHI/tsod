import datetime
import logging
from typing import Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV

from tsod.active_learning.utils import (
    get_as,
    recursive_length_count,
    recursive_round,
    set_session_state_items,
)


def get_neighboring_points(
    indices: List,
    data_column: str,
    points_before: int,
    points_after: int,
    full_df: pd.DataFrame,
    column_for_normalization: str | None = None,
) -> List[List]:
    """
    Given a list of datetime indices, returns the number_neighbors points before and
    after each point (from full_df).
    If column_for_normalization is set, every neighbor's value is divided by this columns value at the corresponding index.
    """
    df: pd.DataFrame = full_df.reset_index(names="date")

    df = df[~df[data_column].isna()]

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
        if abs(normalization_value) < 1e-4:
            if normalization_value < 0:
                normalization_value = -1e-4
            else:
                normalization_value = 1e-4
        for i_2 in range(i - points_before, i + points_after + 1):
            try:
                if i_2 != i:
                    # Access full df for neighboring values
                    sample_neighbors.append(df.loc[i_2, data_column] / normalization_value)
            except KeyError:
                sample_neighbors.append(1.0)

        all_neighbors.append(sample_neighbors)

    return all_neighbors


def get_class_labels_RF(points_before: int, points_after: int):
    class_labels = [f"t-{i}" for i in reversed(range(1, points_before + 1))]
    class_labels.extend([f"t+{i}" for i in range(1, points_after + 1)])

    return class_labels


def construct_training_data_RF(dataset: str | None = None, series: str | None = None):

    points_before = (
        st.session_state["number_points_before"]
        if st.session_state.get("number_points_before") is not None
        else st.session_state["model_library"][
            st.session_state[f"last_model_name_{dataset}_{series}"]
        ]["params"]["points_before"]
    )
    points_after = (
        st.session_state["number_points_after"]
        if st.session_state.get("number_points_after") is not None
        else st.session_state["model_library"][
            st.session_state[f"last_model_name_{dataset}_{series}"]
        ]["params"]["points_after"]
    )
    st.session_state["last_points_before"] = points_before
    st.session_state["last_points_after"] = points_after
    state = get_as(dataset, series)
    outliers = state.df_outlier
    if outliers.empty:
        return
    normal = state.df_normal

    features = []
    labels = []
    features.extend(
        get_neighboring_points(
            indices=outliers.index.to_list(),
            data_column=state.column,
            points_before=points_before,
            points_after=points_after,
            full_df=state.df,
            column_for_normalization=state.column,
        )
    )
    features.extend(
        get_neighboring_points(
            indices=normal.index.to_list(),
            data_column=state.column,
            points_before=points_before,
            points_after=points_after,
            full_df=state.df,
            column_for_normalization=state.column,
        )
    )

    labels.extend([1] * len(outliers))
    labels.extend([0] * len(normal))

    features = pd.DataFrame(features, columns=get_class_labels_RF(points_before, points_after))
    labels = np.array(labels)

    st.session_state["features"] = features
    st.session_state["labels"] = labels


def construct_test_data_RF(dataset: str | None = None, series: str | None = None):
    points_before = (
        st.session_state["number_points_before"]
        if st.session_state.get("number_points_before") is not None
        else st.session_state["model_library"][
            st.session_state[f"last_model_name_{dataset}_{series}"]
        ]["params"]["points_before"]
    )
    points_after = (
        st.session_state["number_points_after"]
        if st.session_state.get("number_points_after") is not None
        else st.session_state["model_library"][
            st.session_state[f"last_model_name_{dataset}_{series}"]
        ]["params"]["points_after"]
    )

    state = get_as(dataset, series)
    dataset = state.dataset
    series = state.column
    outliers = state.df_test_outlier
    normal = state.df_test_normal

    if outliers.empty and normal.empty:
        return

    features = []
    labels = []
    features.extend(
        get_neighboring_points(
            outliers.index.to_list(),
            state.column,
            points_before,
            points_after,
            state.df,
            state.column,
        )
    )
    features.extend(
        get_neighboring_points(
            normal.index.to_list(),
            state.column,
            points_before,
            points_after,
            state.df,
            state.column,
        )
    )
    labels.extend([1] * len(outliers))
    labels.extend([0] * len(normal))

    class_labels = [f"t-{i}" for i in reversed(range(1, points_before + 1))]
    class_labels.extend([f"t+{i}" for i in range(1, points_after + 1)])

    features = pd.DataFrame(features, columns=class_labels)
    labels = np.array(labels)

    st.session_state[f"test_features_{dataset}_{series}"] = features
    st.session_state["test_labels"] = labels

    return True


def train_model(base_obj=None, dataset: str | None = None, series: str | None = None):
    obj = base_obj or st

    state = get_as(dataset, series)
    if recursive_length_count(state.data, exclude_keys="selected"):
        st.session_state[f"old_number_annotations_{state.dataset}_{state.column}"] = {
            k: len(v) for k, v in state.data.items()
        }

    with st.spinner("Constructing features..."):
        if st.session_state["last_method_choice"] == "RF_1":
            construct_training_data_RF(dataset, series)
            construct_test_data_RF(dataset, series)
    with st.spinner("Training Model..."):
        if st.session_state["last_method_choice"] == "RF_1":
            train_random_forest_classifier(obj, dataset, series)

        logging.info("A new model was trained successfully.")


def train_random_forest_classifier(
    base_obj=None, dataset: str | None = None, series: str | None = None
):
    obj = base_obj or st

    state = get_as(dataset, series)
    dataset = state.dataset
    series = state.column

    if "features" not in st.session_state:
        obj.warning("No features were created, not training a model.")
        return
    X = st.session_state["features"]
    y = st.session_state["labels"]

    rfc = RandomForestClassifier(random_state=30)

    forest_params = {
        "max_depth": [int(x) for x in np.linspace(10, 30, num=3)] + [None],
        "max_features": ["sqrt", "log2"],
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        "min_samples_split": [2, 4],
        "bootstrap": [True, False],
    }
    clf = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=forest_params,
        cv=3,
        n_iter=10,
        n_jobs=-1,
        verbose=0,
        random_state=30,
    )

    clf.fit(X, y)

    model_name = f"RF_Classifier ({datetime.datetime.now(tz=ZoneInfo('Europe/Copenhagen')).strftime('%Y-%m-%d %H:%M:%S')})"
    st.session_state[f"last_model_name_{dataset}_{series}"] = model_name
    st.session_state["most_recent_model"] = model_name
    st.session_state["models_trained_this_session"].update([model_name])
    # Update Test metrics
    if f"current_model_test_metrics_{dataset}_{series}" in st.session_state:
        st.session_state[f"previous_model_test_metrics_{dataset}_{series}"] = st.session_state[
            f"current_model_test_metrics_{dataset}_{series}"
        ]
        st.session_state[f"previous_model_train_metrics_{dataset}_{series}"] = st.session_state[
            f"current_model_train_metrics_{dataset}_{series}"
        ]

    model = clf.best_estimator_

    train_preds = model.predict(X)
    train_prec, train_rec, train_f1, train_support = precision_recall_fscore_support(
        y, train_preds, zero_division=1
    )
    st.session_state[f"current_model_train_metrics_{dataset}_{series}"] = recursive_round(
        {"precision": train_prec, "recall": train_rec, "f1": train_f1}
    )

    if f"test_features_{dataset}_{series}" in st.session_state:
        X_test = st.session_state[f"test_features_{dataset}_{series}"]
        y_test = st.session_state["test_labels"]
        test_preds = model.predict(X_test)
        test_prec, test_rec, test_f1, support = precision_recall_fscore_support(
            y_test, test_preds, zero_division=1
        )
        st.session_state[f"current_model_test_metrics_{dataset}_{series}"] = recursive_round(
            {"precision": test_prec, "recall": test_rec, "f1": test_f1}
        )

    if f"current_importances_{dataset}_{series}" in st.session_state:
        st.session_state[f"previous_importances_{dataset}_{series}"] = st.session_state[
            f"current_importances_{dataset}_{series}"
        ]

    df_fi = pd.DataFrame(
        [
            {"Feature": feat, "Feature importance": imp.round(3)}
            for feat, imp in zip(model.feature_names_in_, model.feature_importances_)
        ],
    ).sort_values("Feature importance", ascending=False)

    st.session_state[f"current_importances_{dataset}_{series}"] = df_fi

    st.session_state["model_library"][model_name] = {
        "model": model,
        "type": st.session_state["last_method_choice"],
        "params": {
            "points_before": st.session_state["last_points_before"],
            "points_after": st.session_state["last_points_after"],
        },
        "trained_on_dataset": dataset,
        "trained_on_series": series,
    }

    st.session_state["prediction_models"][model_name] = st.session_state["model_library"][
        model_name
    ]

    st.session_state["prediction_data"][dataset] = [series]


def get_model_predictions(base_obj=None):
    obj = base_obj or st
    models: Dict[str, RandomForestClassifier] = st.session_state["prediction_models"]
    datasets: Dict[str, pd.DataFrame] = st.session_state["prediction_data"]
    if (not models) or (not datasets):
        obj.error("Please add at least one model and one data file.")
        return

    get_model_predictions_RF()


def get_model_predictions_RF():
    model_dicts: Dict[str, Dict] = {
        k: v for k, v in st.session_state["prediction_models"].items() if v["type"] == "RF_1"
    }
    datasets = st.session_state["prediction_data"]

    if (not model_dicts) or (not datasets):
        return

    params = [d["params"] for d in model_dicts.values()]
    start_features = max([p["points_before"] for p in params])
    end_features = max([p["points_after"] for p in params])

    for dataset_name, series_list in datasets.items():
        # for dataset_name, ds in datasets.items():
        ds = st.session_state["data_store"][dataset_name]
        for series in series_list:
            df_series = ds[series]
            if (
                (series not in st.session_state["uploaded_ds_features"][dataset_name])
                or (start_features > st.session_state["RF_features_computed_start"])
                or (end_features > st.session_state["RF_features_computed_end"])
            ):
                with st.spinner(f"{dataset_name} - {series}: Constructing dataset features..."):
                    features = get_neighboring_points(
                        indices=df_series.index,
                        data_column=series,
                        points_before=start_features,
                        points_after=end_features,
                        full_df=df_series,
                        column_for_normalization=series,
                    )

                    feature_names = get_class_labels_RF(start_features, end_features)
                    features = pd.DataFrame(features, columns=feature_names)
                    st.session_state["uploaded_ds_features"][dataset_name][series] = features
                    st.session_state["RF_features_computed_start"] = start_features
                    st.session_state["RF_features_computed_end"] = end_features

            with st.spinner(f"{dataset_name} - {series}: Getting model results..."):
                if series not in st.session_state["inference_results"][dataset_name]:
                    st.session_state["inference_results"][dataset_name][series] = df_series.copy(
                        deep=True
                    )
                for model_name, model_data in model_dicts.items():
                    if model_name in st.session_state["inference_results"][dataset_name][series]:
                        continue
                    # st.session_state["models_to_visualize"][dataset_name].update([model_name])
                    # select relevant subset of ds features, based on what the model needs
                    number_model_features_before = model_data["params"]["points_before"]
                    number_model_features_after = model_data["params"]["points_after"]
                    relevant_model_columns = get_class_labels_RF(
                        number_model_features_before, number_model_features_after
                    )
                    model_feautures = st.session_state["uploaded_ds_features"][dataset_name][
                        series
                    ][relevant_model_columns]
                    results = model_data["model"].predict(model_feautures)
                    probas = model_data["model"].predict_proba(model_feautures)
                    st.session_state["inference_results"][dataset_name][series][
                        model_name
                    ] = results
                    st.session_state["inference_results"][dataset_name][series][
                        f"certainty_{model_name}"
                    ] = np.abs(
                        probas[:, 0] - 0.5
                    )  # lower = more uncertain
                    st.session_state["number_outliers"][dataset_name][series][model_name] = len(
                        results.nonzero()[0].tolist()
                    )
                    st.session_state["available_models"][dataset_name][series].update([model_name])

                    st.session_state["models_to_visualize"][dataset_name][series] = set(
                        sorted(st.session_state["available_models"][dataset_name][series])[-2:]
                    )
