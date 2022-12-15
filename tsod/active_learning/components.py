import datetime
import logging
import os
import pickle
from collections import defaultdict
from contextlib import nullcontext
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_pyecharts
from streamlit_profiler import Profiler

from tsod.active_learning import MEDIA_PATH
from tsod.active_learning.data_structures import AnnotationState, plot_return_value_as_datetime
from tsod.active_learning.instructions import INSTRUCTION_DICT
from tsod.active_learning.modelling import get_model_predictions, train_model
from tsod.active_learning.plotting import (
    feature_importance_plot,
    get_echarts_plot_time_range,
    make_annotation_suggestion_plot,
    make_outlier_distribution_plot,
    make_removed_outliers_example_plots,
    make_time_range_outlier_plot,
)
from tsod.active_learning.upload_data import add_new_data, data_uploader
from tsod.active_learning.utils import (
    MODEL_OPTIONS,
    custom_text,
    fix_random_seeds,
    get_as,
    recursive_ss_search,
    set_session_state_items,
    show_memory_usage,
    ss_recursive_df_memory_usage,
)


def outlier_annotation():
    exp = st.sidebar.expander("Data Upload", expanded=not len(st.session_state["data_store"]))
    data_uploader(exp)
    st.sidebar.title("Annotation Controls")
    data_selection(st.sidebar)
    state = get_as()
    if not state:
        if st.session_state["page_index"] != FUNC_IDX_MAPPING["1. Outlier Annotation"]:
            st.session_state["page_index"] = FUNC_IDX_MAPPING["1. Outlier Annotation"]
            st.experimental_rerun()
        st.info(
            """Upload your data to get started (using 'Data Upload' in the sidebar)!  
        If you just want to try out the application, you can also use some
        randomly generated time series data by clicking the button below."""
        )
        st.button("Add generated data", on_click=generate_example_data)
        st.markdown("***")

        st.warning(
            """For first time users: It is recommended to check out the 
        instructions before starting.  
        If you are unsure what a widget is for, hover your mouse above the little question mark next to it to get some more info."""
        )
        st.button(
            "View instructions",
            on_click=set_session_state_items,
            args=("page_index", FUNC_IDX_MAPPING["Instructions"]),
        )
        st.image(
            str(MEDIA_PATH / "workflow.png"),
            use_column_width=True,
            caption="Basic workflow suggestion",
        )
        return
    with st.sidebar.expander("Actions", expanded=True):
        create_annotation_plot_buttons()
    with st.sidebar.expander("Time Range Selection", expanded=True):
        time_range_selection()
    with st.sidebar.expander("Save / load previous", expanded=True):
        create_save_load_buttons()

    plot = get_echarts_plot_time_range(
        state.start,
        state.end,
        state.column,
        True,
        f"Outlier Annotation - {state.dataset} - {state.column}",
    )

    clicked_point = st_pyecharts(
        plot,
        height=800,
        theme="dark",
        events={
            "click": "function(params) { return [params.data[0], 'click'] }",
            "brushselected": "function(params) { return [params.batch[0].selected, 'brush'] }",
            # "brushselected": """function(params) { return [params.batch[0].selected.filter( obj => {return obj.seriesName === 'Datapoints'})[0].dataIndex, 'brush'] }""",
        },
    )
    process_data_from_echarts_plot(clicked_point)


def model_training():
    st.sidebar.title("Training Controls")
    if not st.session_state["data_store"]:
        st.info(
            """To train models or view annotation summaries, please upload some data
        in the 'Outlier Annotation' - page."""
        )
        return
    fix_random_seeds()
    data_selection(st.sidebar)
    show_annotation_summary()
    train_options()
    test_metrics()
    show_feature_importances()


def model_prediction():
    st.sidebar.title("Prediction Controls")

    if not st.session_state["data_store"]:
        st.info(
            """Please upload a dataset in order to generate and 
        interact with model predictions."""
        )
        return

    prediction_options(st.sidebar)

    if not st.session_state["inference_results"]:
        st.info(
            "To see and interact with model predictions, please choose one or multiple models and datasets \
            in the sidebar, then click 'Generate Predictions.'"
        )
    for dataset_name, series_dict in st.session_state["inference_results"].items():
        for series in series_dict.keys():

            with st.expander(f"{dataset_name} - {series} - Visualization Options", expanded=True):
                st.subheader(f"{dataset_name} - {series}")
                model_choice_options(dataset_name, series)
                prediction_summary_table(dataset_name, series)
                outlier_visualization_options(dataset_name, series)
            with st.expander(f"{dataset_name} - {series} - Retraining options", expanded=True):
                retrain_options(dataset_name, series)
            if not st.session_state["models_to_visualize"][dataset_name][series]:
                continue
            with st.expander(f"{dataset_name} - {series} - Graphs", expanded=True):
                start_time, end_time = make_outlier_distribution_plot(dataset_name, series)
                if start_time is None:
                    if f"last_clicked_range_{dataset_name}_{series}" in st.session_state:
                        start_time, end_time = st.session_state[
                            f"last_clicked_range_{dataset_name}_{series}"
                        ]
                    else:
                        continue
                st.checkbox(
                    "Area select: Only select predicted outliers",
                    True,
                    key=f"only_select_outliers_{dataset_name}_{series}",
                    help="""To select multiple points at once you might want to use
                    one of the area select options.  
                    This checkbox controls whether these selection methods only select
                    datapoints where outliers were predicted (markers) or all datapoints
                    in range.""",
                )
                clicked_point = make_time_range_outlier_plot(
                    dataset_name, series, start_time, end_time
                )
                # pass in dataset_name to activate checking for "select only outliers"
                process_data_from_echarts_plot(clicked_point, dataset_name, series)
                correction_options(dataset_name, series)


def instructions():
    tabs = st.tabs(list(INSTRUCTION_DICT.keys()))

    for i, (k, instruction_func) in enumerate(INSTRUCTION_DICT.items()):
        with tabs[i]:
            st.header(k)
            instruction_func()


def annotation_suggestion():
    # set_session_state_items("page_index", FUNC_IDX_MAPPING["Annotation Suggestion"])

    # for now only allow annotation suggestion for models trained in the current session
    if not (("most_recent_model" in st.session_state) and (st.session_state["inference_results"])):
        st.info(
            """Annotation suggestion requires at least one model trained this session.  
        If a model was trained and it was used to generate predictions for a dataset, annotation suggestion will become available here."""
        )
        return

    st.sidebar.title("Suggestion Controls")
    session_models = sorted(st.session_state["models_trained_this_session"], reverse=True)
    model = st.sidebar.selectbox("Choose model", options=session_models)

    num_points = st.sidebar.slider(
        "Number of points to show on each side of the candidate",
        min_value=10,
        max_value=1000,
        value=50,
        step=10,
        key="suggestion_number_of_points",
    )

    dataset = st.session_state["model_library"][model]["trained_on_dataset"]
    series = st.session_state["model_library"][model]["trained_on_series"]

    state = get_as(dataset, series)

    base_df = st.session_state["inference_results"][dataset][series]
    # candidates are all datapoints that are not already annotated
    df_not_annotated: pd.DataFrame = base_df[
        ~base_df.index.isin(state.all_indices - state.selection)
    ]
    # also filter out skipped values
    df_not_annotated = df_not_annotated[
        ~df_not_annotated.index.isin(
            [p[0] for p in st.session_state["suggested_points_with_annotation"][dataset][series]]
        )
    ]
    # sort by "model uncertainty" => lower = more uncertain
    df_not_annotated.sort_values(f"certainty_{model}", inplace=True)
    int_idx = state.df.index.get_loc(df_not_annotated.index[0])
    start_time = state.df.index[max(int_idx - num_points, 0)]
    end_time = state.df.index[min(int_idx + num_points, len(state.df) - 1)]
    x_value = state.df.index[int_idx]
    y_value = state.df[series][int_idx]

    cert_value = int((df_not_annotated[f"certainty_{model}"][0] / 0.5) * 100)

    c1, c2 = st.columns([6, 1])
    with c1:
        make_annotation_suggestion_plot(start_time, end_time, dataset, series, (x_value, y_value))
    c2.markdown("<br>", unsafe_allow_html=True)
    c2.button(
        "Yes", on_click=annotation_suggestion_callback, args=(dataset, series, "outlier", x_value)
    )
    c2.button(
        "No", on_click=annotation_suggestion_callback, args=(dataset, series, "normal", x_value)
    )
    c2.metric("Model certainty", f"{cert_value} %")
    c2.button(
        "Skip", on_click=annotation_suggestion_callback, args=(dataset, series, "skipped", x_value)
    )
    c2.button(
        "Back to previous",
        on_click=back_to_previous_suggestion_callback,
        args=(dataset, series),
        disabled=len(st.session_state["suggested_points_with_annotation"][dataset][series]) < 1,
        help="Go back to the previous prompt (thereby removing its label).",
    )

    st.info(
        """The presented datapoints are chosen based on the 'model certainty'
    of the predictions. Uncertainty simply describes the degree of disagreement between
    the individual tree classifiers. The points with the lowest certainty will be prompted first."""
    )

    was_retrained = retrain_options(dataset, series)
    # if model was retrained, we need to rerun to switch to the predictions page
    if was_retrained:
        st.experimental_rerun()


def annotation_suggestion_callback(dataset, series, annotation_type, value):
    state = get_as(dataset, series)
    if annotation_type in state.data:
        state.update_data(annotation_type, [value])
    st.session_state["suggested_points_with_annotation"][dataset][series].append(
        (value, annotation_type)
    )


def back_to_previous_suggestion_callback(dataset, series):
    value_to_remove = st.session_state["suggested_points_with_annotation"][dataset][series].pop()
    state = get_as(dataset, series)
    key = value_to_remove[1]
    if key == "skipped":
        return

    state.data[key].discard(value_to_remove[0])
    state._update_df(key)
    state._update_plot_df(key)


def data_download():
    removal_possible = True
    if not (st.session_state["model_library"] and st.session_state["data_store"]):
        st.info(
            """Once a model was trained or uploaded and a dataset was created,
        you will be able to use your models to remove outliers and download
        the resulting data here. """
        )
        removal_possible = False

    st.sidebar.subheader("Download Controls")

    if removal_possible:
        dataset = st.sidebar.selectbox(
            "Select source dataset",
            options=list(st.session_state["data_store"].keys()),
            index=list(st.session_state["data_store"].keys()).index(
                st.session_state["current_dataset"]
            ),
            disabled=len(st.session_state["data_store"]) < 2,
            key="download_dataset",
        )

        series = st.sidebar.multiselect(
            "Select series to remove outliers from",
            options=list(st.session_state["data_store"][dataset].keys()),
            default=st.session_state["current_series"][dataset],
            disabled=len(st.session_state["data_store"][dataset]) < 2,
            key="download_series",
            help="""The final dataset will keep all columns it had when it was uploaded.  
            Here you can choose which of those columns should be cleaned of outliers.  
            You might want to use different models to clean different series.""",
        )
        if not series:
            st.sidebar.warning("Please select at least one series.")
            return

        st.sidebar.selectbox(
            "Select model to use for outlier identification",
            options=sorted(st.session_state["model_library"].keys()),
            index=sorted(st.session_state["model_library"].keys()).index(
                st.session_state["most_recent_model"]
            )
            if "most_recent_model" in st.session_state
            else len(st.session_state["model_library"]) - 1,
            disabled=len(st.session_state["model_library"]) < 2,
            key="download_model",
        )

        method = st.sidebar.radio(
            "Select how to handle predicted outliers",
            options=list(REMOVAL_METHODS.keys()),
            key="download_method",
        )

        st.sidebar.button(
            "Update Preview" if "df_before" in st.session_state else "Preview",
            on_click=remove_outliers,
            help="""Creates a preview by sampling three predicted outliers per  
            series and overlaying a series with the outliers removed according  
            to the chosen method.""",
        )

        if f"df_after_{dataset}" in st.session_state:
            with st.sidebar.expander("Save cleaned data as dataset", expanded=True):
                if "_cleaned_" in dataset:
                    stem = dataset.split("_cleaned")[0]
                    default_ds_name = (
                        f"{stem}_cleaned_{st.session_state['cleaned_dataset_counter'][stem]}"
                    )
                else:
                    default_ds_name = (
                        f"{dataset}_cleaned_{st.session_state['cleaned_dataset_counter'][dataset]}"
                    )
                new_ds_name = st.text_input(
                    "Enter dataset name",
                    value=default_ds_name,
                    max_chars=30,
                    key="cleaned_dataset_to_add",
                    help="""Save cleaned data as a new dataset.  
                    You can then use this dataset to remove more outliers using  
                    another model, add further annotations or download it.""",
                )
                disabled = False
                if (new_ds_name == "") or (new_ds_name == " "):
                    st.warning("Please enter a name.")
                    disabled = True
                if new_ds_name in st.session_state["data_store"]:
                    st.warning("A dataset with this name already exists.")
                    disabled = True
                st.button(
                    "Add dataset", on_click=add_cleaned_dataset, args=(dataset,), disabled=disabled
                )

    if not st.session_state["data_store"]:
        return
    with st.sidebar.expander("Download a dataset", expanded=True):
        ds_to_download = st.selectbox(
            "Select dataset to download",
            options=sorted(st.session_state["data_store"].keys()),
            index=list(st.session_state["data_store"].keys()).index(
                st.session_state["current_dataset"]
            ),
            disabled=len(st.session_state["data_store"]) < 2,
        )
        file_format = st.radio("Select output file format", options=["csv", "xlsx"])

        file_name = f"{ds_to_download}.{file_format}"

        st.download_button(
            "Download dataset",
            file_name=file_name,
            data=get_data_to_download(ds_to_download, file_format),
            on_click=logging.info,
            args=("A dataset was successfully downloaded.",),
        )


@st.cache
def get_data_to_download(dataset: str, file_format: str):
    df_download = pd.DataFrame()
    for df_series in st.session_state["data_store"][dataset].values():
        df_download = df_download.merge(
            df_series,
            left_index=True,
            right_index=True,
            how="outer",
        )

    df_download.dropna(how="all", inplace=True)

    if file_format == "csv":
        return df_download.to_csv().encode("utf-8")

    elif file_format == "xlsx":
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine="xlsxwriter")
        df_download.to_excel(writer, sheet_name="Sheet1")
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]
        format1 = workbook.add_format({"num_format": "0.00"})
        worksheet.set_column("A:A", None, format1)
        writer.save()
        return output.getvalue()


FUNC_MAPPING = {
    "1. Outlier Annotation": outlier_annotation,
    "2. Model Training": model_training,
    "3. Model Prediction": model_prediction,
    "Annotation Suggestion": annotation_suggestion,
    "Data Download": data_download,
    "Instructions": instructions,
}

FUNC_IDX_MAPPING = {k: i for i, k in enumerate(FUNC_MAPPING.keys())}


def add_cleaned_dataset(original_dataset: str):
    ds_name = st.session_state["cleaned_dataset_to_add"]
    if (ds_name == "") or (ds_name == " ") or (ds_name in st.session_state["data_store"]):
        return

    df_new = st.session_state[f"df_after_{original_dataset}"]
    if "_cleaned_" in original_dataset:
        stem = original_dataset.split("_cleaned")[0]
        st.session_state["cleaned_dataset_counter"][stem] += 1
    else:
        st.session_state["cleaned_dataset_counter"][original_dataset] += 1
    add_new_data(df_new, ds_name)


def remove_outliers():
    dataset: str = st.session_state["download_dataset"]
    series: List = st.session_state["download_series"]
    model: str = st.session_state["download_model"]
    method: str = st.session_state["download_method"]

    st.session_state["prediction_data"] = defaultdict(list)
    st.session_state["prediction_data"][dataset] = series
    st.session_state["prediction_models"][model] = st.session_state["model_library"][model]

    get_predictions_callback()
    set_session_state_items("page_index", FUNC_IDX_MAPPING["Data Download"])

    # reconstruct a single dataframe from all series
    df = pd.DataFrame()
    for s in series:
        df_to_add: pd.DataFrame = st.session_state["inference_results"][dataset][s]
        model_columns = [c for c in df_to_add.columns if model in c]
        df_to_add = df_to_add[model_columns + [s]]
        df_to_add = df_to_add.rename(columns=lambda c: f"{s}_{c}" if c in model_columns else c)
        df = df.merge(
            df_to_add,
            left_index=True,
            right_index=True,
            how="outer",
        )

    # add other series that belong to the dataset, but should not be cleaned
    other_series = [s for s in st.session_state["data_store"][dataset] if s not in series]
    for s in other_series:
        df = df.merge(
            st.session_state["data_store"][dataset][s],
            left_index=True,
            right_index=True,
            how="outer",
        )

    df.sort_index(inplace=True, ascending=True)
    df_before = df.copy(deep=True)
    df_new = REMOVAL_METHODS[method](df)

    df_new.dropna(how="all", inplace=True, subset=series + other_series)

    make_removed_outliers_example_plots(df_before, df_new)

    st.session_state[f"df_before_{dataset}"] = df_before[series + other_series]
    st.session_state[f"df_after_{dataset}"] = df_new[series + other_series]


def remove_all_outliers(df: pd.DataFrame) -> pd.DataFrame:
    series: List = st.session_state["download_series"]
    model: str = st.session_state["download_model"]

    for s in series:
        model_column = f"{s}_{model}"
        mask = df[model_column] == 1
        df.loc[mask, s] = np.nan

    return df


def linear_outlier_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    series: List = st.session_state["download_series"]
    model: str = st.session_state["download_model"]

    for s in series:
        model_column = f"{s}_{model}"
        mask = df[model_column] == 1
        df.loc[mask, s] = np.nan
        df[s] = df[s].interpolate()

    return df


REMOVAL_METHODS = {
    "Delete outliers completely": remove_all_outliers,
    "Linear interpolation": linear_outlier_interpolation,
}


def retrain_options(dataset: str, series: str):
    deltas = get_annotation_deltas(dataset, series)
    if (
        (f"last_model_name_{dataset}_{series}" not in st.session_state)
        or (f"old_number_annotations_{dataset}_{series}" not in st.session_state)
        or (not any(abs(v) > 0 for v in deltas.values()))
    ):
        c1, c2 = st.columns([1, 3])
        c2.info(
            """Once you have added further annotations, you can use this button
        to quickly retrain the last model and use it to generate new predictions."""
        )
        disabled = True

    else:
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

        c2.metric("New marked Train Outlier", deltas["outlier"])
        c3.metric("New marked Train Normal", deltas["normal"])
        c4.metric("New marked Test Outlier", deltas["test_outlier"])
        c5.metric("New marked Test Normal", deltas["test_normal"])
        disabled = False

    was_retrained = c1.button(
        "Retrain most recent model with new data",
        key=f"retrain_{dataset}_{series}",
        on_click=retrain_and_repredict,
        args=(dataset, series),
        disabled=disabled,
        help="""Use this button to simplify your workflow and retrain the last model trained on this dataset & series
            (using the added annotation data), generate new predictions and visualize them.""",
    )
    return was_retrained


def dataset_choice_callback():
    ds = st.session_state["dataset_choice"]
    st.session_state["current_dataset"] = ds


def series_choice_callback(dataset: str):
    series = st.session_state["column_choice"]
    st.session_state["current_series"][dataset] = series
    set_session_state_items("expand_data_selection", False)


def data_selection(base_obj=None):
    obj = base_obj or st

    with obj.expander("Data Selection", expanded=st.session_state["expand_data_selection"]):

        datasets = list(st.session_state["data_store"].keys())
        dataset_choice = st.selectbox(
            label="Select dataset",
            options=datasets,
            index=datasets.index(st.session_state["current_dataset"])
            if st.session_state.get("current_dataset") is not None
            else len(datasets) - 1,
            disabled=len(st.session_state["data_store"]) < 2,
            on_change=dataset_choice_callback,
            key="dataset_choice",
        )
        if not dataset_choice:
            return
        columns = sorted(st.session_state["data_store"][dataset_choice].keys())
        current_series = st.session_state["current_series"].get(dataset_choice)
        column_choice = st.selectbox(
            "Select Series",
            columns,
            index=columns.index(current_series)
            if current_series is not None and current_series in columns
            else len(columns) - 1,
            disabled=len(st.session_state["data_store"][dataset_choice]) < 2,
            on_change=series_choice_callback,
            args=(dataset_choice,),
            key="column_choice",
        )

        st.session_state[f"column_choice_{dataset_choice}"] = column_choice


def retrain_and_repredict(dataset: str, series: str, base_obj=None):

    obj = base_obj or st
    fix_random_seeds()

    train_model(obj, dataset, series)

    get_predictions_callback(obj)

    st.session_state["suggested_points_with_annotation"][dataset][series].clear()


def show_all_callback():
    state = get_as()

    st.session_state["start_time"] = state.start
    st.session_state["end_time"] = state.end
    set_session_state_items("use_date_picker", False)


def back_to_selected_time_callback():
    state = get_as()
    state.update_plot(st.session_state["start_time"], st.session_state["end_time"])
    set_session_state_items("use_date_picker", True)


def calendar_callback():
    dates = st.session_state["calendar"]
    if len(dates) < 2:
        return
    state = get_as()

    current_start_time = state.start
    current_end_time = state.end
    start_date, end_date = dates

    new_start = current_start_time.replace(
        year=start_date.year, month=start_date.month, day=start_date.day
    )
    new_end = current_end_time.replace(year=end_date.year, month=end_date.month, day=end_date.day)
    state.update_plot(new_start, new_end)


def time_callback():
    start_time = st.session_state["start_time_widget"]
    end_time = st.session_state["end_time_widget"]
    state = get_as()

    current_start_time = state.start
    current_end_time = state.end

    new_start = current_start_time.replace(hour=start_time.hour, minute=start_time.minute)
    new_end = current_end_time.replace(hour=end_time.hour, minute=end_time.minute)
    state.update_plot(new_start, new_end)


def time_range_selection(base_obj=None) -> pd.DataFrame:
    obj = base_obj or st
    state = get_as()
    if (state.df.index.max() - state.df.index.min()).days > 1:
        obj.date_input(
            "Graph Date Range",
            value=(state.start.date(), state.end.date()),
            min_value=state.df.index.min(),
            max_value=state.df.index.max(),
            on_change=calendar_callback,
            key="calendar",
        )
    inp_col_1, inp_col_2 = obj.columns(2)

    inp_col_1.time_input(
        "Start time",
        value=state.start.time(),
        on_change=time_callback,
        key="start_time_widget",
    )
    inp_col_2.time_input(
        "End time",
        value=state.end.time(),
        on_change=time_callback,
        key="end_time_widget",
    )
    if st.session_state["use_date_picker"]:
        inp_col_1.button("Show All", on_click=show_all_callback)
        c1, c2 = obj.columns(2)
        c1.button(
            "Shift back",
            on_click=shift_plot_window,
            args=("backwards",),
            help="""Moves the displayed plot time range backwards, while keeping the range equal.  
        This can be used to iterate over the dataset in chunks of any size.""",
        )
        c2.button(
            "Shift forward",
            on_click=shift_plot_window,
            args=("forward",),
            help="""Moves the displayed plot time range forward, while keeping the range equal.  
        This can be used to iterate over the dataset in chunks of any size.""",
        )

    else:
        state.update_plot(state.df.index.min(), state.df.index.max())
        obj.button(
            "Back to previous selection",
            on_click=back_to_selected_time_callback,
        )

    form = obj.form(key="time_range_form")

    form.number_input(
        "Set number of datapoints",
        min_value=2,
        max_value=len(state.df),
        value=len(state.df_plot),
        step=100,
        help="""Control the number of datapoints shown in the plot window.  
            When changed, the currently set end timestamp will remain the same.""",
        key="time_range_slider",
    )
    form.form_submit_button("Update", on_click=shift_plot_window_number_of_points)


def shift_plot_window_number_of_points():
    state = get_as()
    number_points = st.session_state["time_range_slider"]
    current_end_time = state.end
    current_end_idx = state.df.sort_index().index.get_loc(current_end_time)
    new_start_idx = current_end_idx - number_points + 1
    new_start_time = state.df.sort_index().index[new_start_idx]
    state.update_plot(start_time=new_start_time)


def shift_plot_window(direction: str):
    state = get_as()
    current_start = state.start
    current_end = state.end

    current_range = current_end - current_start

    if direction == "forward":
        new_start = current_start + current_range
        new_end = current_end + current_range
    else:
        new_start = current_start - current_range
        new_end = current_end - current_range

    if new_start < state.df.index.min():
        new_start = state.df.index.min()
        new_end = new_start + current_range
    if new_end > state.df.index.max():
        new_end = state.df.index.max()
        new_start = new_end - current_range

    state.update_plot(new_start, new_end)


def generate_example_data():
    fix_random_seeds()

    dti = pd.date_range("2018-01-01", periods=5000, freq="10min")

    # generate some simple sine data
    cycles = 50
    resolution = 5000  # how many datapoints to generate
    length = np.pi * 2 * cycles
    data = np.sin(np.arange(0, length, length / resolution))

    # add some random noise, which is not outliers
    data = data + np.random.normal(0, 0.005, data.shape)

    # add some randomly generated outliers
    oulier_idc = np.random.randint(0, data.size, 200)
    data[oulier_idc] *= np.random.normal(0, 2, oulier_idc.shape)

    data_2 = np.cos(np.arange(0, length, length / resolution))
    data_2 = data_2 + np.random.normal(0, 0.005, data_2.shape)
    oulier_idc = np.random.randint(0, data_2.size, 500)
    data_2[oulier_idc] *= np.random.normal(0, 1, oulier_idc.shape)

    df = pd.DataFrame({"Example Series 1": data, "Example Series 2": data_2}, index=dti)

    for c in df.columns:
        st.session_state["data_store"]["Example Dataset"][c] = df[[c]]
        an_st = AnnotationState("Example Dataset", c)
        st.session_state["annotation_state_store"]["Example Dataset"][c] = an_st

    st.session_state["current_dataset"] = "Example Dataset"
    st.session_state["current_series"]["Example Dataset"] = "Example Series 1"


def create_annotation_plot_buttons(base_obj=None):
    obj = base_obj or st

    state = get_as()

    disabled = len(state.selection) < 1
    if disabled:
        obj.info("Select points first, then annotate them using these buttons.")

    custom_text("Training Data", 15, True, base_obj=obj)
    c_1, c_2 = obj.columns(2)
    c_1.button(
        "Mark selection Outlier",
        on_click=state.update_data,
        args=("outlier",),
        disabled=disabled,
        help=None
        if disabled
        else """Use this button to annotate all selected points as outliers
        to be used for training.""",
    )
    c_2.button(
        "Mark selection Normal",
        on_click=state.update_data,
        args=("normal",),
        disabled=disabled,
        help=None
        if disabled
        else """Use this button to annotate all selected points as normal points
        to be used for training.""",
    )
    custom_text("Test Data", 15, True, base_obj=obj)
    c_1, c_2 = obj.columns(2)
    c_1.button(
        "Mark selection Outlier",
        on_click=state.update_data,
        args=("test_outlier",),
        key="mark_test_outlier",
        disabled=disabled,
        help=None
        if disabled
        else """Use this button to annotate all selected points as outliers
        to be used in the test set.  
        This means they will not be used for training, but instead to evalutate the
        performance of the trained model.""",
    )
    c_2.button(
        "Mark selection Normal",
        on_click=state.update_data,
        args=("test_normal",),
        key="mark_test_normal",
        disabled=disabled,
        help=None
        if disabled
        else """Use this button to annotate all selected points as normal
        points to be used in the test set.  
        This means they will not be used for training, but instead to evalutate the
        performance of the trained model.""",
    )
    obj.markdown("***")
    c_1, c_2 = obj.columns(2)

    c_1.button("Clear Selection", on_click=state.clear_selection, disabled=len(state.selection) < 1)
    c_2.button("Clear All", on_click=state.clear_all, disabled=len(state.all_indices) < 1)


def validate_uploaded_file_contents(base_obj=None):
    obj = base_obj or st
    uploaded_files = st.session_state["uploaded_annotation_files"]
    if not uploaded_files:
        return
    state = get_as(return_all_columns=True)
    data_to_load_if_file_ok = defaultdict(list)
    # loop through files if there are multiple
    for file_number, uploaded_file in enumerate(uploaded_files):
        datapoints_tracker = {}
        file_failed = False
        data = pickle.loads(uploaded_file.getvalue())
        # loop through series of uploaded data
        for column, data_dict in data.items():
            if file_failed:
                break
            total_datapoints_loaded = 0
            if column not in state:
                obj.warning(
                    f"""File {uploaded_file.name}:  
                Column {column} not found in current dataset {st.session_state["dataset_choice"]}, skipping."""
                )
                continue
            # loop through different annotation types (outlier, normal, etc.)
            for df in data_dict.values():
                if df.empty:
                    continue
                values_match = (
                    df[column]
                    .astype(float)
                    .round(10)
                    .isin(state[column].df[column].round(10).values)
                    .all()
                )
                index_match = df.index.isin(state[column].df.index).all()

                if (not values_match) or (not index_match):
                    file_failed = True
                    break
                total_datapoints_loaded += len(df)

            if not file_failed:
                datapoints_tracker[column] = total_datapoints_loaded
                data_to_load_if_file_ok[column].append(data_dict)

        if file_failed:
            obj.error(
                f"""{uploaded_file.name}: Did not pass validation for loaded dataset.  
            Either index or values of loaded annotations do not match."""
            )
        else:
            for c, count in datapoints_tracker.items():
                obj.success(
                    f"File {file_number + 1}: Loaded {count} annotations for series {c}.", icon="✅"
                )

            st.session_state["uploaded_annotation_data"][uploaded_file.name] = data

    st.session_state["uploaded_annotation_data"] = data_to_load_if_file_ok


def annotation_file_upload_callback(base_obj=None):
    obj = base_obj or st
    validate_uploaded_file_contents(obj)

    for column, data_list in st.session_state["uploaded_annotation_data"].items():
        state = get_as(column=column)
        for data in data_list:
            for key, df in data.items():
                state.update_data(key, df.index.to_list())


def dev_options(base_obj=None):
    if os.environ.get("TSOD_DEV_MODE", "false") == "false":
        return nullcontext()
    obj = base_obj or st
    exp = obj.expander("Dev Options")
    with exp:
        dev_col_1, dev_col_2 = st.columns(2)
        profile = dev_col_1.checkbox("Profile Code", value=False)
        show_total_mem = dev_col_2.checkbox("Show total Memory Usage", value=False)
        show_ss = dev_col_1.button("Show full Session State")
        show_mem = dev_col_2.checkbox("Show mem. usage of dfs")
        search_str = dev_col_1.text_input("Search SS", max_chars=25, value="")

    if len(search_str) > 1:
        recursive_ss_search(search_str, base_obj=exp)
    if show_mem:
        st.write(ss_recursive_df_memory_usage())
    if show_total_mem:
        show_memory_usage(exp)

    if show_ss:
        st.write(st.session_state)

    return Profiler() if profile else nullcontext()


def create_save_load_buttons(base_obj=None):
    obj = base_obj or st
    state = get_as()
    obj.info(
        f"""Current dataset:  
        {state.dataset}"""
    )
    c_1, c_2 = obj.columns(2)

    file_name = (
        f"{state.dataset}_Annotations_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.bin"
    )

    if not state.all_indices:
        c_1.warning("No Annotations have been added for this dataset.")
    c_1.download_button(
        "Download Annotations",
        pickle.dumps(
            {
                k: v._download_data
                for k, v in get_as(state.dataset, return_all_columns=True).items()
                if (v._download_data) and (k != "selected")
            }
        ),
        file_name=file_name,
        disabled=not state.all_indices,
    )

    c_2.file_uploader(
        "Upload Annotations",
        key="uploaded_annotation_files",
        type="bin",
        on_change=annotation_file_upload_callback,
        args=(obj,),
        accept_multiple_files=True,
    )
    obj.markdown("***")


def get_annotation_deltas(dataset: str, series: str):
    annotation_keys = ["outlier", "normal", "test_outlier", "test_normal"]
    state = get_as(dataset, series)
    if f"old_number_annotations_{dataset}_{series}" in st.session_state:
        deltas = {
            k: len(state.data[k])
            - st.session_state[f"old_number_annotations_{dataset}_{series}"][k]
            for k in annotation_keys
        }
    else:
        deltas = {k: None for k in annotation_keys}

    return deltas


def show_annotation_summary(base_obj=None):
    obj = base_obj or st
    with obj.expander("Annotation Info", expanded=True):
        state = get_as()
        dataset = state.dataset
        series = state.column
        obj.subheader(f"Annotation summary - {state.dataset} - {state.column}")
        c1, c2, c3 = st.columns([1, 1, 2])

        custom_text("Training Data", 15, base_obj=c1, centered=False)
        custom_text("Test Data", 15, base_obj=c2, centered=False)

        deltas = get_annotation_deltas(dataset, series)

        c1.metric(
            "Total labelled Outlier",
            len(state.outlier),
            delta=deltas["outlier"],
            delta_color="normal" if deltas["outlier"] != 0 else "off",
        )
        c1.metric(
            "Total labelled Normal",
            len(state.normal),
            delta=deltas["normal"],
            delta_color="normal" if deltas["normal"] != 0 else "off",
        )
        c2.metric(
            "Total labelled Outlier",
            len(state.test_outlier),
            delta=deltas["test_outlier"],
            delta_color="normal" if deltas["test_outlier"] != 0 else "off",
            help="""Test data will not be used for training.  
            It can be used to measure how well a model performs on new data not seen during training. """,
        )
        c2.metric(
            "Total labelled Normal",
            len(state.test_normal),
            delta=deltas["test_normal"],
            delta_color="normal" if deltas["test_normal"] != 0 else "off",
        )

        if not (len(state.data["outlier"]) and len(state.data["normal"]) > 1):
            c3.warning(
                """In order to train an outlier prediction model, please annotate at least one outlier 
                and two normal points as training data.  
                You can add annotations by  
                1) Marking points in the 'Outlier Annotation' - page  
                2) Uploading a previously created annotation file  
                3) Correcting model predictions in the 'Model prediction' - page  
                Then choose a method and parameters and click on 'Train Outlier Model' in the sidebar."""
            )

    if "classifier" in st.session_state:
        obj.subheader(f"Current model: {st.session_state[f'last_model_name_{dataset}_{series}']}")


def save_method():
    method = st.session_state["current_method_choice"]
    st.session_state["last_method_choice"] = method


def train_options(base_obj=None):
    obj = base_obj or st

    state = get_as()
    dataset = state.dataset
    series = state.column
    if not (len(state.data["outlier"]) and len(state.data["normal"]) > 1):
        return

    with st.sidebar.expander("Modelling Options", expanded=True):
        st.selectbox(
            "Choose OD method",
            options=list(MODEL_OPTIONS.keys()),
            key="current_method_choice",
            format_func=lambda x: MODEL_OPTIONS.get(x),
            on_change=save_method,
            help="Here you can choose what type of outlier detection approach to use.",
        )

    with st.sidebar.expander("Feature Options", expanded=True):
        st.number_input(
            "Points before",
            min_value=1,
            value=st.session_state.get("old_points_before")
            if st.session_state.get("old_points_before") is not None
            else 10,
            key="number_points_before",
            step=5,
            help="How many points before each annotated point to include in its feature set.",
        )
        st.number_input(
            "Points after",
            min_value=0,
            value=st.session_state.get("old_points_after")
            if st.session_state.get("old_points_after") is not None
            else 0,
            key="number_points_after",
            step=5,
            help="How many points after each annotated point to include in its feature set.",
        )

    auto_generate = st.sidebar.checkbox(
        "Auto generate predictions for entire annotation series",
        value=True,
        help="""Often the next step after training a model is looking at its prediction distribution
        over the entire annotated dataset-series combination. 
        By checking this box, predictions for the relevant series will be generated after training, 
        so the results can be viewed / compared straight away in the 'Model Prediction' - page.""",
    )
    train_button = st.sidebar.button("Train Outlier Model", key="train_button")

    if train_button:
        st.session_state["old_points_before"] = st.session_state["number_points_before"]
        st.session_state["old_points_after"] = st.session_state["number_points_after"]

        train_model(obj)
        if auto_generate:
            get_predictions_callback(obj)
        set_session_state_items("page_index", FUNC_IDX_MAPPING["2. Model Training"])
        st.experimental_rerun()
    if f"last_model_name_{dataset}_{series}" in st.session_state:
        st.sidebar.success(
            f"{st.session_state[f'last_model_name_{dataset}_{series}']} finished training."
        )
        if (
            st.session_state[f"last_model_name_{dataset}_{series}"]
            in st.session_state["inference_results"][dataset][series].columns
        ):
            st.sidebar.success(
                f"Predictions for model {st.session_state[f'last_model_name_{dataset}_{series}']} \
                have been generated and can be viewed on the 'Model Prediction' - page."
            )
    if st.session_state["models_trained_this_session"]:
        with st.sidebar.expander("Model Download", expanded=True):
            model_choice = st.selectbox(
                "Choose Model",
                options=sorted(st.session_state["models_trained_this_session"], reverse=True),
            )
            st.download_button(
                "Download model",
                pickle.dumps(st.session_state["model_library"][model_choice]),
                f"{model_choice}.pkl",
            )


def get_predictions_callback(obj=None):
    # set_session_state_items("hide_choice_menus", True)
    set_session_state_items("page_index", FUNC_IDX_MAPPING["3. Model Prediction"])
    get_model_predictions(obj)
    set_session_state_items("prediction_models", {})


def add_uploaded_models(base_obj=None):
    obj = base_obj or st
    # set_session_state_items("prediction_models", {})
    if not st.session_state.current_uploaded_models:
        return
    for data in st.session_state.current_uploaded_models:
        model_data = pickle.loads(data.read())
        model = model_data["model"]
        if not hasattr(model, "predict"):
            obj.error(
                "The uploaded object can not be used for prediction (does not implement 'predict' method)."
            )
            continue

        st.session_state["prediction_models"][data.name] = model_data
        st.session_state["model_library"][data.name] = model_data


def add_session_models():
    to_add = st.session_state["session_models_to_add"]
    uploaded_models = {
        k: v for k, v in st.session_state["prediction_models"].items() if k.endswith(".pkl")
    }
    st.session_state["prediction_models"] = {
        k: st.session_state["model_library"][k] for k in to_add
    }
    st.session_state["prediction_models"].update(uploaded_models)


def add_session_dataset():
    session_ds = st.session_state["pred_session_ds_choice"]
    session_cols = st.session_state["pred_session_col_choice"]

    st.session_state["prediction_data"][session_ds] = session_cols


def prediction_options(base_obj=None):
    obj = base_obj or st
    _, c, _ = obj.columns([2, 5, 2])
    c.button("Generate Predictions", on_click=get_predictions_callback, args=(obj,))

    with obj.expander("Model Choice", expanded=True):
        st.subheader("Choose models for generating predictions")
        st.multiselect(
            "Select models trained this session",
            options=sorted(
                [k for k in st.session_state["model_library"].keys() if not k.endswith(".pkl")]
            ),
            default=[
                k for k in st.session_state["prediction_models"].keys() if not k.endswith(".pkl")
            ],
            on_change=add_session_models,
            key="session_models_to_add",
            help="When a new model is trained, it is automatically pre-selected.",
        )
        st.file_uploader(
            "Or / and select model from disk (optional)",
            type="pkl",
            on_change=add_uploaded_models,
            key="current_uploaded_models",
            args=(obj,),
            accept_multiple_files=True,
        )

        st.subheader("Selected models:")
        st.json(list(st.session_state.prediction_models.keys()))
        if st.session_state.prediction_models:
            st.button(
                "Clear selection", on_click=set_session_state_items, args=("prediction_models", {})
            )
    with obj.expander("Data Choice", expanded=True):
        st.subheader("Select data for generating predictions")
        ds_options = list(st.session_state["data_store"].keys())
        if "most_recent_model" in st.session_state:
            idx = ds_options.index(
                st.session_state["model_library"][st.session_state["most_recent_model"]][
                    "trained_on_dataset"
                ]
            )
        else:
            if len(ds_options):
                idx = len(ds_options) - 1
            else:
                idx = 0
        ds_choice = st.selectbox(
            label="Select datasets uploaded this session",
            options=ds_options,
            index=idx,
            disabled=len(st.session_state["data_store"]) < 2,
            key="pred_session_ds_choice",
        )
        if ds_choice:
            col_options = list(st.session_state["data_store"][ds_choice].keys())
            if "most_recent_model" in st.session_state:
                if (
                    ds_choice
                    == st.session_state["model_library"][st.session_state["most_recent_model"]][
                        "trained_on_dataset"
                    ]
                ):
                    default = st.session_state["model_library"][
                        st.session_state["most_recent_model"]
                    ]["trained_on_series"]
                else:
                    default = st.session_state["prediction_data"].get(ds_choice)
            else:
                default = st.session_state["prediction_data"].get(ds_choice)

            session_ds_columns = st.multiselect(
                "Pick series",
                options=col_options,
                default=default,
                on_change=add_session_dataset,
                key="pred_session_col_choice",
            )
        st.subheader("Selected Series:")
        st.json(st.session_state["prediction_data"])
        # st.json({k: list(v.keys()) for k, v in st.session_state["prediction_data"].items()})
        if st.session_state["prediction_data"]:
            st.button(
                "Clear selection",
                on_click=set_session_state_items,
                args=("prediction_data", {}),
                key="data_clear",
            )
    _, c, _ = obj.columns([2, 5, 2])
    c.button(
        "Generate Predictions", on_click=get_predictions_callback, args=(obj,), key="pred_btn_2"
    )


def remove_model_to_visualize(dataset_name, series, model_name):
    st.session_state["models_to_visualize"][dataset_name][series].discard(model_name)

    # if not recursive_length_count(st.session_state["models_to_visualize"]):
    # st.session_state["hide_choice_menus"] = False


def prediction_summary_table(dataset_name: str, series: str, base_obj=None):
    obj = base_obj or st

    DEFAULT_MARKER_COLORS = ["#e88b0b", "#1778dc", "#1bd476", "#d311e6"]
    model_names = sorted(st.session_state["models_to_visualize"][dataset_name][series])

    if not model_names:
        return

    if len(model_names) > len(DEFAULT_MARKER_COLORS):
        obj.error(
            f"Currently max. number of models is {len(DEFAULT_MARKER_COLORS)}, got {len(model_names)}"
        )
        return

    c1, c2, c4, c5, c6, c7 = obj.columns([5, 3, 6, 3, 3, 3])
    custom_text("Model Name", base_obj=c1, font_size=15, centered=True)
    custom_text("Params", base_obj=c4, font_size=15, centered=False)
    custom_text("Predicted Outliers", base_obj=c5, font_size=15, centered=False)
    custom_text("Predicted Normal", base_obj=c6, font_size=15, centered=False)
    custom_text("Choose Plot Color", base_obj=c7, font_size=15, centered=False)

    obj.markdown("***")

    for i, model in enumerate(model_names):
        _local_obj = obj.container()
        c1, c2, c4, c5, c6, c7 = _local_obj.columns([5, 3, 6, 3, 3, 3])
        custom_text(model, base_obj=c1, font_size=15, centered=True)
        c2.button(
            "Remove",
            key=f"remove_{model}_{dataset_name}_{series}",
            on_click=remove_model_to_visualize,
            args=(dataset_name, series, model),
        )
        c4.json(st.session_state["model_library"][model]["params"], expanded=False)
        custom_text(
            st.session_state["number_outliers"][dataset_name][series][model],
            base_obj=c5,
            font_size=20,
            centered=False,
        )
        custom_text(
            len(st.session_state["inference_results"][dataset_name][series])
            - st.session_state["number_outliers"][dataset_name][series][model],
            base_obj=c6,
            font_size=20,
            centered=False,
        )
        c7.color_picker(
            model,
            key=f"color_{model}_{dataset_name}_{series}",
            label_visibility="collapsed",
            value=DEFAULT_MARKER_COLORS[i],
        )
        obj.markdown("***")


def test_metrics(base_obj=None):
    obj = base_obj or st

    state = get_as()
    dataset = state.dataset
    series = state.column
    if f"last_model_name_{dataset}_{series}" not in st.session_state:
        return

    custom_text(
        f"Most recent model: {st.session_state[f'last_model_name_{dataset}_{series}']}",
        base_obj=obj,
    )
    obj.info(
        """All displayed metrics have a max value of 1 (best possible result)
    and min value of 0 (worst possible result). Hover over the question mark next to the metrics to get info on what they mean."""
    )
    c1, c2, c3, c4 = obj.columns(4)

    current_train_metrics = st.session_state[f"current_model_train_metrics_{dataset}_{series}"]
    prec = current_train_metrics["precision"]
    rec = current_train_metrics["recall"]
    f1 = current_train_metrics["f1"]

    if f"previous_model_train_metrics_{dataset}_{series}" in st.session_state:
        old_metrics = st.session_state[f"previous_model_train_metrics_{dataset}_{series}"]
        old_prec = old_metrics["precision"]
        old_rec = old_metrics["recall"]
        old_f1 = old_metrics["f1"]

        out_prec_diff = (prec[1] - old_prec[1]).round(3)
        out_rec_diff = (rec[1] - old_rec[1]).round(3)
        out_f1_diff = (f1[1] - old_f1[1]).round(3)
        norm_prec_diff = (prec[0] - old_prec[0]).round(3)
        norm_rec_diff = (rec[0] - old_rec[0]).round(3)
        norm_f1_diff = (f1[0] - old_f1[0]).round(3)

    else:
        out_prec_diff, out_rec_diff, out_f1_diff, norm_prec_diff, norm_rec_diff, norm_f1_diff = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    with c1.expander("Train Set Outlier Metrics", expanded=True):
        st.metric(
            "Precision Score",
            prec[1],
            delta=out_prec_diff,
            delta_color="normal" if out_prec_diff != 0.0 else "off",
            help=f"""The ratio of true predicted positives to total predicted positives for the train set outliers.  
            Represents the ability not to classify a normal sample as an outlier.  
            Your score of {prec[1]} means that for the training set, {int(prec[1] * 100)}% of your model's predicted outliers were correct (not labelled as normal points).""",
        )
        st.metric(
            "Recall Score",
            rec[1],
            delta=out_rec_diff,
            delta_color="normal" if out_rec_diff != 0.0 else "off",
            help=f"""The ratio of true predicted positives to total positives for the train set outliers.  
            Represents the ability to correctly predict all the outliers.  
            Your score of {rec[1]} means that for the training set, your model has correctly predicted {int(rec[1] * 100)}% of the annotated outliers.""",
        )
        st.metric(
            "F1 Score",
            f1[1],
            delta=out_f1_diff,
            delta_color="normal" if out_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the train set outliers.",
        )
    with c2.expander("Train Set Normal Metrics", expanded=True):
        st.metric(
            "Precision Score",
            prec[0],
            delta=norm_prec_diff,
            delta_color="normal" if norm_prec_diff != 0.0 else "off",
            help=f"""The ratio of true predicted positives to total predicted positives for the train set normal points.  
            Represents the ability not to classify an outlier sample as a normal point.  
            Your score of {prec[0]} means that for the training set, {int(prec[0] * 100)}% of your model's predicted normal points were correct (not labelled as outliers).""",
        )
        st.metric(
            "Recall Score",
            rec[0],
            delta=norm_rec_diff,
            delta_color="normal" if norm_rec_diff != 0.0 else "off",
            help=f"""The ratio of true predicted positives to total positives for the train set normal points.  
            Represents the ability to correctly predict all the normal points.  
            Your score of {rec[0]} means that for the training set, your model has correctly predicted {int(rec[0] * 100)}% of the annotated normal points.""",
        )
        st.metric(
            "F1 Score",
            f1[0],
            delta=norm_f1_diff,
            delta_color="normal" if norm_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the train set normal points.",
        )

    if f"current_model_test_metrics_{dataset}_{series}" not in st.session_state:
        return

    current_metrics = st.session_state[f"current_model_test_metrics_{dataset}_{series}"]
    prec = current_metrics["precision"]
    rec = current_metrics["recall"]
    f1 = current_metrics["f1"]

    if f"previous_model_test_metrics_{dataset}_{series}" in st.session_state:
        old_metrics = st.session_state[f"previous_model_test_metrics_{dataset}_{series}"]
        old_prec = old_metrics["precision"]
        old_rec = old_metrics["recall"]
        old_f1 = old_metrics["f1"]

        out_prec_diff = (
            (prec[1] - old_prec[1]).round(3) if (len(prec) == 2) and (len(old_prec) == 2) else None
        )
        out_rec_diff = (
            (rec[1] - old_rec[1]).round(3) if (len(rec) == 2) and (len(old_rec) == 2) else None
        )
        out_f1_diff = (
            (f1[1] - old_f1[1]).round(3) if (len(f1) == 2) and (len(old_f1) == 2) else None
        )
        norm_prec_diff = (
            (prec[0] - old_prec[0]).round(3) if (len(prec) == 2) and (len(old_prec) == 2) else None
        )
        norm_rec_diff = (
            (rec[0] - old_rec[0]).round(3) if (len(rec) == 2) and (len(old_rec) == 2) else None
        )
        norm_f1_diff = (
            (f1[0] - old_f1[0]).round(3) if (len(f1) == 2) and (len(old_f1) == 2) else None
        )

    else:
        out_prec_diff, out_rec_diff, out_f1_diff, norm_prec_diff, norm_rec_diff, norm_f1_diff = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    if st.session_state[f"old_number_annotations_{dataset}_{series}"]["test_outlier"] > 0:
        if st.session_state[f"old_number_annotations_{dataset}_{series}"]["test_normal"] > 0:
            idx = 1
        else:
            idx = 0
        with c3.expander("Test Set Outlier Metrics", expanded=True):
            st.metric(
                "Precision Score",
                prec[idx],
                delta=out_prec_diff,
                delta_color="normal" if out_prec_diff != 0.0 else "off",
                help=f"""The ratio of true predicted positives to total predicted positives for the test set outliers.  
                Represents the ability not to classify a normal sample as an outlier.  
                Your score of {prec[idx]} means that for the test set, {int(prec[idx] * 100)}% of your model's predicted outliers were correct (not labelled as normal points).""",
            )
            st.metric(
                "Recall Score",
                rec[idx],
                delta=out_rec_diff,
                delta_color="normal" if out_rec_diff != 0.0 else "off",
                help=f"""The ratio of true predicted positives to total positives for the test set outliers.  
                Represents the ability to correctly predict all the outliers.  
                Your score of {rec[idx]} means that for the test set, your model has correctly predicted {int(rec[idx] * 100)}% of the annotated outliers.""",
            )
            st.metric(
                "F1 Score",
                f1[idx],
                delta=out_f1_diff,
                delta_color="normal" if out_f1_diff != 0.0 else "off",
                help="The harmonic mean of the precision and recall for the outliers.",
            )
    if st.session_state[f"old_number_annotations_{dataset}_{series}"]["test_normal"] > 0:
        with c4.expander("Test Set Normal Metrics", expanded=True):
            st.metric(
                "Precision Score",
                prec[0],
                delta=norm_prec_diff,
                delta_color="normal" if norm_prec_diff != 0.0 else "off",
                help=f"""The ratio of true predicted positives to total predicted positives for the test set normal points.  
                Represents the ability not to classify an outlier sample as a normal point.  
                Your score of {prec[0]} means that for the test set, {int(prec[0] * 100)}% of your model's predicted normal points were correct (not labelled as outliers).""",
            )
            st.metric(
                "Recall Score",
                rec[0],
                delta=norm_rec_diff,
                delta_color="normal" if norm_rec_diff != 0.0 else "off",
                help=f"""The ratio of true predicted positives to total positives for the test set normal points.  
                Represents the ability to correctly predict all the normal points.  
                Your score of {rec[0]} means that for the test set, your model has correctly predicted {int(rec[0] * 100)}% of the annotated normal points.""",
            )
            st.metric(
                "F1 Score",
                f1[0],
                delta=norm_f1_diff,
                delta_color="normal" if norm_f1_diff != 0.0 else "off",
                help="The harmonic mean of the precision and recall for the normal points.",
            )


def model_choice_callback(dataset_name: str, series: str):
    st.session_state["models_to_visualize"][dataset_name][series] = set(
        st.session_state[f"model_choice_{dataset_name}_{series}"]
    )


def model_choice_options(dataset_name: str, series: str):
    if (dataset_name == list(st.session_state["inference_results"].keys())[0]) and (
        series == list(st.session_state["inference_results"][dataset_name].keys())[0]
    ):
        st.info(
            f"""Below you can choose from all models which have generated 
        predictions for this series.  
        Add them to the selection to visualize their results. 
        By default, the two most recently trained models are selected."""
        )
    st.multiselect(
        "Choose models for dataset",
        sorted(st.session_state["available_models"][dataset_name][series]),
        key=f"model_choice_{dataset_name}_{series}",
        default=sorted(st.session_state["models_to_visualize"][dataset_name][series]),
        on_change=model_choice_callback,
        args=(dataset_name, series),
        max_selections=4,
    )

    st.markdown("***")


def outlier_visualization_options(dataset_name: str, series: str):
    if not st.session_state["models_to_visualize"][dataset_name][series]:
        return

    form = st.form(
        f"form_{dataset_name}_{series}",
    )
    form.info(
        "Click on a bar in the distribution plot to view all outliers \
        in that time period. Each time period is chosen so it contains the same number of datapoints."
    )
    c1, c2 = form.columns(2)
    c1.slider(
        "Number of datapoints per bar",
        value=300,
        min_value=10,
        max_value=1000,
        step=1,
        key=f"num_outliers_{dataset_name}_{series}",
        help="""Adjust the number of datapoints each bar represents.""",
    )
    c2.slider(
        "Height of figures (px)",
        value=600,
        min_value=100,
        max_value=1500,
        step=100,
        key=f"figure_height_{dataset_name}_{series}",
    )

    form.checkbox(
        "Only show time ranges containing outliers (predicted or annotated)",
        key=f"only_show_ranges_with_outliers_{dataset_name}_{series}",
        help="""Depending on how well a model is already trainied, there might be many
        time ranges that do not contain any predictied outliers.  
        By setting this option, these ranges are not included on the x axis in the distribution plot.""",
    )

    state = get_as(dataset_name, series)

    c1, c2 = form.columns(2)
    if state.test_outlier:

        c1.checkbox(
            "Hightlight missed test set outliers",
            value=True,
            key=f"highlight_test_{dataset_name}_{series}",
            help="""If this is set, time ranges that contain annotated test outliers
        which the models did not classify as such are marked for easy identification.""",
        )
    if state.outlier:
        c2.checkbox(
            "Hightlight missed train set outliers",
            value=False,
            key=f"highlight_train_{dataset_name}_{series}",
            help="""If this is set, time ranges that contain annotated train outliers
        which the models did not classify as such are marked for easy identification.""",
        )

    form.form_submit_button("Update Distribution Plot")


def show_feature_importances(base_obj=None):
    obj = base_obj or st
    state = get_as()
    dataset = state.dataset
    series = state.column
    if f"last_model_name_{dataset}_{series}" not in st.session_state:
        return

    with obj.expander("Feature Importances", expanded=True):
        c1, c2 = st.columns([2, 1])
        feature_importance_plot(c1)
        c2.dataframe(st.session_state[f"current_importances_{dataset}_{series}"])


def add_slider_selected_points(dataset_name: str, model_name: str):
    start, end = st.session_state[f"outlier_slider_{dataset_name}_{model_name}"]
    coords = st.session_state[f"current_outlier_value_store"][dataset_name][model_name]
    timestamps_to_add = {coords[i][0] for i in range(start, end + 1)}
    state = get_as()
    state.update_selected(timestamps_to_add)


def process_data_from_echarts_plot(
    clicked_point: List | dict | None, dataset_name=None, series=None, base_obj=None
):
    obj = base_obj or st
    state = get_as(dataset_name, series)
    was_updated = False

    if (clicked_point is None) or ((clicked_point[1] == "brush") and (not clicked_point[0])):
        return

    # we want to select only the outlier series, not the datapoints series.
    # This behaviour is set by a checkbox above the plot. It only effects area selection.
    if (
        (clicked_point[1] == "brush")
        and dataset_name
        and series
        and st.session_state[f"only_select_outliers_{dataset_name}_{series}"]
    ):
        model_names = sorted(st.session_state["models_to_visualize"][dataset_name][series])

        relevant_outlier_idc = {
            d["seriesName"]: d["dataIndex"]
            for d in clicked_point[0]
            if d["seriesName"] in model_names
        }
        relevant_data_points = [
            st.session_state["pred_outlier_tracker"][dataset_name][series][k]
            .iloc[v]
            .index.to_list()
            for k, v in relevant_outlier_idc.items()
        ]
        relevant_data_points = set().union(*relevant_data_points)
        was_updated = state.update_selected(relevant_data_points)

    else:
        if clicked_point[1] == "click":
            point_to_process = plot_return_value_as_datetime(clicked_point[0])
            if point_to_process:
                if point_to_process not in state.selection:
                    was_updated = state.update_selected([point_to_process])
        else:
            relevant_series = [s for s in clicked_point[0] if s["seriesName"] == "Datapoints"]
            if not relevant_series:
                return
            relevant_data_idc = relevant_series[0]["dataIndex"]
            was_updated = state.update_selected(
                state.df_plot.iloc[relevant_data_idc].index.to_list()
            )

    if was_updated:
        st.experimental_rerun()


def correction_options(dataset_name: str, series: str, base_obj=None):
    obj = base_obj or st
    obj.subheader("Prediction correction:")

    obj.info(
        """Either select individual points in the above plot or use the area select options
        (top right corner of the plot window) to select multiple points. Then add further
        annotations to correct faulty model predictions."""
    )

    # current_range = st.session_state[f"range_str_{dataset_name}"]
    # df_current_counts = st.session_state[f"current_ranges_counts_{dataset_name}"]

    # model_names = sorted(st.session_state["models_to_visualize"][dataset_name])
    # for model_name in model_names:
    #     model_counts = df_current_counts.loc[current_range, model_name]
    #     if model_counts <= 1:
    #         continue
    #     # if not st.session_state[f"current_outlier_value_store"][dataset_name].get(model_name):
    #     # continue
    #     form = st.form(f"outlier_select_form_{dataset_name}_{model_name}")
    #     c1, c2, c3, c4 = form.columns([5, 7, 1, 2])
    #     custom_text(model_name, 20, base_obj=c1)
    #     if model_counts > 1:
    #         c2.slider(
    #             model_name,
    #             min_value=1,
    #             max_value=model_counts,
    #             value=(1, model_counts),
    #             label_visibility="collapsed",
    #             key=f"outlier_slider_{dataset_name}_{model_name}",
    #         )
    #     c4.form_submit_button(
    #         "Select Outlier points",
    #         on_click=add_slider_selected_points,
    #         args=(dataset_name, model_name),
    #     )

    c1, c2, c3, c4 = obj.columns(4)

    state = get_as(dataset_name, series)

    c1.button(
        "Mark Train Outlier",
        on_click=state.update_data,
        args=("outlier",),
        kwargs={"base_obj": obj},
        key=f"pred_mark_outlier_{dataset_name}_{series}",
    )
    c2.button(
        "Mark Train Normal",
        on_click=state.update_data,
        args=("normal",),
        kwargs={"base_obj": obj},
        key=f"pred_mark_normal_{dataset_name}_{series}",
    )
    c3.button(
        "Mark Test Outlier",
        on_click=state.update_data,
        args=("test_outlier",),
        kwargs={"base_obj": obj},
        key=f"pred_mark_test_outlier_{dataset_name}_{series}",
    )
    c4.button(
        "Mark Test Normal",
        on_click=state.update_data,
        args=("test_normal",),
        kwargs={"base_obj": obj},
        key=f"pred_mark_test_normal_{dataset_name}_{series}",
    )
    c1.button(
        "Clear Selection",
        on_click=state.clear_selection,
        key=f"pred_clear_selection_{dataset_name}_{series}",
    )
    # c3.button("Clear All", on_click=state.clear_all)
