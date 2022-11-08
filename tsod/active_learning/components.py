import datetime
import pickle
from typing import Dict, List
from collections import defaultdict
import pandas as pd
import streamlit as st
from tsod.active_learning.modelling import (
    construct_training_data_RF,
    train_random_forest_classifier,
)
from tsod.active_learning.utils import (
    get_as,
    set_session_state_items,
    custom_text,
    recursive_length_count,
    MODEL_OPTIONS,
    fix_random_seeds,
    _add_to_ss_if_not_in_it,
)
from tsod.active_learning.modelling import get_model_predictions, construct_test_data_RF
from tsod.active_learning.plotting import (
    feature_importance_plot,
    get_echarts_plot_time_range,
    make_outlier_distribution_plot,
    make_time_range_outlier_plot,
)
from tsod.active_learning.data_structures import plot_return_value_as_datetime
from contextlib import nullcontext
from streamlit_profiler import Profiler
from streamlit_echarts import st_pyecharts
from tsod.active_learning.upload_data import data_uploader


def outlier_annotation():
    exp = st.sidebar.expander("Data Upload", expanded=not len(st.session_state["data_store"]))
    data_uploader(exp)
    st.sidebar.title("Annotation Controls")
    with dev_options(st.sidebar):
        data_selection(st.sidebar)
        state = get_as()
        if not state:
            return
        create_annotation_plot_buttons(st.sidebar)
        with st.sidebar.expander("Time Range Selection", expanded=True):
            time_range_selection()

        create_save_load_buttons(st.sidebar)

        plot = get_echarts_plot_time_range(
            state.start,
            state.end,
            state.column,
            True,
            "Outlier Annotation",
        )

        clicked_point = st_pyecharts(
            plot,
            height=1000,
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
    # set_session_state_items("page_index", 1)
    fix_random_seeds()
    with dev_options(st.sidebar):
        data_selection(st.sidebar)
        show_annotation_summary()
        # c1, c2, c3 = st.columns(3)
        train_options()
        test_metrics()
        show_feature_importances()


def model_prediction():
    st.sidebar.title("Prediction Controls")
    with dev_options(st.sidebar):
        prediction_options(st.sidebar)

        if not st.session_state["inference_results"]:
            st.info(
                "To see and interact with model predictions, please choose one or multiple models and datasets \
                in the sidebar, then click 'Generate Predictions.'"
            )
        for dataset_name in st.session_state["prediction_data"].keys():
            if st.session_state["inference_results"].get(dataset_name) is None:
                continue
            with st.expander(f"{dataset_name} - Visualization Options", expanded=True):
                st.subheader(dataset_name)
                model_choice_options(dataset_name)
                prediction_summary_table(dataset_name)
                outlier_visualization_options(dataset_name)
            if not st.session_state["models_to_visualize"][dataset_name]:
                continue
            with st.expander(f"{dataset_name} - Graphs", expanded=True):
                start_time, end_time = make_outlier_distribution_plot(dataset_name)
                if start_time is None:
                    if f"last_clicked_range_{dataset_name}" in st.session_state:
                        start_time, end_time = st.session_state[
                            f"last_clicked_range_{dataset_name}"
                        ]
                    else:
                        continue
                st.checkbox(
                    "Area select: Only select predicted outliers",
                    True,
                    key=f"only_select_outliers_{dataset_name}",
                )
                clicked_point = make_time_range_outlier_plot(dataset_name, start_time, end_time)
                # pass in dataset_name to activate checking for "select only outliers"
                process_data_from_echarts_plot(clicked_point, dataset_name)
                correction_options(dataset_name)


def instructions():
    with open("tsod/active_learning/instructions.md", "r") as f:
        data = f.read()

    st.markdown(data)


def data_selection(base_obj=None):
    obj = base_obj or st

    with obj.expander("Data selection", expanded=st.session_state["expand_data_selection"]):

        dataset_choice = st.selectbox(
            label="Select dataset",
            options=list(st.session_state["data_store"].keys()),
            index=0,
            # index=len(st.session_state["data_store"]) - 1,
            disabled=len(st.session_state["data_store"]) < 2,
            key="dataset_choice",
        )
        if not dataset_choice:
            return
        column_choice = st.selectbox(
            "Select Series",
            list(st.session_state["data_store"][dataset_choice].columns),
            disabled=len(st.session_state["data_store"][dataset_choice].columns) < 2,
            on_change=set_session_state_items,
            args=("expand_data_selection", False),
            key="column_choice",
        )

        st.session_state[f"column_choice_{dataset_choice}"] = column_choice


def time_range_selection(base_obj=None) -> pd.DataFrame:
    obj = base_obj or st
    data = get_as().df
    # data = st.session_state["df_full"]
    dates = obj.date_input(
        "Graph Date Range",
        value=(st.session_state.plot_start_date, st.session_state.plot_end_date),
        min_value=data.index.min(),
        max_value=data.index.max(),
        on_change=set_session_state_items,
        args=(["use_date_picker", "date_shift_buttons_used"], [True, False]),
    )
    if not st.session_state.date_shift_buttons_used:
        if len(dates) == 2:
            start_date, end_date = dates
            st.session_state.plot_start_date = start_date
            st.session_state.plot_end_date = end_date
        else:
            start_date = dates[0]
            end_date = st.session_state.plot_end_date
    else:
        start_date = st.session_state.plot_start_date
        end_date = st.session_state.plot_end_date
    inp_col_1, inp_col_2 = obj.columns(2)

    start_time = inp_col_1.time_input("Start time", value=datetime.time(0, 0, 0))
    end_time = inp_col_2.time_input("End time", value=datetime.time(23, 59, 59))

    state = get_as()

    if st.session_state.use_date_picker:
        inp_col_1.button(
            "Show All", on_click=set_session_state_items, args=("use_date_picker", False)
        )
        c1, c2, c3, c4 = obj.columns(4)
        c1.button("- 1d", on_click=shift_plot_window, args=(-1,))
        c2.button("+ 1d", on_click=shift_plot_window, args=(1,))
        c3.button("- 1w", on_click=shift_plot_window, args=(-7,))
        c4.button("+ 1w", on_click=shift_plot_window, args=(7,))
        start_dt = datetime.datetime.combine(start_date, start_time)
        end_dt = datetime.datetime.combine(end_date, end_time)
    else:
        start_dt = state.df.index.min()
        end_dt = state.df.index.max()
        obj.button(
            "Back to previous selection",
            on_click=set_session_state_items,
            args=("use_date_picker", True),
        )

    state.update_plot(start_dt, end_dt)


def shift_plot_window(days: int):
    current_start = st.session_state.plot_start_date
    current_end = st.session_state.plot_end_date

    current_range = current_end - current_start
    df = get_as().df

    new_start = current_start + datetime.timedelta(days)
    new_end = current_end + datetime.timedelta(days)

    if new_start < df.index.min():
        new_start = df.index.min()
        new_end = new_start + current_range
    if new_end > df.index.max():
        new_end = df.index.max()
        new_start = new_end - current_range

    set_session_state_items("plot_start_date", new_start)
    set_session_state_items("plot_end_date", new_end)
    set_session_state_items("date_shift_buttons_used", True)


def create_annotation_plot_buttons(base_obj=None):
    obj = base_obj or st

    state = get_as()

    obj.subheader("Actions")

    custom_text("Training Data", 15, True, base_obj=obj)
    c_1, c_2 = obj.columns(2)
    c_1.button("Mark selection Outlier", on_click=state.update_data, args=("outlier",))
    c_2.button("Mark selection Normal", on_click=state.update_data, args=("normal",))
    custom_text("Test Data", 15, True, base_obj=obj)
    c_1, c_2 = obj.columns(2)
    c_1.button(
        "Mark selection Outlier",
        on_click=state.update_data,
        args=("test_outlier",),
        key="mark_test_outlier",
    )
    c_2.button(
        "Mark selection Normal",
        on_click=state.update_data,
        args=("test_normal",),
        key="mark_test_normal",
    )
    obj.markdown("***")
    c_1, c_2 = obj.columns(2)

    c_1.button("Clear Selection", on_click=state.clear_selection)
    c_2.button("Clear All", on_click=state.clear_all)

    obj.markdown("***")


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
                    .isin(state[column].df[column].values)
                    .all()
                    # df["Water Level"].astype(float).isin(state.df["Water Level"].values).all()
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

            st.session_state.uploaded_annotation_data[uploaded_file.name] = data

    st.session_state["uploaded_annotation_data"] = data_to_load_if_file_ok

    # if not file_failed:

    # if file_failed:

    # else:
    #     st.session_state.uploaded_annotation_data[uploaded_file.name] = data
    # obj.success(f"{uploaded_file.name}: Loaded.", icon="✅")


def annotation_file_upload_callback(base_obj=None):
    obj = base_obj or st
    validate_uploaded_file_contents(obj)
    # state = get_as()

    for column, data_list in st.session_state.uploaded_annotation_data.items():
        state = get_as(column=column)
        for data in data_list:
            for key, df in data.items():
                state.update_data(key, df.index.to_list())


def dev_options(base_obj=None):
    obj = base_obj or st
    with obj.expander("Dev Options"):
        dev_col_1, dev_col_2 = st.columns(2)
        profile = dev_col_1.checkbox("Profile Code", value=False)
        search_str = dev_col_1.text_input("Search SS", max_chars=25, value="")
        show_ss = dev_col_2.button("Show Session State")
        show_as = dev_col_2.button("Show Annotation State")
    if len(search_str) > 1:
        matches = [k for k in st.session_state.keys() if search_str.lower() in k.lower()]
        df_matches = [m for m in matches if isinstance(st.session_state[m], pd.DataFrame)]
        non_df_matches = [m for m in matches if m not in df_matches]
        for m in df_matches:
            st.write(m)
            st.table(st.session_state[m].head())
        if non_df_matches:
            st.write({k: st.session_state[k] for k in non_df_matches})

        # st.write({k: st.session_state[k] for k in ss_options if search_str.lower() in k.lower()})
    if show_ss:
        st.write(st.session_state)
    if show_as:
        st.write(st.session_state["AS"])
        # st.write(st.session_state["AS"]["ddd"]["Water Level"].df)

    return Profiler() if profile else nullcontext()


def create_save_load_buttons(base_obj=None):
    obj = base_obj or st
    obj.subheader("Save / load previous")
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


def show_annotation_summary(base_obj=None):
    obj = base_obj or st
    with obj.expander("Annotation Info", expanded=True):
        state = get_as()
        obj.subheader(f"Annotation summary - {state.dataset} - {state.column}")
        c1, c2, c3 = st.columns([1, 1, 2])

        custom_text("Training Data", 15, base_obj=c1, centered=False)
        custom_text("Test Data", 15, base_obj=c2, centered=False)

        annotation_keys = ["outlier", "normal", "test_outlier", "test_normal"]
        if "old_number_annotations" in st.session_state:
            deltas = {
                k: len(state.data[k]) - st.session_state["old_number_annotations"][k]
                for k in annotation_keys
            }
        else:
            deltas = {k: None for k in annotation_keys}

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
            help="""Test data will not be used for training. It can be used to measure how well
            a model performs on new data not seen during training. """,
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
        obj.subheader(f"Current model: {st.session_state.last_model_name}")


def train_options(base_obj=None):
    obj = base_obj or st

    with st.sidebar.expander("Modelling Options", expanded=True):
        st.info("Here you can choose what type of outlier detection approach to use.")
        method = st.selectbox(
            "Choose OD method",
            options=list(MODEL_OPTIONS.keys()),
            key="current_method_choice",
            format_func=lambda x: MODEL_OPTIONS.get(x),
        )

    with st.sidebar.expander("Feature Options", expanded=True):
        st.info(
            "Here you can choose how many points before and after each annotated point \
            to include in its feature set."
        )
        st.number_input(
            "Points before",
            min_value=1,
            value=st.session_state.get("old_points_before")
            if st.session_state.get("old_points_before") is not None
            else 1,
            key="number_points_before",
            step=5,
        )
        st.number_input(
            "Points after",
            min_value=0,
            value=st.session_state.get("old_points_after")
            if st.session_state.get("old_points_after") is not None
            else 0,
            key="number_points_after",
            step=5,
        )

    state = get_as()
    if len(state.data["outlier"]) and len(state.data["normal"]) > 1:
        train_button = st.sidebar.button("Train Outlier Model", key="train_button")
    else:
        train_button = False

    if train_button:
        st.session_state["old_points_before"] = st.session_state["number_points_before"]
        st.session_state["old_points_after"] = st.session_state["number_points_after"]

        if recursive_length_count(state.data, exclude_keys="selected"):
            st.session_state["old_number_annotations"] = {k: len(v) for k, v in state.data.items()}

        with st.spinner("Constructing features..."):
            if method == "RF_1":
                construct_training_data_RF()
                construct_test_data_RF()
        with st.spinner("Training Model..."):
            if method == "RF_1":
                train_random_forest_classifier(obj)

    if "last_model_name" in st.session_state:
        st.sidebar.success(f"{st.session_state.last_model_name} finished training.")
        st.sidebar.download_button(
            "Download model",
            pickle.dumps(st.session_state["model_library"][st.session_state["last_model_name"]]),
            f"{st.session_state.last_model_name}.pkl",
        )


def get_predictions_callback(obj=None):
    # set_session_state_items("hide_choice_menus", True)
    set_session_state_items("page_index", 2)
    get_model_predictions(obj)
    set_session_state_items("prediction_models", {})


def add_annotation_to_pred_data(base_obj=None):
    obj = base_obj or st

    if not st.session_state.annotation_data_loaded:
        obj.error("No annotation data has been loaded in this session.")
        return

    if "Annotation Data" not in st.session_state.prediction_data:
        st.session_state.prediction_data["Annotation Data"] = get_as().df.copy(deep=True)


# def add_most_recent_model(base_obj=None):
#     obj = base_obj or st
#     if not st.session_state.last_model_name:
#         obj.error("No model has been trained yet this session.")
#         return
#     model_name = st.session_state.last_model_name
#     st.session_state.prediction_models[model_name] = st.session_state.classifier


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


def add_uploaded_dataset(base_obj=None):
    obj = base_obj or st

    if not st.session_state["uploaded_datasets"]:
        return

    for data in st.session_state.uploaded_datasets:
        try:
            df = pd.read_csv("data/Elev_NW1.csv", index_col=0, parse_dates=True)
            st.session_state["prediction_data"][data.name] = df
        except Exception:
            obj.error(f"Could not read file {data.name}.")


def add_session_models():
    to_add = st.session_state["session_models_to_add"]
    uploaded_models = {
        k: v for k, v in st.session_state["prediction_models"].items() if k.endswith(".pkl")
    }
    st.session_state["prediction_models"] = {
        k: st.session_state["model_library"][k] for k in to_add
    }
    st.session_state["prediction_models"].update(uploaded_models)

    # for model_name in to_add:
    #     if model_name in st.session_state["prediction_models"]:
    #         continue

    #     st.session_state["prediction_models"][model_name] = st.session_state["model_library"][
    #         model_name
    #     ]


def add_session_dataset():
    session_ds = st.session_state["pred_session_ds_choice"]
    session_cols = st.session_state["pred_session_col_choice"]

    st.session_state["prediction_data"][session_ds] = {}
    for col in session_cols:
        st.session_state["prediction_data"][session_ds][col] = st.session_state["data_store"][
            session_ds
        ][col]


def prediction_options(base_obj=None):
    obj = base_obj or st
    _, c, _ = obj.columns([2, 5, 2])
    c.button("Generate Predictions", on_click=get_predictions_callback, args=(obj,))

    with obj.expander("Model Choice", expanded=True):
        # with obj.expander("Model Choice", expanded=not st.session_state["hide_choice_menus"]):
        st.subheader("Choose Models")
        st.info(
            "Add models with which to generate predictions. The most recently trained model is automatically added."
        )
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
        )
        # c1.button("Add most recently trained model", on_click=add_most_recent_model, args=(obj,))
        st.file_uploader(
            "Select model from disk",
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
        # with obj.expander("Data Choice", expanded=not st.session_state["hide_choice_menus"]):
        st.subheader("Select Data")
        st.info("Add datasets for outlier evaluation.")
        # c1, c2 = st.columns(2)
        ds_options = list(st.session_state["data_store"].keys())
        if "last_model_name" in st.session_state:
            idx = ds_options.index(
                st.session_state["model_library"][st.session_state["last_model_name"]][
                    "trained_on_dataset"
                ]
            )
        else:
            if len(ds_options):
                idx = len(ds_options) - 1
            else:
                idx = None
        ds_choice = st.selectbox(
            label="Select datasets created this session",
            options=ds_options,
            index=idx,
            disabled=len(st.session_state["data_store"]) < 2,
            key="pred_session_ds_choice",
        )
        if ds_choice:
            col_options = list(st.session_state["data_store"][ds_choice].columns)
            if "last_model_name" in st.session_state:
                if (
                    ds_choice
                    == st.session_state["model_library"][st.session_state["last_model_name"]][
                        "trained_on_dataset"
                    ]
                ):
                    default = st.session_state["model_library"][
                        st.session_state["last_model_name"]
                    ]["trained_on_series"]
                else:
                    default = None
            else:
                default = None

            session_ds_columns = st.multiselect(
                "Pick columns",
                options=col_options,
                default=default,
                on_change=add_session_dataset,
                key="pred_session_col_choice",
            )
        # c1.button("Add Annotation Data", on_click=add_annotation_to_pred_data)
        # add_annotation_to_pred_data()
        st.file_uploader(
            "Select dataset from disk",
            accept_multiple_files=True,
            key="uploaded_datasets",
            on_change=add_uploaded_dataset,
        )
        st.subheader("Selected Series:")
        st.json({k: list(v.keys()) for k, v in st.session_state["prediction_data"].items()})
        if st.session_state.prediction_data:
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


def remove_model_to_visualize(dataset_name, model_name):
    st.session_state["models_to_visualize"][dataset_name].discard(model_name)

    # if not recursive_length_count(st.session_state["models_to_visualize"]):
    # st.session_state["hide_choice_menus"] = False


def prediction_summary_table(dataset_name: str, base_obj=None):
    obj = base_obj or st

    DEFAULT_MARKER_COLORS = ["#e88b0b", "#1778dc", "#1bd476", "#d311e6"]
    model_predictions: pd.DataFrame() = st.session_state["inference_results"].get(dataset_name)
    if model_predictions is None:
        return
    model_names = sorted(st.session_state["models_to_visualize"][dataset_name])

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
            key=f"remove_{model}_{dataset_name}",
            on_click=remove_model_to_visualize,
            args=(dataset_name, model),
        )
        c4.json(st.session_state["model_library"][model]["params"], expanded=False)
        custom_text(
            st.session_state["number_outliers"][dataset_name][model],
            base_obj=c5,
            font_size=20,
            centered=False,
        )
        custom_text(
            len(st.session_state["inference_results"][dataset_name])
            - st.session_state["number_outliers"][dataset_name][model],
            base_obj=c6,
            font_size=20,
            centered=False,
        )
        c7.color_picker(
            model,
            key=f"color_{model}_{dataset_name}",
            label_visibility="collapsed",
            value=DEFAULT_MARKER_COLORS[i],
        )
        c7.write(st.session_state[f"color_{model}_{dataset_name}"])
        obj.markdown("***")


def test_metrics(base_obj=None):
    if "test_features" not in st.session_state:
        return

    obj = base_obj or st
    custom_text(f"Most recent model: {st.session_state['last_model_name']}", base_obj=obj)
    c1, c2, c3, c4 = obj.columns(4)

    current_train_metrics = st.session_state["current_model_train_metrics"]
    prec = current_train_metrics["precision"]
    rec = current_train_metrics["recall"]
    f1 = current_train_metrics["f1"]

    if "previous_model_train_metrics" in st.session_state:
        old_metrics = st.session_state["previous_model_train_metrics"]
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
            help="The ratio of true predicted positives to total predicted positives for the train set outliers. Represents the ability \
                not to classify a normal sample as an outlier.",
        )
        st.metric(
            "Recall Score",
            rec[1],
            delta=out_rec_diff,
            delta_color="normal" if out_rec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total positives for the train set outliers. Represents the ability \
                to correctly predict all the outliers.",
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
            help="The ratio of true predicted positives to total predicted positives for the train set normal points. Represents the ability \
                not to classify an outlier sample as a normal point.",
        )
        st.metric(
            "Recall Score",
            rec[0],
            delta=norm_rec_diff,
            delta_color="normal" if norm_rec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total positives for the train set normal points. Represents the ability \
                to correctly predict all the normal points.",
        )
        st.metric(
            "F1 Score",
            f1[0],
            delta=norm_f1_diff,
            delta_color="normal" if norm_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the train set normal points.",
        )

    current_metrics = st.session_state["current_model_test_metrics"]
    prec = current_metrics["precision"]
    rec = current_metrics["recall"]
    f1 = current_metrics["f1"]

    if "previous_model_test_metrics" in st.session_state:
        old_metrics = st.session_state["previous_model_test_metrics"]
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

    with c3.expander("Test Set Outlier Metrics", expanded=True):
        st.metric(
            "Precision Score",
            prec[1],
            delta=out_prec_diff,
            delta_color="normal" if out_prec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total predicted positives for the test set outliers. Represents the ability \
                not to classify a normal sample as an outlier.",
        )
        st.metric(
            "Recall Score",
            rec[1],
            delta=out_rec_diff,
            delta_color="normal" if out_rec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total positives for the test set outliers. Represents the ability \
                to correctly predict all the outliers.",
        )
        st.metric(
            "F1 Score",
            f1[1],
            delta=out_f1_diff,
            delta_color="normal" if out_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the outliers.",
        )
    with c4.expander("Test Set Normal Metrics", expanded=True):
        st.metric(
            "Precision Score",
            prec[0],
            delta=norm_prec_diff,
            delta_color="normal" if norm_prec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total predicted positives for the test set normal points. Represents the ability \
                not to classify an outlier sample as a normal point.",
        )
        st.metric(
            "Recall Score",
            rec[0],
            delta=norm_rec_diff,
            delta_color="normal" if norm_rec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total positives for the test set normal points. Represents the ability \
                to correctly predict all the normal points.",
        )
        st.metric(
            "F1 Score",
            f1[0],
            delta=norm_f1_diff,
            delta_color="normal" if norm_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the normal points.",
        )


def model_choice_callback(dataset_name: str):
    st.session_state["models_to_visualize"][dataset_name] = set(
        st.session_state[f"model_choice_{dataset_name}"]
    )


def model_choice_options(dataset_name: str):
    if st.session_state["inference_results"].get(dataset_name) is None:
        return
    st.info(
        f"Here you can choose which of the model predictions that exist for the datset \
            '{dataset_name}' so far to visualize."
    )
    st.multiselect(
        "Choose models for dataset",
        sorted(st.session_state["available_models"][dataset_name]),
        key=f"model_choice_{dataset_name}",
        default=sorted(st.session_state["models_to_visualize"][dataset_name]),
        on_change=model_choice_callback,
        args=(dataset_name,),
        max_selections=4,
    )

    st.markdown("***")


def outlier_visualization_options(dataset_name: str):
    # if st.session_state["inference_results"].get(dataset_name) is None:
    # return

    if not st.session_state["models_to_visualize"][dataset_name]:
        return

    form = st.form(
        f"form_{dataset_name}",
    )
    form.info(
        "Click on a bar in the distribution plot to view all outliers \
        in that time period. Each time period is chosen so it contains the same number of datapoints. That number can be adjusted here."
    )
    form.slider(
        "Number of datapoints per bar",
        value=300,
        min_value=10,
        max_value=1000,
        step=1,
        key=f"num_outliers_{dataset_name}",
    )
    form.slider(
        "Height of figures (px)",
        value=600,
        min_value=100,
        max_value=1500,
        step=100,
        key=f"figure_height_{dataset_name}",
    )

    form.checkbox(
        "Only show time ranges containing outliers (predicted or annotated)",
        key=f"only_show_ranges_with_outliers_{dataset_name}",
    )

    form.form_submit_button("Update Distribution Plot")


def show_feature_importances(base_obj=None):
    obj = base_obj or st
    if "last_model_name" not in st.session_state:
        return

    with obj.expander("Feature Importances", expanded=True):
        c1, c2 = st.columns([2, 1])
        feature_importance_plot(c1)
        c2.dataframe(st.session_state["current_importances"])


def add_slider_selected_points(dataset_name: str, model_name: str):
    start, end = st.session_state[f"outlier_slider_{dataset_name}_{model_name}"]
    coords = st.session_state[f"current_outlier_value_store"][dataset_name][model_name]
    timestamps_to_add = {coords[i][0] for i in range(start, end + 1)}
    state = get_as()
    state.update_selected(timestamps_to_add)


def process_data_from_echarts_plot(
    clicked_point: List | dict | None, dataset_name=None, base_obj=None
):
    obj = base_obj or st
    state = get_as()
    was_updated = False

    if (clicked_point is None) or ((clicked_point[1] == "brush") and (not clicked_point[0])):
        return

    # we want to select only the outlier series, not the datapoints series.
    # This behaviour is set by a checkbox above the plot. It only effects area selection.
    if (
        (clicked_point[1] == "brush")
        and dataset_name
        and st.session_state[f"only_select_outliers_{dataset_name}"]
    ):
        model_names = sorted(st.session_state["models_to_visualize"][dataset_name])

        state = get_as()
        relevant_outlier_idc = {
            d["seriesName"]: d["dataIndex"]
            for d in clicked_point[0]
            if d["seriesName"] in model_names
        }
        relevant_data_points = [
            st.session_state["pred_outlier_tracker"][k].iloc[v].index.to_list()
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


def correction_options(dataset_name: str, base_obj=None):
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

    state = get_as()

    c1.button(
        "Mark Train Outlier",
        on_click=state.update_data,
        args=("outlier",),
        kwargs={"base_obj": obj},
        key=f"pred_mark_outlier_{dataset_name}",
    )
    c2.button(
        "Mark Train Normal",
        on_click=state.update_data,
        args=("normal",),
        kwargs={"base_obj": obj},
        key=f"pred_mark_normal_{dataset_name}",
    )
    c3.button(
        "Mark Test Outlier",
        on_click=state.update_data,
        args=("test_outlier",),
        kwargs={"base_obj": obj},
        key=f"pred_mark_test_outlier_{dataset_name}",
    )
    c4.button(
        "Mark Test Normal",
        on_click=state.update_data,
        args=("test_normal",),
        kwargs={"base_obj": obj},
        key=f"pred_mark_test_normal_{dataset_name}",
    )
    c1.button(
        "Clear Selection",
        on_click=state.clear_selection,
        key=f"pred_clear_selection_{dataset_name}",
    )
    # c3.button("Clear All", on_click=state.clear_all)
