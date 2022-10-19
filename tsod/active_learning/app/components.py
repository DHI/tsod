from ast import arg
import datetime
import pickle

import pandas as pd
import streamlit as st
from tsod.active_learning.modelling import (
    construct_training_data,
    post_training_options,
    train_random_forest_classifier,
)
from tsod.active_learning.utils import (
    get_as,
    set_session_state_items,
    custom_text,
    recursive_length_count,
)
from tsod.active_learning.modelling import get_model_predictions


def filter_data(base_obj=None) -> pd.DataFrame:
    obj = base_obj or st
    obj.subheader("Time range selection")

    data = st.session_state["df_full"]
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

    obj.markdown("***")

    state.update_plot(start_dt, end_dt)


def shift_plot_window(days: int):
    current_start = st.session_state.plot_start_date
    current_end = st.session_state.plot_end_date

    current_range = current_end - current_start
    df = st.session_state.df_full

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


def create_plot_buttons(base_obj=None):
    obj = base_obj or st

    state = get_as()

    obj.subheader("Actions")

    custom_text("Training Data", 15, True, base_obj=obj)
    c_1, c_2 = obj.columns(2)
    c_1.button("Mark selection Outlier", on_click=state.update_data, args=("outlier",))
    c_2.button("Mark selection not Outlier", on_click=state.update_data, args=("normal",))
    custom_text("Test Data", 15, True, base_obj=obj)
    c_1, c_2 = obj.columns(2)
    c_1.button(
        "Mark selection Outlier",
        on_click=state.update_data,
        args=("test_outlier",),
        key="mark_test_outlier",
    )
    c_2.button(
        "Mark selection not Outlier",
        on_click=state.update_data,
        args=("test_normal",),
        key="mark_test_normal",
    )
    c_1.button("Clear Selection", on_click=state.clear_selection)
    c_2.button("Clear All", on_click=state.clear_all)

    obj.markdown("***")


def validate_uploaded_file_contents(base_obj=None):
    obj = base_obj or st
    uploaded_files = st.session_state["uploaded_annotation_files"]
    if not uploaded_files:
        return
    for uploaded_file in uploaded_files:
        file_failed = False
        data = pickle.loads(uploaded_file.getvalue())
        state = get_as()
        for df in data.values():
            if df.empty or file_failed:
                continue

            values_match = (
                df["Water Level"].astype(float).isin(state.df["Water Level"].values).all()
            )
            index_match = df.index.isin(state.df.index).all()

            if (not values_match) or (not index_match):
                file_failed = True
        if file_failed:
            obj.error(f"{uploaded_file.name}: Did not pass validation for loaded dataset.")
        else:
            st.session_state.uploaded_annotation_data[uploaded_file.name] = data
            obj.success(f"{uploaded_file.name}: Loaded.", icon="✅")


def annotation_file_upload_callback(base_obj=None):
    obj = base_obj or st
    validate_uploaded_file_contents(obj)
    # if file_val_results:
    # obj.info("Validation passed")
    # elif file_val_results is None:
    # return
    # else:
    # obj.error("The loaded annotation data points do not all match the loaded data.")
    # return

    state = get_as()
    # file_name = st.session_state.current_uploaded_file.name
    # data = st.session_state.uploaded_annotation_data[file_name]

    for data in st.session_state.uploaded_annotation_data.values():
        for key, df in data.items():
            state.update_data(key, df.index.to_list())

    # state.update_data("outlier", data["outliers"].index.to_list())
    # state.update_data("normal", data["normal"].index.to_list())


def dev_options(base_obj=None):
    obj = base_obj or st
    with obj.expander("Dev Options"):
        dev_col_1, dev_col_2 = st.columns(2)
        profile = dev_col_1.checkbox("Profile Code", value=False)
        show_ss = dev_col_2.button("Show Session State")
        show_as = dev_col_2.button("Show Annotation State")
    if show_ss:
        st.write(st.session_state)
    if show_as:
        st.write(get_as().__dict__)

    return profile


def create_save_load_buttons(base_obj=None):
    obj = base_obj or st
    obj.subheader("Save / load previous")
    c_1, c_2 = obj.columns(2)
    state = get_as()

    file_name = f"Annotations_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.bin"

    if state._download_data:
        c_1.download_button("Save annotations to disk", state.download_data, file_name=file_name)

    c_2.file_uploader(
        "Load annotations from disk",
        key="uploaded_annotation_files",
        type="bin",
        on_change=annotation_file_upload_callback,
        args=(obj,),
        accept_multiple_files=True,
    )
    obj.markdown("***")


def show_info(base_obj=None):
    obj = base_obj or st
    state = get_as()
    obj.subheader("Annotation summary")
    c1, c2, c3 = obj.columns([1, 1, 2])

    custom_text("Training Data", 15, base_obj=c1)
    custom_text("Test Data", 15, base_obj=c2)

    c1.metric("Total number of labelled outliers", len(state.outlier))
    c1.metric("Total number of labelled normal points", len(state.normal))
    c2.metric("Total number of labelled outliers", len(state.test_outlier))
    c2.metric("Total number of labelled normal points", len(state.test_normal))


def train_options(base_obj=None):
    obj = base_obj or st

    train_button = obj.button("Train Random Forest Model")

    if train_button:
        with st.spinner("Constructing features..."):
            construct_training_data()
        with st.spinner("Training Model..."):
            train_random_forest_classifier(obj)
        post_training_options(obj)


def get_predictions_callback(obj=None):
    set_session_state_items("hide_choice_menus", True)
    get_model_predictions(base_obj=obj)


def add_annotation_to_pred_data(base_obj=None):
    obj = base_obj or st

    if not st.session_state.annotation_data_loaded:
        obj.error("No annotation data has been loaded in this session.")
        return

    st.session_state.prediction_data["Annotation Data"] = get_as().df.copy(deep=True)


def add_most_recent_model(base_obj=None):
    obj = base_obj or st
    if not st.session_state.last_model_name:
        obj.error("No model has been trained yet this session.")
        return
    model_name = st.session_state.last_model_name
    st.session_state.prediction_models[model_name] = st.session_state.classifier


def add_uploaded_model(base_obj=None):
    obj = base_obj or st
    if not st.session_state.current_uploaded_model:
        return
    for model in st.session_state.current_uploaded_model:
        clf = pickle.loads(model.read())
        if not hasattr(clf, "predict"):
            obj.error(
                "The uploaded object can not be used for prediction (does not implement 'predict' method)."
            )
            continue

        st.session_state.prediction_models[model.name] = clf


def prediction_options(base_obj=None):
    obj = base_obj or st
    _, c, _ = obj.columns([2, 5, 2])
    c.button("Add Predictions", on_click=get_predictions_callback, args=(obj,))

    with obj.expander("Model Choice", expanded=not st.session_state["hide_choice_menus"]):
        st.subheader("Choose Models")
        st.info("Add models with which to generate predictions.")
        c1, c2 = st.columns(2)
        c1.button("Add most recently trained model", on_click=add_most_recent_model, args=(obj,))
        c2.file_uploader(
            "Select model from disk",
            type="pkl",
            on_change=add_uploaded_model,
            key="current_uploaded_model",
            args=(obj,),
            accept_multiple_files=True,
        )

        st.subheader("Selected models:")
        st.json(list(st.session_state.prediction_models.keys()))
        if st.session_state.prediction_models:
            st.button(
                "Clear selection", on_click=set_session_state_items, args=("prediction_models", {})
            )
    with obj.expander("Data Choice", expanded=not st.session_state["hide_choice_menus"]):
        st.subheader("Select Data")
        st.info("Add datasets for outlier evaluation.")
        c1, c2 = st.columns(2)
        c1.button("Add Annotation Data", on_click=add_annotation_to_pred_data)
        add_annotation_to_pred_data()
        st.subheader("Selected Files:")
        st.json(list(st.session_state.prediction_data.keys()))
        if st.session_state.prediction_data:
            st.button(
                "Clear selection",
                on_click=set_session_state_items,
                args=("prediction_data", {}),
                key="data_clear",
            )
    _, c, _ = obj.columns([2, 5, 2])
    c.button("Add Predictions", on_click=get_predictions_callback, args=(obj,), key="pred_btn_2")


def remove_model_to_visualize(dataset_name, model_name):
    st.session_state["models_to_visualize"][dataset_name].discard(model_name)

    if not recursive_length_count(st.session_state["models_to_visualize"]):
        st.session_state["hide_choice_menus"] = False


def prediction_summary_table(dataset_name: str, base_obj=None):
    obj = base_obj or st
    obj.subheader(dataset_name)

    DEFAULT_COLORS = ["#f11a1a", "#2ada49", "#1e11e6", "#40e0d3"]

    model_predictions = st.session_state["inference_results"].get(dataset_name)
    if not model_predictions:
        return

    model_names = st.session_state["models_to_visualize"][dataset_name]

    if not model_names:
        return

    if len(model_names) > len(DEFAULT_COLORS):
        obj.error(
            f"Currently max. number of models is {len(DEFAULT_COLORS)}, got {len(model_names)}"
        )
        return

    c1, c2, c3, c4, c5 = obj.columns([5, 2, 4, 4, 3])
    custom_text("Model Name", base_obj=c1, font_size=20, centered=True)
    custom_text("Predicted Outliers", base_obj=c3, font_size=20, centered=False)
    custom_text("Predicted Normal", base_obj=c4, font_size=20, centered=False)
    custom_text("Choose Plot Color", base_obj=c5, font_size=20, centered=False)

    obj.markdown("***")

    for i, model in enumerate(model_names):
        _local_obj = obj.container()
        c1, c2, c3, c4, c5 = _local_obj.columns([5, 2, 4, 4, 3])
        custom_text(model, base_obj=c1, font_size=15, centered=True)
        c2.button(
            "Remove",
            key=f"remove_{model}",
            on_click=remove_model_to_visualize,
            args=(dataset_name, model),
        )
        custom_text(
            st.session_state["number_outliers"][dataset_name][model],
            base_obj=c3,
            font_size=20,
            centered=False,
        )
        custom_text(
            len(st.session_state["prediction_data"][dataset_name])
            - st.session_state["number_outliers"][dataset_name][model],
            base_obj=c4,
            font_size=20,
            centered=False,
        )
        c5.color_picker(
            model, key=f"color_{model}", label_visibility="collapsed", value=DEFAULT_COLORS[i]
        )
        obj.markdown("***")
