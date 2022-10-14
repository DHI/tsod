import datetime
import pickle

import pandas as pd
import streamlit as st
from tsod.active_learning.modelling import (
    construct_training_data,
    post_training_options,
    train_random_forest_classifier,
)
from tsod.active_learning.utils import get_as, set_session_state_items
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

    c_1, c_2 = obj.columns(2)
    c_1.button("Mark selection Outlier", on_click=state.update_outliers)
    c_2.button("Mark selection not Outlier", on_click=state.update_normal)
    c_1.button("Clear Selection", on_click=state.clear_selection)
    c_2.button("Clear All", on_click=state.clear_all)

    obj.markdown("***")


@st.cache(persist=True)
def prepare_download(outliers: pd.DataFrame, normal: pd.DataFrame):
    return pickle.dumps(
        {
            "outliers": outliers,
            "normal": normal,
        }
    )


def validate_uploaded_file_contents():
    uploaded_file = st.session_state["current_uploaded_file"]
    if not uploaded_file:
        return None
    data = pickle.loads(uploaded_file.getvalue())
    state = get_as()
    if data["outliers"].empty:
        outlier_data_match, outlier_index_match = True, True
    else:
        outlier_data_match = (
            data["outliers"]["Water Level"].astype(float).isin(state.df["Water Level"].values).all()
        )
        outlier_index_match = data["outliers"].index.isin(state.df.index).all()
    if data["normal"].empty:
        normal_data_match, normal_index_match = True, True
    else:
        normal_data_match = (
            data["normal"]["Water Level"].astype(float).isin(state.df["Water Level"].values).all()
        )
        normal_index_match = data["normal"].index.isin(state.df.index).all()

    if outlier_data_match and outlier_index_match and normal_data_match and normal_index_match:
        st.session_state.uploaded_annotation_data[uploaded_file.name] = data
        return True
    return False


def annotation_file_upload_callback(base_obj=None):
    obj = base_obj or st
    file_val_results = validate_uploaded_file_contents()
    if file_val_results:
        obj.info("Validation passed")
    elif file_val_results is None:
        return
    else:
        obj.error("The loaded annotation data points do not all match the loaded data.")
        return

    state = get_as()
    file_name = st.session_state.current_uploaded_file.name
    data = st.session_state.uploaded_annotation_data[file_name]

    state.update_outliers(data["outliers"].index.to_list())
    state.update_normal(data["normal"].index.to_list())
    obj.success("Loaded annotations", icon="✅")


def create_save_load_buttons(base_obj=None):
    obj = base_obj or st
    obj.subheader("Save / load previous")
    c_1, c_2 = obj.columns(2)
    state = get_as()

    to_save = prepare_download(state.df_outlier, state.df_normal)
    file_name = f"Annotations_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.bin"

    c_1.download_button("Save annotations to disk", to_save, file_name=file_name)

    c_2.file_uploader(
        "Load annotations from disk",
        key="current_uploaded_file",
        type="bin",
        on_change=annotation_file_upload_callback,
        args=(obj,),
    )
    obj.markdown("***")


def show_info(base_obj=None):
    obj = base_obj or st
    state = get_as()
    obj.subheader("Annotation summary")

    obj.metric("Total number of labelled outliers", len(state.outlier))
    obj.metric("Total number of labelled normal points", len(state.normal))


def train_options(base_obj=None):
    obj = base_obj or st

    train_button = obj.button("Train Random Forest Model")

    if train_button:
        with st.spinner("Constructing features..."):
            construct_training_data()
        with st.spinner("Training Model..."):
            train_random_forest_classifier(obj)
        post_training_options(obj)


def start_button_click_callback(obj=None):
    with st.spinner("Getting model results..."):
        get_model_predictions(base_obj=obj)


def add_annotation_to_pred_data(base_obj=None):
    obj = base_obj or st

    if not st.session_state.annotation_data_loaded:
        obj.error("No annotation data has been loaded in this session.")
        return

    st.session_state.prediction_data["Annotation Data"] = get_as().df


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
    clf = pickle.loads(st.session_state.current_uploaded_model.read())
    if not hasattr(clf, "predict"):
        obj.error(
            "The uploaded object can not be used for prediction (does not implement 'predict' method."
        )
        return

    st.session_state.prediction_models[st.session_state.current_uploaded_model.name] = clf


def prediction_options(base_obj=None):
    obj = base_obj or st
    _, c, _ = obj.columns([2, 5, 2])
    c.button("Get Predictions", on_click=start_button_click_callback, args=(obj,))

    obj.subheader("Choose Models")
    c1, c2 = obj.columns(2)
    c1.button("Add most recently trained model", on_click=add_most_recent_model, args=(obj,))
    c2.file_uploader(
        "Select model from disk",
        type="pkl",
        on_change=add_uploaded_model,
        key="current_uploaded_model",
        args=(obj,),
    )

    # obj.markdown("***")

    obj.subheader("Selected models:")
    obj.write(list(st.session_state.prediction_models.keys()))
    if st.session_state.prediction_models:
        obj.button(
            "Clear selection", on_click=set_session_state_items, args=("prediction_models", {})
        )
    obj.markdown("***")

    obj.subheader("Select Data")
    obj.info("Use 'Add Annotation Data' to add the same data that was loaded in for annotation.")
    c1, c2 = obj.columns(2)
    c1.button("Add Annotation Data", on_click=add_annotation_to_pred_data)
    obj.markdown("***")

    obj.subheader("Selected Files:")
    obj.write(list(st.session_state.prediction_data.keys()))
    if st.session_state.prediction_data:
        obj.button(
            "Clear selection",
            on_click=set_session_state_items,
            args=("prediction_data", {}),
            key="data_clear",
        )
    obj.markdown("***")
    _, c, _ = obj.columns([2, 5, 2])
    c.button("Get Predictions", on_click=start_button_click_callback, args=(obj,), key="pred_btn_2")
