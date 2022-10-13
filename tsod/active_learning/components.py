import datetime
import pickle

import pandas as pd
import streamlit as st
from tsod.active_learning.modelling import (
    construct_training_data,
    post_training_options,
    train_random_forest_classifier,
)
from tsod.active_learning.utils import get_as


def filter_data(data: pd.DataFrame, base_obj=None) -> pd.DataFrame:
    obj = base_obj or st
    obj.subheader("Time range selection")

    dates = obj.date_input(
        "Graph range start date",
        value=(data.index.max().date() - datetime.timedelta(days=7), data.index.max().date()),
        min_value=data.index.min(),
        max_value=data.index.max(),
    )
    if len(dates) == 2:
        start_date, end_date = dates
        st.session_state.end_date = end_date
    else:
        start_date = dates[0]
        end_date = st.session_state.end_date
    inp_col_1, inp_col_2 = obj.columns(2)

    start_time = inp_col_1.time_input("Start time", value=datetime.time(0, 0, 0))
    end_time = inp_col_2.time_input("End time", value=datetime.time(23, 59, 59))

    obj.markdown("***")

    start_dt = datetime.datetime.combine(start_date, start_time)
    end_dt = datetime.datetime.combine(end_date, end_time)

    state = get_as()

    state.update_plot(start_dt, end_dt)


def create_plot_buttons(base_obj=None):
    obj = base_obj or st

    state = get_as()

    obj.subheader("Annotation actions")

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

    if validate_uploaded_file_contents():
        obj.info("Validation passed")
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
    info = {
        "Total number of labelled outliers": [len(state.outlier)],
        "Total number of labelled normal points": [len(state.normal)],
    }
    obj.table(info)


def train_options(base_obj=None):
    obj = base_obj or st

    train_button = obj.button("Train Random Forest Model")

    if train_button:
        with st.spinner("Constructing features..."):
            construct_training_data()
        with st.spinner("Training Model..."):
            train_random_forest_classifier()
        post_training_options(obj)
