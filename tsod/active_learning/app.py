from typing import List
from matplotlib.pyplot import show
import streamlit as st

from streamlit_echarts import st_echarts
from streamlit_echarts import st_pyecharts
import pandas as pd
import numpy as np
from pyecharts.charts import Line
import plotly.express as px
import pickle
import datetime

from pyecharts import options as opts
from streamlit_plotly_events import plotly_events
import plotly.graph_objs as go

from tsod.active_learning.modelling import (
    construct_training_data,
    train_random_forest_classifier,
    show_post_training_info,
)
from streamlit_profiler import Profiler
from contextlib import nullcontext

from tsod.active_learning.data_structures import AnnotationState


SELECT_OPTIONS = {"Point": "zoom", "Box": "select", "Lasso": "lasso"}
SELECT_INFO = {
    "Point": "Select individual points, drag mouse to zoom in.",
    "Box": "Draw Box to select all points in range.",
    "Lasso": "Draw Lasso to select all points in range.",
}


def get_as() -> AnnotationState:
    return st.session_state.AS


@st.cache(allow_output_mutation=True)
def load_data(file_name: str = "TODO"):
    df = pd.read_csv("data/Elev_NW1.csv", index_col=0, parse_dates=True)
    df["Water Level"] = df["Water Level"].astype(np.float16)
    return df


def prepare_session_state():
    if "AS" not in st.session_state:
        st.session_state.AS = AnnotationState(st.session_state["df_full"])

    if "uploaded_annotation_data" not in st.session_state:
        st.session_state.uploaded_annotation_data = {}

    if "current_files_in_AS" not in st.session_state:
        st.session_state.current_files_in_AS = set()


@st.experimental_memo(persist="disk")
def create_cachable_line_plot(
    start_time, end_time, data_file_identifier: str = "TODO"
) -> go.Figure:
    plot_data = get_as().df_plot
    timestamps = plot_data.index.to_list()

    return px.line(
        plot_data,
        x=timestamps,
        y="Water Level",
        markers=True,
    )


def create_annotation_plot(base_obj=None) -> go.Figure:
    obj = base_obj or st
    obj.subheader("Data selection options")
    selection_method = obj.selectbox("Data Selection Method", list(SELECT_OPTIONS.keys()))
    obj.info(SELECT_INFO[selection_method])
    obj.markdown("***")
    state = get_as()

    fig = create_cachable_line_plot(state.start, state.end)

    df_plot = state.df_plot

    df_selected = df_plot[df_plot.index.isin(state.selection)]
    df_marked_out = df_plot[df_plot.index.isin(state.outlier)]
    df_marked_not_out = df_plot[df_plot.index.isin(state.normal)]

    if not df_selected.empty:
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_selected.index,
                y=df_selected["Water Level"],
                name=f"Selected ({len(df_selected)})",
                marker=dict(color="Purple", size=8, line=dict(color="Black", width=1)),
            )
        )

    if not df_marked_out.empty:
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_marked_out.index,
                y=df_marked_out["Water Level"],
                name=f"Marked Outlier ({len(df_marked_out)})",
                marker=dict(color="Red", size=12, line=dict(color="Black", width=1)),
            )
        )
    if not df_marked_not_out.empty:
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_marked_not_out.index,
                y=df_marked_not_out["Water Level"],
                name=f"Marked Not Outlier ({len(df_marked_not_out)})",
                marker=dict(color="Green", size=12, line=dict(color="Black", width=1)),
            )
        )

    fig.update_layout(dragmode=SELECT_OPTIONS[selection_method])
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True, autorange=True),
            type="date",
        )
    )

    return fig


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


@st.cache(persist=True)
def prepare_download(outliers: pd.DataFrame, normal: pd.DataFrame):
    return pickle.dumps(
        {
            "outliers": outliers,
            "normal": normal,
        }
    )


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
        obj.success("Finished training.")
        show_post_training_info(obj)


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Controls")

    #############################################################
    st.sidebar.subheader("Development")
    dev_col_1, dev_col_2 = st.sidebar.columns(2)
    profile = dev_col_1.checkbox("Profile Code", value=False)
    show_ss = dev_col_2.button("Show Session State")
    show_as = dev_col_2.button("Show Annotation State")
    if show_ss:
        st.write(st.session_state)
    if show_as:
        st.write(get_as().data)
    #############################################################

    with Profiler() if profile else nullcontext():

        data = load_data()

        if "df_full" not in st.session_state:
            st.session_state["df_full"] = data
        prepare_session_state()
        state = get_as()

        tab1, tab2, tab3 = st.tabs(["Outlier annotation", "Model training", "Model prediction"])
        create_plot_buttons(st.sidebar)
        filter_data(data, st.sidebar)
        plot_time_ph = st.sidebar.container()

        create_save_load_buttons(st.sidebar)

        fig = create_annotation_plot(plot_time_ph)

        with tab1:
            selection = plotly_events(fig, select_event=True, override_height=1000)
        selection = {e["x"] for e in selection}
        state.update_selected(selection)

        # st.write(st.session_state)
        # tab1.write(st.session_state.AS.data)

        with tab2:
            col_1, col_2 = st.columns([1, 3])
            show_info(col_1)
            train_options(col_1)

    # st.write(st.session_state.selected_points)

    ############ Echarts ################

    # line = Line()

    # xaxis_opts = opts.AxisOpts(type_="time", is_scale=True, name="Date & Time")

    # yaxis_opts = opts.AxisOpts(type_="value", name="Water Level")

    # SCROLL_ZOOM = opts.DataZoomOpts(
    #     type_="inside",
    #     range_start=0,
    #     range_end=100,
    # )

    # SLIDER_ZOOM = opts.DataZoomOpts(
    #     type_="slider",
    #     range_start=0,
    #     range_end=100,
    # )

    # tooltip_opts = opts.TooltipOpts(axis_pointer_type="cross")

    # label_opts = opts.LabelOpts(is_show=False)

    # line.add_xaxis(timestamps).add_yaxis(
    #     "Water Level",
    #     water_levels,
    #     is_symbol_show=True,
    #     label_opts=label_opts,
    #     symbol="circle",
    # ).set_global_opts(
    #     xaxis_opts=xaxis_opts,
    #     yaxis_opts=yaxis_opts,
    #     datazoom_opts=[SLIDER_ZOOM],
    #     toolbox_opts=tooltip_opts,
    # )

    # test = st_pyecharts(
    #     line,
    #     height="600px",
    #     events={
    #         "globalout": "function(params) {return params}",
    #         # "globalout": "function(params) {console.log(params)}",
    #         # "click": "function(params) {console.log(params)}",
    #     },
    #     # events={
    #     # "mousedown": "function(params) {return params.data}",
    #     # "click": "function(params) {return params.data}",
    #     # },
    # )

    # st.write(test)

    # st_echarts(option)


if __name__ == "__main__":
    main()
