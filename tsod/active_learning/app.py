from typing import List
import streamlit as st

from streamlit_echarts import st_echarts
from streamlit_echarts import st_pyecharts
import pandas as pd
import numpy as np
from pyecharts.charts import Line
import datetime
import plotly.express as px
import pickle

from pyecharts import options as opts
from streamlit_plotly_events import plotly_events
import plotly.graph_objs as go

from tsod.active_learning.data_prep import construct_training_data
from streamlit_profiler import Profiler
from contextlib import nullcontext


SELECT_OPTIONS = {"Point": "zoom", "Box": "select", "Lasso": "lasso"}
SELECT_INFO = {
    "Point": "Select individual points, drag mouse to zoom in.",
    "Box": "Draw Box to select all points in range.",
    "Lasso": "Draw Lasso to select all points in range.",
}


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("data/Elev_NW1.csv", index_col=0, parse_dates=True)
    df["Water Level"] = df["Water Level"].astype(np.float16)
    # df["ts"] = df.index.values.astype(np.int64) // 10**6
    # df.set_index("ts", inplace=True)
    return df


@st.cache
def filter_on_date(df, start, end):
    return df[df.index.to_series().between(start, end)]


def prepare_session_state():
    if "selected_points" not in st.session_state:
        st.session_state.selected_points = set()
    if "marked_outlier" not in st.session_state:
        st.session_state.marked_outlier = set()
    if "marked_normal" not in st.session_state:
        st.session_state.marked_normal = set()

    if "df_marked_out" not in st.session_state:
        st.session_state.df_marked_out = pd.DataFrame()
    if "df_marked_not_out" not in st.session_state:
        st.session_state.df_marked_not_out = pd.DataFrame()
    if "df_plot" not in st.session_state:
        st.session_state.df_plot = pd.DataFrame()


def clear_selection():
    st.session_state.selected_points = set()


def mark_selected_outlier():
    if st.session_state.selected_points.intersection(st.session_state.marked_normal):
        st.warning(
            "Some of the selected points have already been marked as normal and are ignored."
        )

    to_add = st.session_state.selected_points.difference(st.session_state.marked_normal)

    if not to_add.issubset(st.session_state.marked_outlier):
        st.session_state.marked_outlier.update(to_add)
        clear_selection()


def mark_selected_not_outlier():
    if st.session_state.selected_points.intersection(st.session_state.marked_outlier):
        st.warning(
            "Some of the selected points have already been marked as outliers and are ignored."
        )

    to_add = st.session_state.selected_points.difference(st.session_state.marked_outlier)

    if not to_add.issubset(st.session_state.marked_normal):
        st.session_state.marked_normal.update(to_add)
        clear_selection()


def update_selected_points(selection: set):
    if not selection.issubset(st.session_state.selected_points):
        for s in selection:
            # Plotly sometimes returns selected points as timestamp
            if isinstance(s, int):
                st.session_state.selected_points.add(datetime.datetime.fromtimestamp(s / 1000))
            else:
                st.session_state.selected_points.add(s)
        st.session_state.selected_points.update(selection)
        st.experimental_rerun()


@st.experimental_memo(persist="disk")
def create_cachable_line_plot(start_time, end_time, file_identifier: str = "TODO") -> go.Figure:
    plot_data = st.session_state.df_plot
    timestamps = plot_data.index.to_list()

    return px.line(
        plot_data,
        x=timestamps,
        y="Water Level",
        markers=True,
    )


def create_annotation_plot(plot_data: pd.DataFrame, base_obj=None) -> go.Figure:
    obj = base_obj or st
    obj.subheader("Data selection options")
    selection_method = obj.selectbox("Data Selection Method", list(SELECT_OPTIONS.keys()))
    obj.info(SELECT_INFO[selection_method])
    obj.markdown("***")

    fig = create_cachable_line_plot(st.session_state.start_time, st.session_state.end_time)

    df_selected = plot_data[plot_data.index.isin(st.session_state.selected_points)]
    df_marked_out = plot_data[plot_data.index.isin(st.session_state.marked_outlier)]
    df_marked_not_out = plot_data[plot_data.index.isin(st.session_state.marked_normal)]

    st.session_state["df_marked_out"] = df_marked_out
    st.session_state["df_marked_not_out"] = df_marked_not_out

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=df_selected.index,
            y=df_selected["Water Level"],
            name="Selected",
            marker=dict(color="Purple", size=8, line=dict(color="Black", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=df_marked_out.index,
            y=df_marked_out["Water Level"],
            name="Marked Outlier",
            marker=dict(color="Red", size=12, line=dict(color="Black", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=df_marked_not_out.index,
            y=df_marked_not_out["Water Level"],
            name="Marked Not Outlier",
            marker=dict(color="Green", size=10, line=dict(color="Black", width=0.5)),
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
    inp_col_1, inp_col_2 = obj.columns(2)
    # inp_col_1, inp_col_2, inp_col_3 = obj.columns([2, 2, 10], gap="medium")

    start_date = inp_col_1.date_input(
        "Graph range start date",
        value=data.index.max().date() - datetime.timedelta(days=7),
        min_value=data.index.min(),
        max_value=data.index.max(),
    )
    end_date = inp_col_2.date_input(
        "Graph range end date",
        value=data.index.max().date(),
        min_value=data.index.min(),
        max_value=data.index.max(),
    )

    start_time = inp_col_1.time_input("Start time", value=datetime.time(0, 0, 0))
    end_time = inp_col_2.time_input("End time", value=datetime.time(23, 59, 59))

    obj.markdown("***")

    start_dt = datetime.datetime.combine(start_date, start_time)
    end_dt = datetime.datetime.combine(end_date, end_time)

    st.session_state.start_time = start_dt
    st.session_state.end_time = end_dt

    df_plot = filter_on_date(data, start_dt, end_dt)

    st.session_state["df_plot"] = df_plot

    return df_plot


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

    obj.subheader("Annotation actions")
    obj.button("Clear selection", on_click=clear_selection)

    c_1, c_2 = obj.columns(2)
    c_1.button("Mark selection Outlier", on_click=mark_selected_outlier)
    c_2.button("Mark selection not Outlier", on_click=mark_selected_not_outlier)
    obj.markdown("***")


def create_save_load_buttons(base_obj=None):
    if st.session_state.df_marked_out.empty:
        return

    obj = base_obj or st

    obj.subheader("Save / load previous")

    c_1, c_2 = obj.columns(2)

    to_save = prepare_download(
        st.session_state["df_marked_out"], st.session_state["df_marked_not_out"]
    )
    file_name = f"Annotations_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.bin"

    c_1.download_button("Save annotations to disk", to_save, file_name=file_name)

    obj.markdown("***")


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Controls")
    profile = st.sidebar.checkbox("Profile Code", value=False)

    with Profiler() if profile else nullcontext():

        prepare_session_state()
        data = load_data()
        st.session_state["df_full"] = data

        tab1, tab2 = st.tabs(["Outlier annotation", "Model prediction"])
        create_plot_buttons(st.sidebar)
        create_save_load_buttons(st.sidebar)
        plot_data = filter_data(data, st.sidebar)

        fig = create_annotation_plot(plot_data, st.sidebar)

        with tab1:
            selection = plotly_events(fig, select_event=True, override_height=1000)
        selection = {e["x"] for e in selection}

        update_selected_points(selection)

        with st.spinner("Constructing features..."):
            construct_training_data()

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
