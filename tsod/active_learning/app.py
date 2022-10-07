from typing import List
import streamlit as st

from streamlit_echarts import st_echarts
from streamlit_echarts import st_pyecharts
import pandas as pd
import numpy as np
from pyecharts.charts import Line
import datetime
import plotly.express as px

from pyecharts import options as opts
from streamlit_plotly_events import plotly_events
import plotly.graph_objs as go


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


def main():
    st.set_page_config(layout="wide")
    prepare_session_state()
    data = load_data()
    tab1, tab2 = st.tabs(["Outlier annotation", "Model prediction"])
    inp_col_1, inp_col_2, inp_col_3 = tab1.columns([2, 2, 10], gap="medium")

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

    inp_col_1.markdown("***")
    inp_col_2.markdown("***")
    selection_method = inp_col_1.selectbox("Data Selection Method", list(SELECT_OPTIONS.keys()))
    inp_col_2.info(SELECT_INFO[selection_method])
    button_cols = tab1.columns([1, 1, 1, 7])

    button_cols[0].button("Clear selection", on_click=clear_selection)
    button_cols[1].button("Mark selected outlier", on_click=mark_selected_outlier)
    button_cols[2].button("Mark selected not outlier", on_click=mark_selected_not_outlier)

    start_dt = datetime.datetime.combine(start_date, start_time)
    end_dt = datetime.datetime.combine(end_date, end_time)

    plot_data = filter_on_date(data, start_dt, end_dt)

    timestamps = plot_data.index.to_list()
    fig = px.line(
        plot_data,
        x=timestamps,
        y="Water Level",
        markers=True,
        width=1200,
        height=700,
    )

    df_selected = plot_data[plot_data.index.isin(st.session_state.selected_points)]
    df_marked_out = plot_data[plot_data.index.isin(st.session_state.marked_outlier)]
    df_marked_not_out = plot_data[plot_data.index.isin(st.session_state.marked_normal)]

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
    with tab1:
        selection = plotly_events(fig, select_event=True, override_height=700)
    selection = {e["x"] for e in selection}
    selection

    update_selected_points(selection)

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
