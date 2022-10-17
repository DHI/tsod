from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Line
from streamlit_echarts import st_echarts, st_pyecharts
from streamlit_plotly_events import plotly_events
from streamlit_profiler import Profiler
from tsod.active_learning.utils import SELECT_INFO, SELECT_OPTIONS, get_as


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
                        # dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            # rangeslider=dict(visible=True, autorange=True),
            type="date",
        ),
    )

    return fig


def make_prediction_plot_for_dataset(dataset_name: str, base_obj=None):
    obj = base_obj or st
    obj.subheader(dataset_name)
    c1, c2, c3 = obj.columns(3)

    colors = ["red", "green", "blue"]

    model_names = list(st.session_state["inference_results"][dataset_name].keys())

    markers = []
    dataset = st.session_state["prediction_data"][dataset_name]
    x_data = dataset.index.to_list()
    y_data = dataset["Water Level"].to_list()

    info_table = []

    for model_number, model_name in enumerate(model_names):
        model_predictions = st.session_state["inference_results"][dataset_name][model_name]

        outliers_idc = np.random.choice(model_predictions.nonzero()[0], 10)
        info_table.append(
            {
                "Model Name": model_name,
                "Predicted Outliers": outliers_idc.shape[0],
                "Predicted Normal": len(dataset) - outliers_idc.shape[0],
            }
        )

        df_marker: pd.DataFrame = dataset.iloc[outliers_idc].sort_index()

        for i, (idx, row) in enumerate(df_marker.iterrows()):
            markers.append(
                opts.MarkPointItem(
                    name=f"Outlier {i+1} {model_name}",
                    coord=[idx, row["Water Level"].item()],
                    symbol="pin",
                    itemstyle_opts=opts.ItemStyleOpts(color=colors[model_number]),
                    value=i + 1,
                )
            )

    obj.table(info_table)
    line = (
        Line()
        .add_xaxis(x_data)
        # .add_xaxis(pd.Series(state.df_plot.index).astype(str).to_list())
        .add_yaxis(
            "Water Level",
            y_data,
            color="yellow",
            label_opts=opts.LabelOpts(is_show=False),
            # markpoint_opts=opts.MarkPointOpts(
            # data=[{"coord": [x_data[10], y_data[10]], "name": "TESTER"}]
            # )
            markpoint_opts=opts.MarkPointOpts(
                data=markers,
                # data=[
                #     opts.MarkPointItem(
                #         name="Outlier 1 Model 1",
                #         coord=[x_data[300], y_data[300]],
                #         symbol="pin",
                #         itemstyle_opts=opts.ItemStyleOpts(color="red"),
                #     )
                # ]
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="time", is_scale=True, name="Date & Time"),
            yaxis_opts=opts.AxisOpts(type_="value", name="Water Level"),
            datazoom_opts=[
                opts.DataZoomOpts(
                    type_="slider",
                    range_start=0,
                    range_end=70,
                ),
                opts.DataZoomOpts(
                    type_="inside",
                    range_start=0,
                    range_end=100,
                ),
            ],
        )
    )
    # tooltip_opts = opts.TooltipOpts(axis_pointer_type="cross")

    st_pyecharts(line, height="500px", theme="dark")
