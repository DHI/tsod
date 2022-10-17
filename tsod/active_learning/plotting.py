from collections import defaultdict
import datetime
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Line, Bar
from streamlit_echarts import st_echarts, st_pyecharts
from streamlit_plotly_events import plotly_events
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

    colors = ["red", "green", "blue"]
    symbols = ["pin", "arrow", "diamond"]

    dataset: pd.DataFrame = st.session_state["prediction_data"][dataset_name]

    model_predictions = st.session_state["inference_results"][dataset_name]
    MODEL_NAMES = list(model_predictions.keys())

    predicted_outliers = {
        model_name: model_predictions[model_name].nonzero()[0].tolist()
        for model_name in MODEL_NAMES
    }

    for model_name, model_preds in model_predictions.items():
        dataset[model_name] = model_preds

    dataset["outlier_group"] = dataset[MODEL_NAMES].sum(axis=1).cumsum() // 10

    ts = []
    outlier_counts = defaultdict(list)
    for i, (_, group) in enumerate(dataset.groupby("outlier_group")):
        for model in MODEL_NAMES:
            outlier_counts[model].append(group[model].sum().item())
        if i > 0:
            ts.append(group.index[0])

        if i == 5:
            break

    ts.insert(0, dataset.index.min())
    # ts.append(dataset.index.max())

    ranges = [f"{ts[i]} - {ts[i+1]}" for i in range(len(ts) - 1)]

    info_table = [
        {
            "Model Name": model_name,
            "Predicted Outliers": len(outliers),
            "Predicted Normal": len(dataset) - len(outliers),
        }
        for model_name, outliers in predicted_outliers.items()
    ]

    obj.table(info_table)

    bar = (
        Bar()
        .add_xaxis(ranges)
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Distribution of outliers per model",
                subtitle="Click on bar to isolate time range",
            ),
            xaxis_opts=opts.AxisOpts(
                is_scale=True, name="Time Range", name_location="middle", name_gap=30
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name="Number of outliers",
                name_rotate=90,
                name_location="middle",
                name_gap=50,
            ),
        )
    )

    for i, model in enumerate(MODEL_NAMES):
        bar = bar.add_yaxis(
            model,
            outlier_counts[model],
            stack=True,
            label_opts=opts.LabelOpts(is_show=False),
            color=colors[i],
        )

    clicked_range = st_pyecharts(
        bar,
        height="500px",
        theme="dark",
        events={"click": "function(params) { return params.name }"},
    )

    def _get_start_and_end_date(clicked_range: str):
        start_str, end_str = clicked_range.split(" - ")

        start_time = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")

        return start_time, end_time

    if not clicked_range:
        return

    start_time, end_time = _get_start_and_end_date(clicked_range)
    df_plot = dataset[dataset.index.to_series().between(start_time, end_time)]
    x_data = df_plot.index.to_list()
    y_data = df_plot["Water Level"].to_list()

    st.write(df_plot)

    # return
    markers = []

    for model_number, model_name in enumerate(MODEL_NAMES):

        # outliers_idc = sorted(np.random.choice(model_predictions.nonzero()[0], 20).tolist())
        counter = 1
        for i, row in df_plot.iterrows():
            # for i, int_index in enumerate(outliers_idc):
            if row[model_name] == 1:
                markers.append(
                    opts.MarkPointItem(
                        name=f"Outlier {counter} {model_name}",
                        coord=[i, row["Water Level"].item()],
                        # coord=[x_data[int_index], y_data[int_index]],
                        symbol=symbols[model_number],
                        # symbol="pin",
                        itemstyle_opts=opts.ItemStyleOpts(color=colors[model_number]),
                        value=counter,
                    )
                )
                counter += 1

    line = (
        Line()
        .add_xaxis(x_data)
        # .add_xaxis(pd.Series(state.df_plot.index).astype(str).to_list())
        .add_yaxis(
            "Water Level",
            y_data,
            color="yellow",
            label_opts=opts.LabelOpts(is_show=False),
            markpoint_opts=opts.MarkPointOpts(data=markers),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="time", is_scale=True, name="Date & Time"),
            yaxis_opts=opts.AxisOpts(type_="value", name="Water Level"),
            datazoom_opts=[
                opts.DataZoomOpts(
                    type_="slider",
                    range_start=0,
                    range_end=100,
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
