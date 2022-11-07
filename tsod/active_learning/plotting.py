from collections import defaultdict
import datetime
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Scatter
from streamlit_echarts import st_pyecharts
from tsod.active_learning.utils import SELECT_INFO, SELECT_OPTIONS, get_as

ANNOTATION_COLORS = {
    "selected": "#8c259a",
    "outlier": "#e60b0b",
    "normal": "#3fc762",
    "test_outlier": "#fd7c99",
    "test_normal": "#0fefc7",
}
MARKER_SIZES = {"selected": 10, "outlier": 12, "normal": 12, "test_outlier": 12, "test_normal": 12}
MARKER_VALUES = {
    "selected": "S",
    "outlier": "O",
    "normal": "N",
    "test_outlier": "TO",
    "test_normal": "TN",
}
MARKER_HOVER = {
    "selected": "Selected Point",
    "outlier": "Training Outlier",
    "normal": "Traning Normal",
    "test_outlier": "Test Outlier",
    "test_normal": "Test Normal",
}


@st.experimental_memo(persist="disk", show_spinner=False)
def create_cachable_line_plot(
    start_time, end_time, data_file_identifier: str = "TODO"
) -> go.Figure:
    with st.spinner("Fetching updated annotation plot..."):
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

    for series_name in state.data:
        if not hasattr(state, f"df_plot_{series_name}"):
            continue
        df_series: pd.DataFrame = getattr(state, f"df_plot_{series_name}")
        if df_series.empty:
            continue

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_series.index,
                y=df_series["Water Level"],
                name=f"{series_name.replace('_', ' ').title()} ({len(df_series)})",
                marker=dict(
                    color=ANNOTATION_COLORS[series_name],
                    size=MARKER_SIZES[series_name],
                    line=dict(color="Black", width=1),
                ),
            )
        )

    fig.update_layout(dragmode=SELECT_OPTIONS[selection_method])
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
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


@st.cache(persist=True, max_entries=100, show_spinner=False)
def cachable_get_outlier_counts(
    dataset_name: str,
    model_names: List,
    train_outliers: set,
    test_outliers: set,
    number_of_datapoints: int,
) -> pd.DataFrame:
    with st.spinner("Creating new distribution plot..."):
        state = get_as()
        dataset: pd.DataFrame = st.session_state["inference_results"][dataset_name]

        dataset["outlier_group"] = range(len(dataset))
        dataset["outlier_group"] = (dataset["outlier_group"] // number_of_datapoints).astype(
            np.int16
        )

        threshold_timestamps = (
            dataset.reset_index().groupby("outlier_group")["index"].first().to_list()
        )
        threshold_timestamps.append(dataset.index.max())
        ranges = [f"{i} - {j}" for i, j in zip(threshold_timestamps, threshold_timestamps[1:])]

        df_out = pd.DataFrame(
            index=ranges, columns=model_names + ["Marked Train Outliers", "Marked Test Outliers"]
        )

        annotated_outliers = state.df_outlier
        annotated_test_outliers = state.df_test_outlier
        for group_index, (_, group) in enumerate(dataset.groupby("outlier_group")):
            for model_index, model in enumerate(model_names):
                df_out.iat[group_index, model_index] = group[model].sum().item()

            df_out.iat[group_index, -2] = len(
                annotated_outliers[
                    annotated_outliers.index.to_series().between(group.index[0], group.index[-1])
                ]
            )
            df_out.iat[group_index, -1] = len(
                annotated_test_outliers[
                    annotated_test_outliers.index.to_series().between(
                        group.index[0], group.index[-1]
                    )
                ]
            )
        return df_out


def make_outlier_distribution_plot(dataset_name: str):
    model_names = sorted(st.session_state["models_to_visualize"][dataset_name])
    if not model_names:
        return None, None
    state = get_as()

    df_counts = cachable_get_outlier_counts(
        dataset_name,
        model_names,
        state.outlier,
        state.test_outlier,
        number_of_datapoints=st.session_state[f"num_outliers_{dataset_name}"],
    )

    if st.session_state[f"only_show_ranges_with_outliers_{dataset_name}"]:
        df_counts = df_counts[df_counts.any(axis=1)]

    st.session_state[f"current_ranges_counts_{dataset_name}"] = df_counts

    bar = (
        Bar()
        .add_xaxis(df_counts.index.to_list())
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Distribution of outliers per model",
                subtitle="Click on bar to isolate time range",
            ),
            xaxis_opts=opts.AxisOpts(
                is_scale=True,
                name="Time Range",
                name_location="middle",
                name_gap=30,
                axistick_opts=opts.AxisTickOpts(is_inside=True, is_align_with_label=True),
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name="Number of outliers",
                name_rotate=90,
                name_location="middle",
                name_gap=50,
                boundary_gap="30%",
            ),
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
            legend_opts=opts.LegendOpts(pos_top=10, pos_right=10, orient="vertical"),
            tooltip_opts=opts.TooltipOpts(
                axis_pointer_type="shadow",
                trigger="axis",
            ),
        )
    )
    for series in df_counts.columns:
        if df_counts[series].any():
            bar = bar.add_yaxis(
                series,
                df_counts[series].to_list(),
                stack=series,
                label_opts=opts.LabelOpts(is_show=False),
                category_gap="40%",
            )

    colors = [st.session_state[f"color_{m}_{dataset_name}"] for m in model_names]
    if state.outlier:
        colors.append("#e60b0b")
    if state.test_outlier:
        colors.append("#fd7c99")

    bar.set_colors(colors)
    clicked_range = st_pyecharts(
        bar,
        height=f"{st.session_state[f'figure_height_{dataset_name}']}px",
        theme="dark",
        events={
            "click": "function(params) { return params.name }",
            # "dblclick": "function(params) { console.log(params) } ",
        },
        key=f"distribution_plot_{dataset_name}",
    )

    def _get_start_and_end_date(clicked_range: str):
        if not clicked_range:
            return None, None
        start_str, end_str = clicked_range.split(" - ")
        start_time = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        st.session_state[f"last_clicked_range_{dataset_name}"] = start_time, end_time
        st.session_state[f"range_str_{dataset_name}"] = clicked_range
        return start_time, end_time

    return _get_start_and_end_date(clicked_range)


def get_echarts_plot_time_range(
    start_time,
    end_time,
    data_column,
    include_annotations,
    plot_title: str,
    dataset_name: str = None,
):
    state = get_as()
    state.update_plot(start_time, end_time)

    x_data = state.df_plot.index.to_list()
    y_data = state.df_plot[data_column].to_list()
    plot = (
        Line(init_opts=opts.InitOpts(animation_opts=opts.AnimationOpts(animation=False)))
        .add_xaxis(x_data)
        .add_yaxis(
            data_column,
            y_data,
            color="yellow",
            label_opts=opts.LabelOpts(is_show=False),
            # markpoint_opts=opts.MarkPointOpts(data=markers),
            is_symbol_show=False,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=plot_title,
                subtitle="Click on points or markers to select them. Activate different selection modes in the toolbar.",
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name=data_column,
                name_rotate=90,
                name_location="middle",
                name_gap=50,
            ),
            xaxis_opts=opts.AxisOpts(
                type_="time",
                is_scale=True,
                name="Date & Time",
                name_location="middle",
                name_gap=-20,
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    type_="slider",
                    range_start=0,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    type_="inside",
                ),
            ],
            legend_opts=opts.LegendOpts(pos_top=40, pos_right=10, orient="vertical"),
            brush_opts=opts.BrushOpts(
                throttle_type="debounce",
                throttle_delay=500,
                brush_mode="multiple",
                brush_type="lineX",
                tool_box=["lineX", "rect", "clear"],
                # brush_link="all",
                # geo_index=0,
                series_index="all",
                # x_axis_index=0,
                # x_axis_index=2,
                # y_axis_index=2,
                # out_of_brush={"symbol": "pin"},
                out_of_brush={"colorAlpha": 0.1},
            ),
            tooltip_opts=opts.TooltipOpts(axis_pointer_type="line", trigger="axis"),
        )
    )

    scatter = (
        Scatter()
        .add_xaxis(x_data)
        .add_yaxis(
            "Datapoints",
            y_data,
            label_opts=opts.LabelOpts(is_show=False),
            symbol_size=3,
            itemstyle_opts=opts.ItemStyleOpts(color="#dce4e3"),
            is_selected=len(x_data) < 2000,
            tooltip_opts=opts.TooltipOpts(is_show=False),
        )
    )
    plot = plot.overlap(scatter)
    if not include_annotations:
        return plot

    df_selected = getattr(state, "df_plot_selected", None)

    for series_name in state.data:
        if not hasattr(state, f"df_plot_{series_name}"):
            continue

        df = getattr(state, f"df_plot_{series_name}")
        if df_selected is not None and series_name != "selected":
            df = df[~df.index.isin(df_selected.index)]
        if df.empty:
            continue
        plot.overlap(
            Scatter()
            .add_xaxis(df.index.to_list())
            .add_yaxis(
                MARKER_HOVER[series_name],
                df["Water Level"].to_list(),
                label_opts=opts.LabelOpts(is_show=False),
                symbol_rotate=0,
                symbol="roundRect",
                symbol_size=15,
                # color="#dce4e3",
                itemstyle_opts=opts.ItemStyleOpts(opacity=1, color=ANNOTATION_COLORS[series_name]),
                tooltip_opts=opts.TooltipOpts(is_show=False),
            )
        )
    return plot


def make_time_range_outlier_plot(dataset_name: str, start_time, end_time):
    symbols = ["pin", "arrow", "diamond", "triangle"]
    sizes = [50, 30, 30, 30]

    dataset: pd.DataFrame = st.session_state["inference_results"][dataset_name]

    model_names = sorted(st.session_state["models_to_visualize"][dataset_name])

    df_plot = dataset[dataset.index.to_series().between(start_time, end_time)]

    plot = get_echarts_plot_time_range(
        start_time,
        end_time,
        "Water Level",
        plot_title=f"Outliers {start_time} - {end_time}",
        include_annotations=True,
    )

    # x_data = df_plot.index.to_list()
    # y_data = df_plot["Water Level"].to_list()
    # markers = []
    # outlier_value_store = defaultdict(dict)
    pred_outlier_tracker = {}
    for model_number, model_name in enumerate(model_names):
        df_outlier = df_plot[df_plot[model_name] > 0]
        pred_outlier_tracker[model_name] = df_outlier
        if df_outlier.empty:
            continue
        plot.overlap(
            Scatter()
            .add_xaxis(df_outlier.index.to_list())
            .add_yaxis(
                model_name,
                df_outlier["Water Level"].to_list(),
                label_opts=opts.LabelOpts(is_show=False),
                symbol_rotate=-90 * model_number,
                symbol="pin",
                symbol_size=40,
                itemstyle_opts=opts.ItemStyleOpts(
                    opacity=1, color=st.session_state[f"color_{model_name}_{dataset_name}"]
                ),
                tooltip_opts=opts.TooltipOpts(formatter="{a} <br>Outlier predicted"),
            )
        )

    st.session_state["pred_outlier_tracker"] = pred_outlier_tracker

    clicked_point = st_pyecharts(
        plot,
        height=f"{st.session_state[f'figure_height_{dataset_name}']}px",
        theme="dark",
        events={
            "click": "function(params) { return [params.data[0], 'click'] }",
            # "brushselected": "function(params) { console.log(params) }",
            "brushselected": "function(params) { return [params.batch[0].selected, 'brush'] }",
            # "brushselected": "function(params) { return [params.batch[0].selected[1].dataIndex, 'brush'] }",
        },
        # key=f"time_range_plot_{dataset_name}",
    )

    return clicked_point


def feature_importance_plot(base_obj=None):
    obj = base_obj or st

    df_new: pd.DataFrame = st.session_state["current_importances"]

    if "previous_importances" in st.session_state:
        df_old: pd.DataFrame = st.session_state["previous_importances"]
        df_plot = df_new.merge(df_old, how="left", on="Feature", suffixes=("", " before"))
        df_plot["diff"] = (
            df_plot["Feature importance"] - df_plot["Feature importance before"]
        ).round(3)
        df_plot["diff_text"] = df_plot["diff"].apply(lambda x: str(x) if x <= 0 else f"+{x}")
    else:
        df_plot = df_new

    df_plot = df_plot.iloc[:10].sort_values("Feature importance")

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot["Feature importance"],
                y=df_plot["Feature"],
                orientation="h",
                name="Current Importances",
                text=df_plot["Feature importance"],
            )
        ]
    )

    if "previous_importances" in st.session_state:
        fig.add_trace(
            go.Bar(
                x=df_plot["diff"],
                y=df_plot["Feature"],
                orientation="h",
                name="Change to previous model",
                text=df_plot["diff_text"],
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=0),
        legend=dict(yanchor="bottom", y=1.0, xanchor="left", x=0.2, orientation="h"),
        barmode="relative",
        title={
            "text": f"Feature importances {st.session_state['last_model_name']}",
            "y": 1.0,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        # title_text=f"Feature importances {st.session_state['last_model_name']}",
    )
    # with obj:
    # st_pyecharts(bar, theme="dark")
    # obj.bar_chart(df_new, x=["Feature importance"])
    obj.plotly_chart(fig, use_container_width=True)
