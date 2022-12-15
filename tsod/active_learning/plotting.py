import copy
import datetime
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Scatter, EffectScatter
from streamlit_echarts import st_pyecharts
from tsod.active_learning.utils import get_as

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


@st.cache(persist=True, max_entries=100, show_spinner=False)
def cachable_get_outlier_counts(
    dataset_name: str,
    series: str,
    model_names: List,
    train_outliers: set,
    test_outliers: set,
    number_of_datapoints: int,
) -> pd.DataFrame:
    with st.spinner("Creating new distribution plot..."):
        state = get_as(dataset_name, series)
        dataset: pd.DataFrame = st.session_state["inference_results"][dataset_name][series]

        dataset["outlier_group"] = range(len(dataset))
        dataset["outlier_group"] = (dataset["outlier_group"] // number_of_datapoints).astype(
            np.int16
        )

        threshold_timestamps = (
            dataset.reset_index().groupby("outlier_group")["index"].first().to_list()
        )
        threshold_timestamps.append(dataset.index.max())
        ranges = [f"{i} - {j}" for i, j in zip(threshold_timestamps, threshold_timestamps[1:])]

        out_columns = copy.deepcopy(model_names)
        for m in model_names:
            out_columns.extend([f"{m} Missed Train Outliers", f"{m} Missed Test Outliers"])
        out_columns.extend(["Marked Train Outliers", "Marked Test Outliers"])
        df_out = pd.DataFrame(index=ranges, columns=out_columns)

        annotated_outliers = state.df_outlier
        annotated_test_outliers = state.df_test_outlier
        for group_index, (_, group) in enumerate(dataset.groupby("outlier_group")):
            outliers_in_this_group = annotated_outliers[
                annotated_outliers.index.to_series().between(group.index[0], group.index[-1])
            ]
            test_outliers_in_this_group = annotated_test_outliers[
                annotated_test_outliers.index.to_series().between(group.index[0], group.index[-1])
            ]
            df_out.iat[group_index, -2] = len(outliers_in_this_group)
            df_out.iat[group_index, -1] = len(test_outliers_in_this_group)

            for model_index, model in enumerate(model_names):
                model_pred_outliers = group[group[model] == 1].index
                df_out.iat[group_index, model_index] = len(model_pred_outliers)
                df_out.at[ranges[group_index], f"{model} Missed Train Outliers"] = np.count_nonzero(
                    outliers_in_this_group.index.isin(model_pred_outliers) == False
                )
                df_out.at[ranges[group_index], f"{model} Missed Test Outliers"] = np.count_nonzero(
                    test_outliers_in_this_group.index.isin(model_pred_outliers) == False
                )

        return df_out


def make_outlier_distribution_plot(dataset_name: str, series: str):
    model_names = sorted(st.session_state["models_to_visualize"][dataset_name][series])
    if not model_names:
        return None, None
    state = get_as(dataset=dataset_name, column=series)

    df_counts = cachable_get_outlier_counts(
        dataset_name,
        series,
        model_names,
        state.outlier,
        state.test_outlier,
        number_of_datapoints=st.session_state[f"num_outliers_{dataset_name}_{series}"],
    )

    if st.session_state[f"only_show_ranges_with_outliers_{dataset_name}_{series}"]:
        df_counts = df_counts[df_counts.any(axis=1)]

    bar = (
        Bar()
        .add_xaxis(df_counts.index.to_list())
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Distribution Plot - Number of outliers per model",
                subtitle="Click on bar to isolate time range",
                padding=15,
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
    for ann_series in df_counts.columns:
        if "missed" in ann_series.lower():
            continue
        if df_counts[ann_series].any():
            bar = bar.add_yaxis(
                ann_series,
                df_counts[ann_series].to_list(),
                stack=ann_series,
                label_opts=opts.LabelOpts(is_show=False),
                category_gap="40%",
            )

    colors = [st.session_state[f"color_{m}_{dataset_name}_{series}"] for m in model_names]
    if state.outlier:
        colors.append("#e60b0b")
    if state.test_outlier:
        colors.append("#fd7c99")

    for m in model_names:
        if not (
            st.session_state.get(f"highlight_train_{dataset_name}_{series}")
            or st.session_state.get(f"highlight_test_{dataset_name}_{series}")
        ):
            break
        df_missed = df_counts[df_counts[f"{m} Missed Train Outliers"] > 0]
        if st.session_state.get(f"highlight_train_{dataset_name}_{series}"):
            df_missed = df_counts[df_counts[f"{m} Missed Train Outliers"] > 0]
            if not df_missed.empty:
                effect_scatter = (EffectScatter().add_xaxis(df_missed.index.tolist())).add_yaxis(
                    f"{m} Missed Training Outliers",
                    df_missed[m].tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    tooltip_opts=opts.TooltipOpts(is_show=False),
                    symbol="triangle",
                )
                bar = bar.overlap(effect_scatter)
                colors.append(st.session_state[f"color_{m}_{dataset_name}_{series}"])
        if st.session_state.get(f"highlight_test_{dataset_name}_{series}"):
            df_missed = df_counts[df_counts[f"{m} Missed Test Outliers"] > 0]
            if not df_missed.empty:
                effect_scatter = (EffectScatter().add_xaxis(df_missed.index.tolist())).add_yaxis(
                    f"{m} Missed Test Outliers",
                    df_missed[m].tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    tooltip_opts=opts.TooltipOpts(is_show=False),
                    symbol_size=15,
                )
                bar = bar.overlap(effect_scatter)
                colors.append(st.session_state[f"color_{m}_{dataset_name}_{series}"])

    bar.set_colors(colors)

    clicked_range = st_pyecharts(
        bar,
        height=f"{st.session_state[f'figure_height_{dataset_name}_{series}']}px",
        theme="dark",
        events={
            "click": "function(params) { return params.name }",
            # "dblclick": "function(params) { console.log(params) } ",
        },
        # key=f"distribution_plot_{dataset_name}",
    )

    def _get_start_and_end_date(clicked_range: str):
        if not clicked_range:
            return None, None
        start_str, end_str = clicked_range.split(" - ")
        start_time = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        st.session_state[f"last_clicked_range_{dataset_name}_{series}"] = start_time, end_time
        # st.session_state[f"range_str_{dataset_name}"] = clicked_range
        return start_time, end_time

    return _get_start_and_end_date(clicked_range)


def make_annotation_suggestion_plot(
    start_time, end_time, dataset_name, series, point_to_highlight: tuple
):
    state = get_as(dataset_name, series)
    state.update_plot(start_time, end_time)

    x_data = state.df_plot.index.to_list()
    y_data = state.df_plot[series].to_list()
    plot = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis(
            series,
            y_data,
            color="yellow",
            label_opts=opts.LabelOpts(is_show=False),
            is_symbol_show=False,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Is this point an outlier?",
                subtitle="Labels generated here will be added directly to the training data.",
                padding=15,
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name=series,
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
            datazoom_opts=opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            legend_opts=opts.LegendOpts(pos_top=40, pos_right=10, orient="vertical"),
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
            is_selected=len(x_data) < 10000,
            tooltip_opts=opts.TooltipOpts(is_show=False),
        )
    )
    plot = plot.overlap(scatter)
    effect_scatter = (
        EffectScatter()
        .add_xaxis([point_to_highlight[0]])
        .add_yaxis(
            "Candidate",
            [point_to_highlight[1]],
            label_opts=opts.LabelOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(is_show=False),
            symbol_size=30,
            symbol="pin",
        )
    )
    plot = plot.overlap(effect_scatter)

    st_pyecharts(plot, theme="dark", height="600px")


def get_echarts_plot_time_range(
    start_time,
    end_time,
    data_column,
    include_annotations,
    plot_title: str,
    dataset_name: str = None,
    series: str = None,
):
    state = get_as(dataset_name, series)
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
                subtitle="Click on points or markers to select them. Activate different selection modes in the toolbar (top right). Zoom using the mouse wheel.",
                padding=15,
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
                series_index="all",
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
            is_selected=len(x_data) < 10000,
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
                df[state.column].to_list(),
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


def make_time_range_outlier_plot(dataset_name: str, series: str, start_time, end_time):
    dataset: pd.DataFrame = st.session_state["inference_results"][dataset_name][series]
    model_names = sorted(st.session_state["models_to_visualize"][dataset_name][series])

    df_plot = dataset[dataset.index.to_series().between(start_time, end_time)]
    state = get_as(dataset_name, series)

    plot = get_echarts_plot_time_range(
        start_time,
        end_time,
        state.column,
        plot_title=f"Outlier Plot {start_time} - {end_time}",
        include_annotations=True,
        dataset_name=dataset_name,
        series=series,
    )

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
                df_outlier[state.column].to_list(),
                label_opts=opts.LabelOpts(is_show=False),
                symbol_rotate=-90 * model_number,
                symbol="pin",
                symbol_size=40,
                itemstyle_opts=opts.ItemStyleOpts(
                    opacity=1, color=st.session_state[f"color_{model_name}_{dataset_name}_{series}"]
                ),
                tooltip_opts=opts.TooltipOpts(formatter="{a} <br>Outlier predicted"),
            )
        )

    st.session_state["pred_outlier_tracker"][dataset_name][series] = pred_outlier_tracker

    clicked_point = st_pyecharts(
        plot,
        height=f"{st.session_state[f'figure_height_{dataset_name}_{series}']}px",
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

    state = get_as()
    dataset = state.dataset
    series = state.column

    df_new: pd.DataFrame = st.session_state[f"current_importances_{dataset}_{series}"]

    if f"previous_importances_{dataset}_{series}" in st.session_state:
        df_old: pd.DataFrame = st.session_state[f"previous_importances_{dataset}_{series}"]
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

    if f"previous_importances_{dataset}_{series}" in st.session_state:
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
            "text": f"Feature importances {st.session_state[f'last_model_name_{dataset}_{series}']}",
            "y": 1.0,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )
    obj.plotly_chart(fig, use_container_width=True)


def make_removed_outliers_example_plots(df_before: pd.DataFrame, df_new: pd.DataFrame):
    dataset: str = st.session_state["download_dataset"]
    series: List = st.session_state["download_series"]
    model: str = st.session_state["download_model"]

    st.header(f"Outlier removal preview - {model}")

    c1, c2 = st.columns(2)
    c1.metric("Current number of total rows in dataset (all series)", len(df_before))
    c2.metric("Total number of rows after outlier removal (all series)", len(df_new))

    for s in series:
        outlier_mask = df_before[f"{s}_{model}"] == 1
        number_outlier = outlier_mask.sum()
        number_to_show = min(number_outlier, 3)
        changes_sample = (
            df_before[df_before[f"{s}_{model}"] == 1].sample(number_to_show, random_state=1).index
        )

        st.subheader(s)

        cols = st.columns(3)
        cols[0].metric("Number of predicted outliers in this series", number_outlier)
        cols[1].metric("Number of non-NaN entries before", len(df_before[~df_before[s].isna()]))
        cols[2].metric("Number of non-NaN entries after", len(df_new[~df_new[s].isna()]))

        if not number_outlier:
            continue

        for i in range(number_to_show):
            int_idx = df_before.index.get_loc(changes_sample[i])
            start_idx = max(0, int_idx - 20)
            end_idx = min(len(df_before) - 1, int_idx + 20)
            start_time = df_before.index[start_idx]
            end_time = df_before.index[end_idx]
            df_plot_before = df_before[df_before.index.to_series().between(start_time, end_time)][
                [s]
            ].dropna()
            df_plot_new = df_new[df_new.index.to_series().between(start_time, end_time)][
                [s]
            ].dropna()

            plot = (
                Line()
                .add_xaxis(df_plot_before.index.tolist())
                .add_yaxis(
                    "Before",
                    df_plot_before[s].tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    is_symbol_show=False,
                )
            )
            plot = plot.overlap(
                Line()
                .add_xaxis(df_plot_new.index.tolist())
                .add_yaxis(
                    "After",
                    df_plot_new[s].tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    is_symbol_show=False,
                )
            )

            plot.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"Sample # {i+1}",
                    padding=15,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    name=s,
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
                datazoom_opts=opts.DataZoomOpts(type_="inside", range_start=30, range_end=70),
                legend_opts=opts.LegendOpts(pos_top=40, pos_right=10, orient="vertical"),
                tooltip_opts=opts.TooltipOpts(axis_pointer_type="line", trigger="axis"),
            )

            plot.set_colors(["Yellow", "Green"])

            with cols[i]:
                st_pyecharts(plot, theme="dark")
