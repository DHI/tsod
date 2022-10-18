from collections import defaultdict
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Line, Bar
from streamlit_echarts import st_pyecharts
from tsod.active_learning.utils import SELECT_INFO, SELECT_OPTIONS, get_as


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


def make_outlier_distribution_plot(dataset_name: str, base_obj=None):
    obj = base_obj or st
    dataset: pd.DataFrame = st.session_state["prediction_data"][dataset_name]

    number_of_outliers = sum(
        [n for n in st.session_state["number_outliers"][dataset_name].values()]
    )

    if number_of_outliers < 200:
        return dataset.index.min(), dataset.index.max()

    model_predictions = st.session_state["inference_results"][dataset_name]
    model_names = list(model_predictions.keys())

    form = obj.form(
        f"form_{dataset_name}",
    )
    form.info(
        "A large number of outliers was found. Click on a bar in the distribution plot to view all outliers \
        in that time period. Each time period is chosen so it contains the same number of total outliers. \
        That number can be adjusted here."
    )
    form.slider(
        "Number of total outliers per bar",
        value=20,
        min_value=1,
        max_value=150,
        step=1,
        key=f"num_outliers_{dataset_name}",
    )

    form.form_submit_button("Update Distribution Plot")

    for model_name, model_preds in model_predictions.items():
        dataset[model_name] = model_preds.astype(np.int8)

    dataset["outlier_group"] = (
        dataset[model_names].sum(axis=1).cumsum()
        // st.session_state[f"num_outliers_{dataset_name}"]
    ).astype(np.int16)

    ts = []
    outlier_counts = defaultdict(list)
    for i, (_, group) in enumerate(dataset.groupby("outlier_group")):
        for model in model_names:
            # st.write(f"{model}: adding {group[model].sum().item()}")
            outlier_counts[model].append(group[model].sum().item())
        if i > 0:
            ts.append(group.index[0])

        if i == 20:
            break

    ts.insert(0, dataset.index.min())
    # ts.append(dataset.index.max())

    ranges = [f"{i} - {j}" for i, j in zip(ts, ts[1:])]

    bar = (
        Bar()
        .add_xaxis(ranges)
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
        )
    )
    for i, model in enumerate(model_names):
        bar = bar.add_yaxis(
            model,
            outlier_counts[model],
            stack="x",
            label_opts=opts.LabelOpts(is_show=False),
        )

    bar.set_colors([st.session_state[f"color_{m}"] for m in model_names])

    clicked_range = st_pyecharts(
        bar,
        height="500px",
        theme="dark",
        events={"click": "function(params) { return params.name }"},
    )

    def _get_start_and_end_date(clicked_range: str):
        if not clicked_range:
            return None, None
        start_str, end_str = clicked_range.split(" - ")
        start_time = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        return start_time, end_time

    return _get_start_and_end_date(clicked_range)


def make_time_range_outlier_plot(dataset_name: str, start_time, end_time):
    symbols = ["pin", "arrow", "diamond"]

    dataset: pd.DataFrame = st.session_state["prediction_data"][dataset_name]

    model_predictions = st.session_state["inference_results"][dataset_name]
    model_names = list(model_predictions.keys())

    df_plot = dataset[dataset.index.to_series().between(start_time, end_time)]
    x_data = df_plot.index.to_list()
    y_data = df_plot["Water Level"].to_list()
    markers = []

    for model_number, model_name in enumerate(model_names):

        counter = 1
        for i, row in df_plot.iterrows():
            if row[model_name] == 1:
                markers.append(
                    opts.MarkPointItem(
                        name=f"Outlier {counter} {model_name}",
                        coord=[i, row["Water Level"].item()],
                        symbol=symbols[model_number],
                        itemstyle_opts=opts.ItemStyleOpts(
                            color=st.session_state[f"color_{model_name}"]
                        ),
                        value=counter,
                    )
                )
                counter += 1

    line = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis(
            "Water Level",
            y_data,
            color="yellow",
            label_opts=opts.LabelOpts(is_show=False),
            markpoint_opts=opts.MarkPointOpts(data=markers),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Water Level {start_time} - {end_time}",
                # subtitle="Click on bar to isolate time range",
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name="Water Level",
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
                    range_start=0,
                    range_end=100,
                ),
            ],
            # toolbox_opts=opts.TooltipOpts(axis_pointer_type="cross", trigger="axis"),
        )
    )
    st_pyecharts(line, height="500px", theme="dark")
