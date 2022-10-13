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
