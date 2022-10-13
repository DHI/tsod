from contextlib import nullcontext
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Line
from streamlit_echarts import st_echarts, st_pyecharts
from streamlit_plotly_events import plotly_events
from streamlit_profiler import Profiler
from tsod.active_learning.data_structures import AnnotationState
from tsod.active_learning.components import (
    construct_training_data,
    create_plot_buttons,
    create_save_load_buttons,
    filter_data,
    show_info,
    train_options,
)
from tsod.active_learning.utils import get_as
from tsod.active_learning.plotting import create_annotation_plot


@st.cache(allow_output_mutation=True)
def load_data(file_name: str = "TODO"):
    df = pd.read_csv("data/Elev_NW1.csv", index_col=0, parse_dates=True)
    df["Water Level"] = df["Water Level"].astype(np.float16)
    # df["ts"] = df.index.values.astype(np.int64) // 10**6
    return df


def prepare_session_state():
    if "AS" not in st.session_state:
        st.session_state.AS = AnnotationState(st.session_state["df_full"])

    if "uploaded_annotation_data" not in st.session_state:
        st.session_state.uploaded_annotation_data = {}

    if "current_files_in_AS" not in st.session_state:
        st.session_state.current_files_in_AS = set()

    if "prediction_models" not in st.session_state:
        st.session_state.prediction_models = {}

    if "last_model_name" not in st.session_state:
        st.session_state.last_model_name = None

    if "in_prediction_mode" not in st.session_state:
        st.session_state.in_prediction_mode = False


def add_most_recent_model(base_obj=None):
    obj = base_obj or st
    if not st.session_state.last_model_name:
        obj.error("No model has been trained yet this session.")
        return
    model_name = st.session_state.last_model_name
    st.session_state.prediction_models[model_name] = st.session_state.classifier


def add_uploaded_model(base_obj=None):
    obj = base_obj or st
    if not st.session_state.current_uploaded_model:
        return
    clf = pickle.loads(st.session_state.current_uploaded_model.read())
    if not hasattr(clf, "predict"):
        obj.error(
            "The uploaded object can not be used for prediction (does not implement 'predict' method."
        )
        return

    st.session_state.prediction_models[st.session_state.current_uploaded_model.name] = clf


def clear_model_selection():
    st.session_state.prediction_models = {}


def start_button_click_callback():
    st.session_state["in_prediction_mode"] = True


def back_button_callback():
    st.session_state["in_prediction_mode"] = False


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

        with tab1:
            if not st.session_state["in_prediction_mode"]:
                fig = create_annotation_plot(plot_time_ph)
                selection = plotly_events(fig, select_event=True, override_height=1000)
                selection = {e["x"] for e in selection}
                state.update_selected(selection)

        with tab2:
            col_1, col_2 = st.columns([1, 2])
            show_info(col_1)
            train_options(col_1)

        with tab3:
            tab3_col1, tab3_col2, tab3_col3 = st.columns(3)
            # form = tab3_col1.form("pred_form")
            tab3_col1.subheader("Choose Models")
            tab3_col1.button(
                "Add most recently trained model", on_click=add_most_recent_model, args=(tab3_col1,)
            )
            tab3_col1.file_uploader(
                "Select model from disk",
                on_change=add_uploaded_model,
                key="current_uploaded_model",
                args=(tab3_col1,),
            )

            tab3_col2.subheader("Selected models:")
            tab3_col2.write(list(st.session_state.prediction_models.keys()))
            if st.session_state.prediction_models:
                tab3_col2.button("Clear selection", on_click=clear_model_selection)

            start_button = st.button("Get predictions", on_click=start_button_click_callback)
            back_button = st.button("Back to Annotation Mode", on_click=back_button_callback)

            if st.session_state["in_prediction_mode"]:
                x_data = state.df_plot.index
                y_data = state.df_plot["Water Level"].to_list()

                # st.write(y_data)
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
                            data=[
                                opts.MarkPointItem(
                                    name="Outlier 1 Model 1",
                                    coord=[x_data[300], y_data[300]],
                                    symbol="pin",
                                    itemstyle_opts=opts.ItemStyleOpts(color="red"),
                                )
                            ]
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
                # st_pyecharts(line, theme="dark")
                st_pyecharts(line, height="700px", theme="dark")

    # tooltip_opts = opts.TooltipOpts(axis_pointer_type="cross")


if __name__ == "__main__":
    main()
