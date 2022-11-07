import streamlit as st
from streamlit_plotly_events import plotly_events
from tsod.active_learning.components import (
    create_annotation_plot_buttons,
    create_save_load_buttons,
    time_range_selection,
    dev_options,
    process_data_from_echarts_plot,
    outlier_annotation,
    model_training,
    model_prediction,
)
from tsod.active_learning.utils import get_as, init_session_state
from tsod.active_learning.plotting import get_echarts_plot_time_range
from streamlit_echarts import st_pyecharts
from streamlit_option_menu import option_menu


def main():
    st.set_page_config(layout="wide", page_icon="📈", page_title="Outlier Annotation Tool")
    # st.sidebar.title("Annotation Controls")
    init_session_state()

    func_mapping = {
        "Outlier Annotation": outlier_annotation,
        "Model Training": model_training,
        "Model Prediction": model_prediction,
    }

    choice = option_menu(
        "Time Series Outlier Detection",
        list(func_mapping.keys()),
        orientation="horizontal",
        default_index=st.session_state["page_index"],
    )

    func_mapping[choice]()

    # with dev_options(st.sidebar):

    #     init_session_state()
    #     state = get_as()

    #     create_annotation_plot_buttons(st.sidebar)
    #     filter_data(st.sidebar)
    #     plot_time_ph = st.sidebar.container()

    #     create_save_load_buttons(st.sidebar)

    #     plot = get_echarts_plot_time_range(
    #         state.start,
    #         state.end,
    #         "Water Level",
    #         True,
    #         "Outlier Annotation",
    #     )

    #     clicked_point = st_pyecharts(
    #         plot,
    #         height=1000,
    #         theme="dark",
    #         events={
    #             "click": "function(params) { return [params.data[0], 'click'] }",
    #             "brushselected": "function(params) { return [params.batch[0].selected[1].dataIndex, 'brush'] }",
    #         },
    #     )

    # process_data_from_echarts_plot(clicked_point)


if __name__ == "__main__":
    main()
