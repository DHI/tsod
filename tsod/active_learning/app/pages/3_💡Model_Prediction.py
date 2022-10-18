import streamlit as st
from tsod.active_learning.plotting import (
    make_time_range_outlier_plot,
    make_outlier_distribution_plot,
)
from tsod.active_learning.utils import init_session_state
from tsod.active_learning.app.components import (
    prediction_options,
    dev_options,
    prediction_summary_table,
)
from contextlib import nullcontext
from streamlit_profiler import Profiler


def main():
    st.set_page_config(layout="wide", page_icon="💡", page_title="Model Prediction")
    init_session_state()
    st.sidebar.title("Prediction Controls")
    profile = dev_options(st.sidebar)

    with Profiler() if profile else nullcontext():
        prediction_options(st.sidebar)
        for dataset_name in st.session_state["inference_results"].keys():
            with st.expander("Prediction Summary", expanded=True):
                prediction_summary_table(dataset_name)
            start_time, end_time = make_outlier_distribution_plot(dataset_name)
            if start_time:
                make_time_range_outlier_plot(dataset_name, start_time, end_time)


if __name__ == "__main__":
    main()
