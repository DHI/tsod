import streamlit as st
from tsod.active_learning.plotting import make_prediction_plot_for_dataset
from tsod.active_learning.utils import init_session_state
from tsod.active_learning.app.components import prediction_options, dev_options
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
            make_prediction_plot_for_dataset(dataset_name)


if __name__ == "__main__":
    main()
