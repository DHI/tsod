import streamlit as st
from tsod.active_learning.plotting import make_prediction_plot_for_dataset
from tsod.active_learning.utils import get_as, init_session_state
from tsod.active_learning.components import prediction_options
from contextlib import nullcontext
from streamlit_profiler import Profiler


def main():
    st.set_page_config(layout="wide", page_icon="💡", page_title="Model Prediction")
    init_session_state()
    st.sidebar.title("Prediction Controls")
    #############################################################
    dev_options = st.sidebar.checkbox("Show Dev Options")
    if dev_options:
        st.sidebar.subheader("Development")
        dev_col_1, dev_col_2 = st.sidebar.columns(2)
        profile = dev_col_1.checkbox("Profile Code", value=False)
        show_ss = dev_col_2.button("Show Session State")
        show_as = dev_col_2.button("Show Annotation State")
        st.sidebar.markdown("***")
        if show_ss:
            st.write(st.session_state)
        if show_as:
            st.write(get_as().data)
    else:
        profile = False
    #############################################################

    with Profiler() if profile else nullcontext():
        prediction_options(st.sidebar)

        # if st.session_state["in_prediction_mode"]:
        for dataset_name in st.session_state["inference_results"].keys():
            make_prediction_plot_for_dataset(dataset_name)


if __name__ == "__main__":
    main()
