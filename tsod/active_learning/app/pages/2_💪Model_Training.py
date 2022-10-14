import streamlit as st

from tsod.active_learning.components import show_info, train_options
from tsod.active_learning.utils import get_as, init_session_state
from contextlib import nullcontext
from streamlit_profiler import Profiler


def main():
    st.set_page_config(layout="wide", page_icon="💪", page_title="Model Training")
    init_session_state()
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
        col_1, col_2 = st.columns([1, 2])
        show_info(col_1)
        train_options(col_1)


if __name__ == "__main__":
    main()
