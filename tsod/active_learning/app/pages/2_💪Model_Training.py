import streamlit as st

from tsod.active_learning.app.components import show_info, train_options, dev_options
from tsod.active_learning.utils import get_as, init_session_state
from contextlib import nullcontext
from streamlit_profiler import Profiler


def main():
    st.set_page_config(layout="wide", page_icon="💪", page_title="Model Training")
    init_session_state()
    profile = dev_options(st.sidebar)

    with Profiler() if profile else nullcontext():
        col_1, col_2 = st.columns([1, 2])
        show_info(col_1)
        train_options(col_1)


if __name__ == "__main__":
    main()
