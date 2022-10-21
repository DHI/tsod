import streamlit as st

from tsod.active_learning.app.components import (
    show_info,
    train_options,
    dev_options,
    test_metrics,
    show_feature_importances,
)
from tsod.active_learning.utils import get_as, init_session_state
from contextlib import nullcontext
from streamlit_profiler import Profiler


def main():
    st.set_page_config(layout="wide", page_icon="💪", page_title="Model Training")
    init_session_state()
    profile = dev_options(st.sidebar)

    with Profiler() if profile else nullcontext():
        show_info()
        st.sidebar.button("Train Random Forest Model", key="train_button")

        c1, c2, c3 = st.columns(3)
        train_options()
        test_metrics()
        show_feature_importances()


if __name__ == "__main__":
    main()
