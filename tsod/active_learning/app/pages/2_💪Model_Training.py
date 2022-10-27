from contextlib import nullcontext

import streamlit as st
from streamlit_profiler import Profiler
from tsod.active_learning.app.components import (
    dev_options,
    show_feature_importances,
    show_info,
    test_metrics,
    train_options,
)
from tsod.active_learning.utils import init_session_state


def main():
    st.set_page_config(layout="wide", page_icon="💪", page_title="Model Training")
    init_session_state()
    profile = dev_options(st.sidebar)

    with Profiler() if profile else nullcontext():
        show_info()
        c1, c2, c3 = st.columns(3)
        train_options()
        test_metrics()
        show_feature_importances()


if __name__ == "__main__":
    main()
