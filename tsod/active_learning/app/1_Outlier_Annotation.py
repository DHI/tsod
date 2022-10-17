from contextlib import nullcontext

import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_profiler import Profiler
from tsod.active_learning.app.components import (
    create_plot_buttons,
    create_save_load_buttons,
    filter_data,
    dev_options,
)
from tsod.active_learning.utils import get_as, init_session_state
from tsod.active_learning.plotting import create_annotation_plot


def main():
    st.set_page_config(layout="wide", page_icon="📈", page_title="Outlier Annotation Tool")
    st.sidebar.title("Annotation Controls")

    profile = dev_options(st.sidebar)
    with Profiler() if profile else nullcontext():

        init_session_state()
        state = get_as()

        create_plot_buttons(st.sidebar)
        filter_data(st.sidebar)
        plot_time_ph = st.sidebar.container()

        create_save_load_buttons(st.sidebar)

        fig = create_annotation_plot(plot_time_ph)
        selection = plotly_events(fig, select_event=True, override_height=1000)
        selection = {e["x"] for e in selection}
        state.update_selected(selection)


if __name__ == "__main__":
    main()
