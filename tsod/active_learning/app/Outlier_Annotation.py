from contextlib import nullcontext

import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_profiler import Profiler
from tsod.active_learning.components import (
    create_plot_buttons,
    create_save_load_buttons,
    filter_data,
)
from tsod.active_learning.utils import get_as, init_session_state
from tsod.active_learning.plotting import create_annotation_plot


def main():
    st.set_page_config(layout="wide", page_icon="📈", page_title="Outlier Annotation Tool")
    st.sidebar.title("Annotation Controls")

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
