import sys

# quick hack to fix python path on streamlit cloud, I'm sure there's a better way
sys.path.append("/app/tsod")

import streamlit as st
from streamlit_option_menu import option_menu

from tsod.active_learning.components import FUNC_MAPPING, dev_options
from tsod.active_learning.utils import init_session_state


def main():
    st.set_page_config(
        layout="wide",
        page_icon="https://static.thenounproject.com/png/2196104-200.png",
        page_title="Outlier Annotation Tool",
    )
    init_session_state()

    icons = [
        "graph-up",
        "file-bar-graph",
        "lightbulb",
        "question-square",
        "download",
        "info-circle",
    ]

    with st.sidebar:
        choice = option_menu(
            "Time Series Outlier Detection",
            list(FUNC_MAPPING.keys()),
            default_index=st.session_state["page_index"],
            icons=icons,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "orange", "font-size": "15px"},
                "menu-title": {"font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#AFADB4",
                },
                "nav-link-selected": {"background-color": "green"},
            },
            menu_icon="",
        )
    with dev_options(st.sidebar):
        FUNC_MAPPING[choice]()


if __name__ == "__main__":
    main()
