import streamlit as st
from tsod.active_learning.components import (
    outlier_annotation,
    model_training,
    model_prediction,
)
from tsod.active_learning.utils import init_session_state
from streamlit_option_menu import option_menu


def main():
    st.set_page_config(layout="wide", page_icon="📈", page_title="Outlier Annotation Tool")
    init_session_state()

    func_mapping = {
        "Outlier Annotation": outlier_annotation,
        "Model Training": model_training,
        "Model Prediction": model_prediction,
    }

    with st.sidebar:
        choice = option_menu(
            "Time Series Outlier Detection",
            list(func_mapping.keys()),
            # orientation="horizontal",
            default_index=st.session_state["page_index"],
            icons=["graph-up", "file-bar-graph", "lightbulb"],
            styles={
                "container": {"padding": "0!important"},
                # "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "15px"},
                "menu-title": {"font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px",
                    # "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
            menu_icon="",
        )

    func_mapping[choice]()


if __name__ == "__main__":
    main()
