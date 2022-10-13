import streamlit as st

from tsod.active_learning.data_structures import AnnotationState


SELECT_OPTIONS = {"Point": "zoom", "Box": "select", "Lasso": "lasso"}
SELECT_INFO = {
    "Point": "Select individual points, drag mouse to zoom in.",
    "Box": "Draw Box to select all points in range.",
    "Lasso": "Draw Lasso to select all points in range.",
}


def get_as() -> AnnotationState:
    return st.session_state.AS
