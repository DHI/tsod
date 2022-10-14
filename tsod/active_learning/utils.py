import datetime
from typing import Any, List
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from tsod.active_learning.data_structures import AnnotationState


SELECT_OPTIONS = {"Point": "zoom", "Box": "select", "Lasso": "lasso"}
SELECT_INFO = {
    "Point": "Select individual points, drag mouse to zoom in.",
    "Box": "Draw Box to select all points in range.",
    "Lasso": "Draw Lasso to select all points in range.",
}


@st.cache(allow_output_mutation=True)
def load_data(file_name: str = "TODO"):
    df = pd.read_csv("data/Elev_NW1.csv", index_col=0, parse_dates=True)
    df["Water Level"] = df["Water Level"].astype(np.float16)
    # df["ts"] = df.index.values.astype(np.int64) // 10**6
    return df


def get_as() -> AnnotationState:
    return st.session_state.AS


def _add_to_ss_if_not_in_it(key: str, init_value: Any):
    if key not in st.session_state:
        st.session_state[key] = init_value


def init_session_state():
    _add_to_ss_if_not_in_it("df_full", load_data())
    _add_to_ss_if_not_in_it("AS", AnnotationState(st.session_state["df_full"]))
    _add_to_ss_if_not_in_it("annotation_data_loaded", True)
    _add_to_ss_if_not_in_it("uploaded_annotation_data", {})
    _add_to_ss_if_not_in_it("prediction_models", {})
    _add_to_ss_if_not_in_it("prediction_data", {})
    _add_to_ss_if_not_in_it("last_model_name", None)
    _add_to_ss_if_not_in_it("use_date_picker", True)
    _add_to_ss_if_not_in_it("inference_results", defaultdict(dict))
    _add_to_ss_if_not_in_it("uploaded_ds_features", {})
    _add_to_ss_if_not_in_it(
        "plot_start_date",
        st.session_state["df_full"].index.max().date() - datetime.timedelta(days=7),
    )
    _add_to_ss_if_not_in_it("plot_end_date", st.session_state["df_full"].index.max().date())
    _add_to_ss_if_not_in_it("date_shift_buttons_used", False)


def set_session_state_items(key: str | List[str], value: Any | List[Any]):
    if [type(key), type(value)].count(list) == 1:
        raise ValueError("Either both or neither of key and value should be list.")

    if isinstance(key, list):
        assert len(key) == len(value)
        for k, v in zip(key, value):
            st.session_state[k] = v

    else:
        st.session_state[key] = value
