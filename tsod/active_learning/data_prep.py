import streamlit as st
import pandas as pd
import datetime
import numpy as np


def construct_training_data(hours_before_and_after: int = 1):
    outliers: pd.DataFrame = st.session_state["df_marked_out"]
    if outliers.empty:
        return
    normal: pd.DataFrame = st.session_state["df_marked_not_out"]
    df = st.session_state["df_full"]

    features = []
    labels = []

    for i in outliers.index:
        start_time = i - datetime.timedelta(hours=hours_before_and_after)
        end_time = i + datetime.timedelta(hours=hours_before_and_after)
        features.append(df[start_time:end_time]["Water Level"].tolist())
        labels.append(1)

    for i in normal.index:
        start_time = i - datetime.timedelta(hours=hours_before_and_after)
        end_time = i + datetime.timedelta(hours=hours_before_and_after)
        features.append(df[start_time:end_time]["Water Level"].tolist())
        labels.append(0)

    # This check is needed in case we are at the very start/end of a series
    target_length = max([len(l) for l in features])
    for feat in features:
        while len(feat) < target_length:
            feat.append(np.nan)

    features = np.array(features)
    labels = np.array(labels)

    # st.write(features)
    # st.write(labels)
