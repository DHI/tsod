import streamlit as st
from tsod.active_learning.plotting import (
    make_time_range_outlier_plot,
    make_outlier_distribution_plot,
)
from tsod.active_learning.utils import init_session_state
from tsod.active_learning.app.components import (
    prediction_options,
    dev_options,
    prediction_summary_table,
    model_choice_options,
    outlier_visualization_options,
    process_data_from_outlier_plot,
)


def main():
    st.set_page_config(layout="wide", page_icon="💡", page_title="Model Prediction")
    init_session_state()
    st.sidebar.title("Prediction Controls")

    with dev_options(st.sidebar):
        prediction_options(st.sidebar)
        for dataset_name in st.session_state["prediction_data"].keys():
            with st.expander(f"{dataset_name} - Visualization Options", expanded=True):
                st.subheader(dataset_name)
                model_choice_options(dataset_name)
                prediction_summary_table(dataset_name)
                outlier_visualization_options(dataset_name)
            with st.expander(f"{dataset_name} - Graphs", expanded=True):
                start_time, end_time = make_outlier_distribution_plot(dataset_name)
                if start_time is None:
                    if f"last_clicked_range_{dataset_name}" in st.session_state:
                        start_time, end_time = st.session_state[
                            f"last_clicked_range_{dataset_name}"
                        ]
                    else:
                        continue
                clicked_point = make_time_range_outlier_plot(dataset_name, start_time, end_time)
                process_data_from_outlier_plot(clicked_point, dataset_name)


if __name__ == "__main__":
    main()
