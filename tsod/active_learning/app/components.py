import datetime
import pickle
from typing import Dict, List

import pandas as pd
import streamlit as st
from tsod.active_learning.modelling import (
    construct_training_data_RF,
    train_random_forest_classifier,
)
from tsod.active_learning.utils import (
    get_as,
    set_session_state_items,
    custom_text,
    recursive_length_count,
    MODEL_OPTIONS,
)
from tsod.active_learning.modelling import get_model_predictions, construct_test_data_RF
from tsod.active_learning.plotting import feature_importance_plot
from tsod.active_learning.data_structures import plot_return_value_as_datetime


def filter_data(base_obj=None) -> pd.DataFrame:
    obj = base_obj or st
    obj.subheader("Time range selection")

    data = st.session_state["df_full"]
    dates = obj.date_input(
        "Graph Date Range",
        value=(st.session_state.plot_start_date, st.session_state.plot_end_date),
        min_value=data.index.min(),
        max_value=data.index.max(),
        on_change=set_session_state_items,
        args=(["use_date_picker", "date_shift_buttons_used"], [True, False]),
    )
    if not st.session_state.date_shift_buttons_used:
        if len(dates) == 2:
            start_date, end_date = dates
            st.session_state.plot_start_date = start_date
            st.session_state.plot_end_date = end_date
        else:
            start_date = dates[0]
            end_date = st.session_state.plot_end_date
    else:
        start_date = st.session_state.plot_start_date
        end_date = st.session_state.plot_end_date
    inp_col_1, inp_col_2 = obj.columns(2)

    start_time = inp_col_1.time_input("Start time", value=datetime.time(0, 0, 0))
    end_time = inp_col_2.time_input("End time", value=datetime.time(23, 59, 59))

    state = get_as()

    if st.session_state.use_date_picker:
        inp_col_1.button(
            "Show All", on_click=set_session_state_items, args=("use_date_picker", False)
        )
        c1, c2, c3, c4 = obj.columns(4)
        c1.button("- 1d", on_click=shift_plot_window, args=(-1,))
        c2.button("+ 1d", on_click=shift_plot_window, args=(1,))
        c3.button("- 1w", on_click=shift_plot_window, args=(-7,))
        c4.button("+ 1w", on_click=shift_plot_window, args=(7,))
        start_dt = datetime.datetime.combine(start_date, start_time)
        end_dt = datetime.datetime.combine(end_date, end_time)
    else:
        start_dt = state.df.index.min()
        end_dt = state.df.index.max()
        obj.button(
            "Back to previous selection",
            on_click=set_session_state_items,
            args=("use_date_picker", True),
        )

    obj.markdown("***")

    state.update_plot(start_dt, end_dt)


def shift_plot_window(days: int):
    current_start = st.session_state.plot_start_date
    current_end = st.session_state.plot_end_date

    current_range = current_end - current_start
    df = st.session_state.df_full

    new_start = current_start + datetime.timedelta(days)
    new_end = current_end + datetime.timedelta(days)

    if new_start < df.index.min():
        new_start = df.index.min()
        new_end = new_start + current_range
    if new_end > df.index.max():
        new_end = df.index.max()
        new_start = new_end - current_range

    set_session_state_items("plot_start_date", new_start)
    set_session_state_items("plot_end_date", new_end)
    set_session_state_items("date_shift_buttons_used", True)


def create_annotation_plot_buttons(base_obj=None):
    obj = base_obj or st

    state = get_as()

    obj.subheader("Actions")

    custom_text("Training Data", 15, True, base_obj=obj)
    c_1, c_2 = obj.columns(2)
    c_1.button("Mark selection Outlier", on_click=state.update_data, args=("outlier",))
    c_2.button("Mark selection Normal", on_click=state.update_data, args=("normal",))
    custom_text("Test Data", 15, True, base_obj=obj)
    c_1, c_2 = obj.columns(2)
    c_1.button(
        "Mark selection Outlier",
        on_click=state.update_data,
        args=("test_outlier",),
        key="mark_test_outlier",
    )
    c_2.button(
        "Mark selection Normal",
        on_click=state.update_data,
        args=("test_normal",),
        key="mark_test_normal",
    )
    c_1.button("Clear Selection", on_click=state.clear_selection)
    c_2.button("Clear All", on_click=state.clear_all)

    obj.markdown("***")


def validate_uploaded_file_contents(base_obj=None):
    obj = base_obj or st
    uploaded_files = st.session_state["uploaded_annotation_files"]
    if not uploaded_files:
        return
    for uploaded_file in uploaded_files:
        file_failed = False
        data = pickle.loads(uploaded_file.getvalue())
        state = get_as()
        for df in data.values():
            if df.empty or file_failed:
                continue

            values_match = (
                df["Water Level"].astype(float).isin(state.df["Water Level"].values).all()
            )
            index_match = df.index.isin(state.df.index).all()

            if (not values_match) or (not index_match):
                file_failed = True
        if file_failed:
            obj.error(f"{uploaded_file.name}: Did not pass validation for loaded dataset.")
        else:
            st.session_state.uploaded_annotation_data[uploaded_file.name] = data
            obj.success(f"{uploaded_file.name}: Loaded.", icon="✅")


def annotation_file_upload_callback(base_obj=None):
    obj = base_obj or st
    validate_uploaded_file_contents(obj)
    state = get_as()

    for data in st.session_state.uploaded_annotation_data.values():
        for key, df in data.items():
            state.update_data(key, df.index.to_list())


def dev_options(base_obj=None):
    obj = base_obj or st
    with obj.expander("Dev Options"):
        dev_col_1, dev_col_2 = st.columns(2)
        profile = dev_col_1.checkbox("Profile Code", value=False)
        search_str = dev_col_1.text_input("Search SS", max_chars=25, value="")
        show_ss = dev_col_2.button("Show Session State")
        show_as = dev_col_2.button("Show Annotation State")
    if len(search_str) > 1:
        matches = [k for k in st.session_state.keys() if search_str.lower() in k.lower()]
        df_matches = [m for m in matches if isinstance(st.session_state[m], pd.DataFrame)]
        non_df_matches = [m for m in matches if m not in df_matches]
        for m in df_matches:
            st.write(m)
            st.table(st.session_state[m].head())
        if non_df_matches:
            st.write({k: st.session_state[k] for k in non_df_matches})

        # st.write({k: st.session_state[k] for k in ss_options if search_str.lower() in k.lower()})
    if show_ss:
        st.write(st.session_state)
    if show_as:
        st.write(get_as().__dict__)

    return profile


def create_save_load_buttons(base_obj=None):
    obj = base_obj or st
    obj.subheader("Save / load previous")
    c_1, c_2 = obj.columns(2)
    state = get_as()

    file_name = f"Annotations_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.bin"

    if state._download_data:
        c_1.download_button("Download Annotations", state.download_data, file_name=file_name)

    c_2.file_uploader(
        "Upload Annotations",
        key="uploaded_annotation_files",
        type="bin",
        on_change=annotation_file_upload_callback,
        args=(obj,),
        accept_multiple_files=True,
    )
    obj.markdown("***")


def show_info(base_obj=None):
    obj = base_obj or st
    with obj.expander("Annotation Info", expanded=True):
        state = get_as()
        st.subheader("Annotation summary")
        c1, c2, c3 = st.columns([1, 1, 2])

        custom_text("Training Data", 15, base_obj=c1, centered=False)
        custom_text("Test Data", 15, base_obj=c2, centered=False)

        c1.metric("Total labelled Outlier", len(state.outlier))
        c1.metric("Total labelled Normal", len(state.normal))
        c2.metric("Total labelled Outlier", len(state.test_outlier))
        c2.metric("Total labelled Normal", len(state.test_normal))

    if "classifier" in st.session_state:
        obj.subheader(f"Current model: {st.session_state.last_model_name}")


def train_options(base_obj=None):
    obj = base_obj or st

    with st.sidebar.expander("Modelling Options", expanded=True):
        st.info("Here you can choose what type of outlier detection approach to use.")
        method = st.selectbox(
            "Choose OD method",
            options=list(MODEL_OPTIONS.keys()),
            key="current_method_choice",
            format_func=lambda x: MODEL_OPTIONS.get(x),
        )

    with st.sidebar.expander("Feature Options", expanded=True):
        st.info(
            "Here you can choose how many points before and after each annotated point \
            to include in its feature set."
        )
        st.number_input(
            "Points before",
            min_value=1,
            value=st.session_state.get("old_points_before")
            if st.session_state.get("old_points_before") is not None
            else 10,
            key="number_points_before",
            step=5,
        )
        st.number_input(
            "Points after",
            min_value=0,
            value=st.session_state.get("old_points_after")
            if st.session_state.get("old_points_after") is not None
            else 10,
            key="number_points_after",
            step=5,
        )

    st.sidebar.button("Train Random Forest Model", key="train_button")

    if st.session_state.train_button:
        st.session_state["old_points_before"] = st.session_state["number_points_before"]
        st.session_state["old_points_after"] = st.session_state["number_points_after"]

        with st.spinner("Constructing features..."):
            if method == "RF_1":
                construct_training_data_RF()
                construct_test_data_RF()
        with st.spinner("Training Model..."):
            if method == "RF_1":
                train_random_forest_classifier(obj)

    if "last_model_name" in st.session_state:
        st.sidebar.download_button(
            "Download model",
            pickle.dumps(st.session_state["model_library"][st.session_state["last_model_name"]]),
            f"{st.session_state.last_model_name}.pkl",
        )


def get_predictions_callback(obj=None):
    # set_session_state_items("hide_choice_menus", True)
    get_model_predictions(obj)
    set_session_state_items("prediction_models", {})


def add_annotation_to_pred_data(base_obj=None):
    obj = base_obj or st

    if not st.session_state.annotation_data_loaded:
        obj.error("No annotation data has been loaded in this session.")
        return

    if "Annotation Data" not in st.session_state.prediction_data:
        st.session_state.prediction_data["Annotation Data"] = get_as().df.copy(deep=True)


# def add_most_recent_model(base_obj=None):
#     obj = base_obj or st
#     if not st.session_state.last_model_name:
#         obj.error("No model has been trained yet this session.")
#         return
#     model_name = st.session_state.last_model_name
#     st.session_state.prediction_models[model_name] = st.session_state.classifier


def add_uploaded_models(base_obj=None):
    obj = base_obj or st
    # set_session_state_items("prediction_models", {})
    if not st.session_state.current_uploaded_models:
        return
    for data in st.session_state.current_uploaded_models:
        model_data = pickle.loads(data.read())
        model = model_data["model"]
        if not hasattr(model, "predict"):
            obj.error(
                "The uploaded object can not be used for prediction (does not implement 'predict' method)."
            )
            continue

        st.session_state["prediction_models"][data.name] = model_data
        st.session_state["model_library"][data.name] = model_data


def add_uploaded_dataset(base_obj=None):
    obj = base_obj or st

    if not st.session_state["uploaded_datasets"]:
        return

    for data in st.session_state.uploaded_datasets:
        try:
            df = pd.read_csv("data/Elev_NW1.csv", index_col=0, parse_dates=True)
            st.session_state["prediction_data"][data.name] = df
        except Exception:
            obj.error(f"Could not read file {data.name}.")


def prediction_options(base_obj=None):
    obj = base_obj or st
    _, c, _ = obj.columns([2, 5, 2])
    c.button("Generate Predictions", on_click=get_predictions_callback, args=(obj,))

    with obj.expander("Model Choice", expanded=True):
        # with obj.expander("Model Choice", expanded=not st.session_state["hide_choice_menus"]):
        st.subheader("Choose Models")
        st.info(
            "Add models with which to generate predictions. The most recently trained model is automatically added."
        )
        # c1, c2 = st.columns(2)
        # c1.button("Add most recently trained model", on_click=add_most_recent_model, args=(obj,))
        st.file_uploader(
            "Select model from disk",
            type="pkl",
            on_change=add_uploaded_models,
            key="current_uploaded_models",
            args=(obj,),
            accept_multiple_files=True,
        )

        st.subheader("Selected models:")
        st.json(list(st.session_state.prediction_models.keys()))
        if st.session_state.prediction_models:
            st.button(
                "Clear selection", on_click=set_session_state_items, args=("prediction_models", {})
            )
    with obj.expander("Data Choice", expanded=True):
        # with obj.expander("Data Choice", expanded=not st.session_state["hide_choice_menus"]):
        st.subheader("Select Data")
        st.info("Add datasets for outlier evaluation.")
        c1, c2 = st.columns(2)
        c1.button("Add Annotation Data", on_click=add_annotation_to_pred_data)
        add_annotation_to_pred_data()
        c2.file_uploader(
            "Select file from disk",
            accept_multiple_files=True,
            key="uploaded_datasets",
            on_change=add_uploaded_dataset,
        )
        st.subheader("Selected Files:")
        st.json(list(st.session_state.prediction_data.keys()))
        if st.session_state.prediction_data:
            st.button(
                "Clear selection",
                on_click=set_session_state_items,
                args=("prediction_data", {}),
                key="data_clear",
            )
    _, c, _ = obj.columns([2, 5, 2])
    c.button(
        "Generate Predictions", on_click=get_predictions_callback, args=(obj,), key="pred_btn_2"
    )


def remove_model_to_visualize(dataset_name, model_name):
    st.session_state["models_to_visualize"][dataset_name].discard(model_name)

    # if not recursive_length_count(st.session_state["models_to_visualize"]):
    # st.session_state["hide_choice_menus"] = False


def prediction_summary_table(dataset_name: str, base_obj=None):
    obj = base_obj or st

    DEFAULT_MARKER_COLORS = ["#e88b0b", "#1778dc", "#1bd476", "#d311e6"]
    model_predictions: pd.DataFrame() = st.session_state["inference_results"].get(dataset_name)
    if model_predictions is None:
        return
    model_names = sorted(st.session_state["models_to_visualize"][dataset_name])

    if not model_names:
        return

    if len(model_names) > len(DEFAULT_MARKER_COLORS):
        obj.error(
            f"Currently max. number of models is {len(DEFAULT_MARKER_COLORS)}, got {len(model_names)}"
        )
        return

    c1, c2, c4, c5, c6, c7 = obj.columns([5, 3, 6, 3, 3, 3])
    custom_text("Model Name", base_obj=c1, font_size=15, centered=True)
    custom_text("Params", base_obj=c4, font_size=15, centered=False)
    custom_text("Predicted Outliers", base_obj=c5, font_size=15, centered=False)
    custom_text("Predicted Normal", base_obj=c6, font_size=15, centered=False)
    custom_text("Choose Plot Color", base_obj=c7, font_size=15, centered=False)

    obj.markdown("***")

    for i, model in enumerate(model_names):
        _local_obj = obj.container()
        c1, c2, c4, c5, c6, c7 = _local_obj.columns([5, 3, 6, 3, 3, 3])
        custom_text(model, base_obj=c1, font_size=15, centered=True)
        c2.button(
            "Remove",
            key=f"remove_{model}_{dataset_name}",
            on_click=remove_model_to_visualize,
            args=(dataset_name, model),
        )
        c4.json(st.session_state["model_library"][model]["params"], expanded=False)
        custom_text(
            st.session_state["number_outliers"][dataset_name][model],
            base_obj=c5,
            font_size=20,
            centered=False,
        )
        custom_text(
            len(st.session_state["inference_results"][dataset_name])
            - st.session_state["number_outliers"][dataset_name][model],
            base_obj=c6,
            font_size=20,
            centered=False,
        )
        c7.color_picker(
            model,
            key=f"color_{model}_{dataset_name}",
            label_visibility="collapsed",
            value=DEFAULT_MARKER_COLORS[i],
        )
        obj.markdown("***")


def test_metrics(base_obj=None):
    if "test_features" not in st.session_state:
        return

    obj = base_obj or st
    custom_text(f"Most recent model: {st.session_state['last_model_name']}", base_obj=obj)
    c1, c2, c3, c4 = obj.columns(4)

    current_train_metrics = st.session_state["current_model_train_metrics"]
    prec = current_train_metrics["precision"]
    rec = current_train_metrics["recall"]
    f1 = current_train_metrics["f1"]

    if "previous_model_train_metrics" in st.session_state:
        old_metrics = st.session_state["previous_model_train_metrics"]
        old_prec = old_metrics["precision"]
        old_rec = old_metrics["recall"]
        old_f1 = old_metrics["f1"]

        out_prec_diff = (prec[1] - old_prec[1]).round(3)
        out_rec_diff = (rec[1] - old_rec[1]).round(3)
        out_f1_diff = (f1[1] - old_f1[1]).round(3)
        norm_prec_diff = (prec[0] - old_prec[0]).round(3)
        norm_rec_diff = (rec[0] - old_rec[0]).round(3)
        norm_f1_diff = (f1[0] - old_f1[0]).round(3)

    else:
        out_prec_diff, out_rec_diff, out_f1_diff, norm_prec_diff, norm_rec_diff, norm_f1_diff = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    with c1.expander("Train Set Outlier Metrics", expanded=True):
        st.metric(
            "Precision Score",
            prec[1],
            delta=out_prec_diff,
            delta_color="normal" if out_prec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total predicted positives for the train set outliers. Represents the ability \
                not to classify a normal sample as an outlier.",
        )
        st.metric(
            "Recall Score",
            rec[1],
            delta=out_rec_diff,
            delta_color="normal" if out_rec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total positives for the train set outliers. Represents the ability \
                to correctly predict all the outliers.",
        )
        st.metric(
            "F1 Score",
            f1[1],
            delta=out_f1_diff,
            delta_color="normal" if out_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the train set outliers.",
        )
    with c2.expander("Train Set Normal Metrics", expanded=True):
        st.metric(
            "Precision Score",
            prec[0],
            delta=norm_prec_diff,
            delta_color="normal" if norm_prec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total predicted positives for the train set normal points. Represents the ability \
                not to classify an outlier sample as a normal point.",
        )
        st.metric(
            "Recall Score",
            rec[0],
            delta=norm_rec_diff,
            delta_color="normal" if norm_rec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total positives for the train set normal points. Represents the ability \
                to correctly predict all the normal points.",
        )
        st.metric(
            "F1 Score",
            f1[0],
            delta=norm_f1_diff,
            delta_color="normal" if norm_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the train set normal points.",
        )

    current_metrics = st.session_state["current_model_test_metrics"]
    prec = current_metrics["precision"]
    rec = current_metrics["recall"]
    f1 = current_metrics["f1"]

    if "previous_model_test_metrics" in st.session_state:
        old_metrics = st.session_state["previous_model_test_metrics"]
        old_prec = old_metrics["precision"]
        old_rec = old_metrics["recall"]
        old_f1 = old_metrics["f1"]

        out_prec_diff = (prec[1] - old_prec[1]).round(3)
        out_rec_diff = (rec[1] - old_rec[1]).round(3)
        out_f1_diff = (f1[1] - old_f1[1]).round(3)
        norm_prec_diff = (prec[0] - old_prec[0]).round(3)
        norm_rec_diff = (rec[0] - old_rec[0]).round(3)
        norm_f1_diff = (f1[0] - old_f1[0]).round(3)

    else:
        out_prec_diff, out_rec_diff, out_f1_diff, norm_prec_diff, norm_rec_diff, norm_f1_diff = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    with c3.expander("Test Set Outlier Metrics", expanded=True):
        st.metric(
            "Precision Score",
            prec[1],
            delta=out_prec_diff,
            delta_color="normal" if out_prec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total predicted positives for the test set outliers. Represents the ability \
                not to classify a normal sample as an outlier.",
        )
        st.metric(
            "Recall Score",
            rec[1],
            delta=out_rec_diff,
            delta_color="normal" if out_rec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total positives for the test set outliers. Represents the ability \
                to correctly predict all the outliers.",
        )
        st.metric(
            "F1 Score",
            f1[1],
            delta=out_f1_diff,
            delta_color="normal" if out_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the outliers.",
        )
    with c4.expander("Test Set Normal Metrics", expanded=True):
        st.metric(
            "Precision Score",
            prec[0],
            delta=norm_prec_diff,
            delta_color="normal" if norm_prec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total predicted positives for the test set normal points. Represents the ability \
                not to classify an outlier sample as a normal point.",
        )
        st.metric(
            "Recall Score",
            rec[0],
            delta=norm_rec_diff,
            delta_color="normal" if norm_rec_diff != 0.0 else "off",
            help="The ratio of true predicted positives to total positives for the test set normal points. Represents the ability \
                to correctly predict all the normal points.",
        )
        st.metric(
            "F1 Score",
            f1[0],
            delta=norm_f1_diff,
            delta_color="normal" if norm_f1_diff != 0.0 else "off",
            help="The harmonic mean of the precision and recall for the normal points.",
        )


def model_choice_callback(dataset_name: str):
    st.session_state["models_to_visualize"][dataset_name] = set(
        st.session_state[f"model_choice_{dataset_name}"]
    )


def model_choice_options(dataset_name: str):
    if st.session_state["inference_results"].get(dataset_name) is None:
        return
    st.info(
        f"Here you can choose which of the predictions that exist for the datset '{dataset_name}' so far to visualize."
    )
    st.multiselect(
        "Choose models for dataset",
        sorted(st.session_state["available_models"][dataset_name]),
        key=f"model_choice_{dataset_name}",
        default=sorted(list(st.session_state["models_to_visualize"][dataset_name])),
        on_change=model_choice_callback,
        args=(dataset_name,),
    )

    st.markdown("***")


def outlier_visualization_options(dataset_name: str):
    if st.session_state["inference_results"].get(dataset_name) is None:
        return

    form = st.form(
        f"form_{dataset_name}",
    )
    form.info(
        "A large number of outliers is about to be visualized. Click on a bar in the distribution plot to view all outliers \
        in that time period. Each time period is chosen so it contains the same number of datapoints. That number can be adjusted here."
    )
    form.slider(
        "Number of datapoints per bar",
        value=300,
        min_value=10,
        max_value=1000,
        step=1,
        key=f"num_outliers_{dataset_name}",
    )
    form.slider(
        "Height of figures (px)",
        value=600,
        min_value=100,
        max_value=1500,
        step=100,
        key=f"figure_height_{dataset_name}",
    )

    form.checkbox(
        "Only show time ranges containing outliers (predicted or annotated)",
        key=f"only_show_ranges_with_outliers_{dataset_name}",
    )

    form.form_submit_button("Update Distribution Plot")


def show_feature_importances(base_obj=None):
    obj = base_obj or st
    if "last_model_name" not in st.session_state:
        return

    st.sidebar.success(f"{st.session_state.last_model_name} finished training.")

    with obj.expander("Feature Importances", expanded=True):
        c1, c2 = st.columns([2, 1])
        feature_importance_plot(c1)
        c2.dataframe(st.session_state["current_importances"])


def process_point_from_outlier_plot(selected_point: List | Dict, dataset_name: str, base_obj=None):
    obj = base_obj or st
    state = get_as()
    if isinstance(selected_point, list):  # plot point clicked
        point_to_process = plot_return_value_as_datetime(selected_point[0])
    elif isinstance(selected_point, dict):  # marker clicked
        point_to_process = plot_return_value_as_datetime(selected_point["coord"][0])
    else:
        point_to_process = None

    if point_to_process:
        if point_to_process not in state.selection:
            state.update_selected([point_to_process])
        else:
            state.selection.remove(point_to_process)
            st.experimental_rerun()

    if not state.all_indices:
        return
    obj.subheader("Prediction correction:")

    obj.info(
        "Either select individual points in the above plot or using the below slider. \
        Then use the buttons to add annotations."
    )

    c1, c2, c3, c4 = obj.columns(4)
    # c1, c2, c3 = obj.columns([2, 1, 1])
    # c1.json(
    # list(state.selection),
    # expanded=len(state.selection) < 5,
    # )

    c1.button(
        "Mark Train Outlier",
        on_click=state.update_data,
        args=("outlier",),
        key=f"pred_mark_outlier_{dataset_name}",
    )
    c2.button(
        "Mark Train Normal",
        on_click=state.update_data,
        args=("normal",),
        key=f"pred_mark_normal_{dataset_name}",
    )
    c3.button(
        "Mark Test Outlier",
        on_click=state.update_data,
        args=("test_outlier",),
        key=f"pred_mark_test_outlier_{dataset_name}",
    )
    c4.button(
        "Mark Test Normal",
        on_click=state.update_data,
        args=("test_normal",),
        key=f"pred_mark_test_normal_{dataset_name}",
    )
    c1.button(
        "Clear Selection",
        on_click=state.clear_selection,
        key=f"pred_clear_selection_{dataset_name}",
    )
    # c3.button("Clear All", on_click=state.clear_all)
