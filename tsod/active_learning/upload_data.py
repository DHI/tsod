import logging
from pathlib import Path

import dateutil
import mikeio
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.parser._parser import ParserError

from tsod.active_learning.data_structures import AnnotationState


def datetime_unififer(data=pd.DataFrame, date_column=str, base_obj=None):
    obj = base_obj or st
    try:
        data[date_column] = pd.to_datetime(data[date_column], unit="ns", utc=False)
        return data
    except ParserError as e:
        pass
    try:
        data[date_column] = pd.to_datetime(
            data[date_column].str.replace("*", " "), unit="ns", utc=False
        )
        return data
    except ParserError:
        pass
    try:
        data[date_column] = pd.to_datetime(
            data[date_column].str.replace("'T'", " "), unit="ns", utc=False
        )
        return data
    except ParserError:
        pass
    try:
        data[date_column] = pd.to_datetime(
            data[date_column],
            format="%d/%b/%Y:%H:%M:%S %z",
            exact=False,
            utc=False,
        )
        return data
    except ParserError:
        pass
    try:
        tzmapping = {
            "CET": dateutil.tz.gettz("Europe/Berlin"),
            "CEST": dateutil.tz.gettz("Europe/Berlin"),
            "PDT": dateutil.tz.gettz("US/Pacific"),
        }
        data[date_column] = (
            data[date_column].str.replace("*", " ").apply(dateutil.parser.parse, tzinfos=tzmapping)
        )
        return data
    except ParserError:
        pass
    try:
        data[date_column] = pd.to_datetime(
            data[date_column],
            format="%Y-%m-%d*%H:%M:%S:%f",
            exact=False,
            utc=False,
        )
        return data
    except ParserError:
        obj.error("Data does not contain recognized timestamp column.")
    # except Exception as e:
    # obj.error("Data does not contain recognized timestamp column.")


def data_upload_callback_old(base_obj=None):
    obj = base_obj or st
    datafiles = st.session_state["data_upload"]
    if not datafiles:
        return
    dataframe = pd.DataFrame()
    st.write(datafiles)
    file_handle = "_".join(sorted([Path(f.name).stem for f in datafiles]))
    for file in datafiles:
        if (
            file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            or file.type == "application/vnd.ms-excel"
        ):
            # read the excel file
            df = pd.read_excel(file)
            # check which column contains the datetime
            if len(df.select_dtypes(include=["datetime"]).columns) != 0:
                date_column = df.select_dtypes(include=["datetime"]).columns[0]
            elif len(df.select_dtypes(include=["object"]).columns) != 0:
                date_column = df.select_dtypes(include=["object"]).columns[0]
            else:
                obj.error("Data does not contain recognized timestamp column.")

            df = datetime_unififer(df, date_column, obj)
            # set datetime as index
            df = df.set_index(date_column)
            # append the list of uploaded dataframes
        elif file.type == "text/csv":
            df = pd.read_csv(file)
            # check which column contains the datetime (object)
            if len(df.select_dtypes(include=["datetime"]).columns) != 0:
                date_column = df.select_dtypes(include=["datetime"]).columns[0]
            elif len(df.select_dtypes(include=["object"]).columns) != 0:
                date_column = df.select_dtypes(include=["object"]).columns[0]
            else:
                obj.error("Data does not contain recognized timestamp column.")
            # standardize the datetime object
            df = datetime_unififer(df, date_column)
            # set datetime as index
            df = df.set_index(date_column)
        elif file.name.endswith(".dfs0"):
            # elif file.type == "application/octet-stream":
            data = file.getvalue()
            TEMP_FILE = Path("test.dfs0")

            with TEMP_FILE.open("wb") as f:
                f.write(data)

            data = mikeio.read("test.dfs0")
            df = data.to_dataframe()
            TEMP_FILE.unlink()

            df.index = df.index.tz_localize(tz="UTC")

        else:
            obj.error("This datatype is not supported.")

        # check intersecting rows in dataframes
        idx = dataframe.index.intersection(df.index)
        clm = dataframe.columns.intersection(df.columns)
        df_diff = dataframe[clm].eq(df[clm]).loc[idx]
        df_null = pd.notnull(dataframe[clm].loc[idx])

        # st.write(dataframe[dataframe.index.isin(idx)].dtypes)
        # st.write(df[df.index.isin(idx)].dtypes)

        # st.write(len(idx))

        # check if values differ and are not none
        if False in df_diff.values and False in df_diff.eq(df_null).values:
            obj.error(
                f"File {file.name} contains different values for the same timestamps and cannot be integrated for training the model."
            )
        # fill none values with real data if applicable
        elif (
            len(df.index.difference(dataframe.index)) == 0
            and len(df.columns.difference(dataframe.columns)) == 0
            and True in pd.isnull(dataframe.loc[idx]).values
        ):
            dataframe.fillna(df, inplace=True)
        # if indexes/columns are new, concat the dataframe with the new values
        else:
            if (
                df[[d for d in df.columns if d not in clm]]
                .loc[
                    df[[d for d in df.columns if d not in clm]].index.intersection(dataframe.index)
                ]
                .shape[1]
                > 0
            ):
                # concat by column, if there exist additional columns in one dataframe with the same index
                dataframe = pd.concat(
                    [
                        dataframe,
                        df[[d for d in df.columns if d not in clm]].loc[
                            df[[d for d in df.columns if d not in clm]].index.intersection(
                                dataframe.index
                            )
                        ],
                    ],
                    axis=1,
                )

            # concat dataframes with unique indexes by row
            dataframe = pd.concat([dataframe, df.loc[df.index.difference(dataframe.index)]])
            obj.info(f"File {file.name} uploaded and concatenated.")
    return dataframe


def data_uploader(base_obj=None):
    obj = base_obj or st
    form = obj.form("data_upload_form", clear_on_submit=True)
    form.file_uploader(
        label="Upload data from disk",
        accept_multiple_files=True,
        type=["csv", "dfs0", "xlsx", "xls"],
        key="data_upload",
        # on_change=data_upload_callback,
        args=(obj,),
    )

    form.text_input(
        "Optional: Enter dataset name",
        max_chars=30,
        key="data_name",
        help="""This name will be used for data selection and when downloading annotation data.  
        If left empty, the dataset will be named based on the uploaded file names.""",
    )

    form.form_submit_button("Upload", on_click=data_upload_callback, args=(obj,))

    # data_upload_callback(obj)


def data_upload_callback(base_obj=None):
    obj = base_obj or st
    datafiles = st.session_state["data_upload"]
    if not datafiles:
        obj.warning("No files selected.")
        return
    # st.write(datafiles)
    dataframe = pd.DataFrame()
    unique_columns = set()
    if st.session_state["data_name"] != "":
        file_handle = st.session_state["data_name"]
    else:
        file_handle = "_".join(sorted([Path(f.name).stem for f in datafiles]))
    for i, file in enumerate(datafiles):
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif Path(file.name).suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(file)
        elif file.name.endswith(".dfs0"):
            data = file.getvalue()
            TEMP_FILE = Path("temp.dfs0")

            with TEMP_FILE.open("wb") as f:
                f.write(data)

            data = mikeio.read("temp.dfs0")
            df = data.to_dataframe().reset_index()
            TEMP_FILE.unlink()

            # df.index = df.index.tz_localize(tz="UTC")

        if len(df.select_dtypes(include=["datetime"]).columns) != 0:
            date_column = df.select_dtypes(include=["datetime"]).columns[0]
        elif len(df.select_dtypes(include=["object"]).columns) != 0:
            date_column = df.select_dtypes(include=["object"]).columns[0]
        else:
            obj.error(f"Data in file {file.name} does not contain recognized timestamp column.")
            return

        df.rename(columns={date_column: "index_to_be"}, inplace=True)
        df.drop_duplicates(inplace=True)

        df = datetime_unififer(df, "index_to_be")

        unique_columns.update({c for c in df.drop(columns="index_to_be").columns})

        if dataframe.empty:
            dataframe = df
            continue

        dataframe = dataframe.merge(df, on="index_to_be", how="outer", suffixes=["", i])

        # after merging a new df, we validate the added data
        # for each unique target colum
        for column in unique_columns:
            # get all series that have been merged for that column
            cols = sorted([c for c in dataframe.columns if c.startswith(column)])

            # compare the newest entry to all previous ones
            newest_column = cols[-1]
            newest_mask = ~dataframe[newest_column].isna()
            for c in cols[:-1]:
                # get mask where other series is not null
                compare_mask = ~dataframe[c].isna()

                # need to compare values for all rows where both columns have entries
                to_compare = dataframe.loc[newest_mask & compare_mask]

                conflict_mask = ~np.isclose(to_compare[c], to_compare[newest_column])
                if conflict_mask.any():
                    conflict_data = to_compare.loc[conflict_mask].iloc[0]
                    obj.error(
                        f"""Found conflicting data for series '{column}' in uploaded files (multiple values
                    for same timestamp).  
                    First mismatch:  
                    Timestamp: {conflict_data["index_to_be"]}  
                    Value 1: {conflict_data[newest_column].item()}  
                    Value 2: {conflict_data[c].item()}"""
                    )

                    return

    # combine all merged series into single one
    for column in unique_columns:
        cols = sorted([c for c in dataframe.columns if c.startswith(column)])[1:]

        for c in cols:
            dataframe[column].fillna(dataframe[c], inplace=True)

    dataframe = dataframe.set_index("index_to_be", drop=True)[list(unique_columns)]
    dataframe.index.name = None
    if len(dataframe.columns) > 1:
        st.session_state["expand_data_selection"] = True

    if len(st.session_state["data_store"]) > 1:
        st.session_state["expand_data_selection"] = True

    obj.success("Data uploaded and validated", icon="✅")
    obj.write(f"Total rows: {len(dataframe)}")
    obj.write("NaN values:")
    obj.write(dataframe.isna().sum())

    add_new_data(dataframe, file_handle)

    # For deployment 'logging'
    logging.info("A new dataset was successfully uploaded.")


def add_new_data(df: pd.DataFrame, dataset_name: str):
    """Splits a dataframe into its individual series, adds those series
    to the data store and instantiates new AnnotationState instances for them."""

    for col in df:
        sub_df = df[[col]]
        sub_df = sub_df[~sub_df[col].isna()]
        st.session_state["data_store"][dataset_name][col] = sub_df
        an_st = AnnotationState(dataset_name, col)
        st.session_state["annotation_state_store"][dataset_name][col] = an_st

    st.session_state["current_dataset"] = dataset_name
    st.session_state["current_series"][dataset_name] = sorted(df.columns)[0]
