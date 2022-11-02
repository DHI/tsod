import streamlit as st
import mikeio
import pandas as pd
from pathlib import Path
from dateutil.parser._parser import ParserError
import dateutil
import numpy as np


def datetime_unififer(data=pd.DataFrame, date_column=str, base_obj=None):
    obj = base_obj or st
    try:
        data[date_column] = pd.to_datetime(data[date_column], unit="ns", utc=True)
        return data
    except ParserError as e:
        pass
    try:
        data[date_column] = pd.to_datetime(
            data[date_column].str.replace("*", " "), unit="ns", utc=True
        )
        return data
    except ParserError:
        pass
    try:
        data[date_column] = pd.to_datetime(
            data[date_column].str.replace("'T'", " "), unit="ns", utc=True
        )
        return data
    except ParserError:
        pass
    try:
        data[date_column] = pd.to_datetime(
            data[date_column],
            format="%d/%b/%Y:%H:%M:%S %z",
            exact=False,
            utc=True,
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
            data[date_column]
            .str.replace("*", " ")
            .apply(dateutil.parser.parse, tzinfos=tzmapping)
        )
        return data
    except ParserError:
        pass
    try:
        data[date_column] = pd.to_datetime(
            data[date_column],
            format="%Y-%m-%d*%H:%M:%S:%f",
            exact=False,
            utc=True,
        )
        return data
    except ParserError:
        obj.error("Data does not contain recognized timestamp column.")
    # except Exception as e:
    # obj.error("Data does not contain recognized timestamp column.")


def data_upload_parser(base_obj=None):
    obj = base_obj or st
    datafiles = obj.file_uploader(
        label="test",
        accept_multiple_files=True,
        type=["csv", "dfs0", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    if not datafiles:
        obj.info("No file uploaded.")
    else:
        dataframe = pd.DataFrame()
        for file in datafiles:
            if (
                file.type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
                        df[[d for d in df.columns if d not in clm]].index.intersection(
                            dataframe.index
                        )
                    ]
                    .shape[1]
                    > 0
                ):
                    # concat by column, if there exist additional columns in one dataframe with the same index
                    dataframe = pd.concat(
                        [
                            dataframe,
                            df[[d for d in df.columns if d not in clm]].loc[
                                df[
                                    [d for d in df.columns if d not in clm]
                                ].index.intersection(dataframe.index)
                            ],
                        ],
                        axis=1,
                    )

                # concat dataframes with unique indexes by row
                dataframe = pd.concat(
                    [dataframe, df.loc[df.index.difference(dataframe.index)]]
                )
                obj.info(f"File {file.name} uploaded and concatenated.")
        return dataframe