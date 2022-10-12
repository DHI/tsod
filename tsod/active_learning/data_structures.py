from collections import defaultdict
import datetime
from typing import Sequence
import pandas as pd
import streamlit as st


class AnnotationState:
    def __init__(self, df_full: pd.DataFrame) -> None:
        self.df = df_full
        self.data = defaultdict(set)
        self.df_selected = pd.DataFrame()
        self.df_outlier = pd.DataFrame()
        self.df_normal = pd.DataFrame()
        self.start = None
        self.end = None

    @classmethod
    def from_other_state(cls, other):
        state = cls(other.df)
        state.__dict__ = other.__dict__

        return state

    @property
    def selection(self):
        return self.data["selected"]

    @property
    def outlier(self):
        return self.data["outlier"]

    @property
    def normal(self):
        return self.data["normal"]

    def update_selected(self, data: Sequence):
        to_add = {self.value_as_datetime(e) for e in set(data)}
        if not to_add.issubset(self.data["selected"]):
            self.data["selected"].update(to_add)
            self.df_selected = self._filter_df("selected")
            st.experimental_rerun()

    def update_outliers(self, data_to_add: Sequence | None = None, base_obj=None):
        """
        Updates outliers either from passed data or from stored selected data.
        """
        obj = base_obj or st

        _data = (
            {self.value_as_datetime(e) for e in set(data_to_add)} if data_to_add else self.selection
        )

        if _data.intersection(self.data["normal"]):
            obj.warning(
                "Some of the selected points have already been marked as normal and were overwritten."
            )
            self.data["normal"] = self.data["normal"] - _data
        if not _data.issubset(self.outlier):
            self.data["outlier"].update(_data)
            self.df_outlier = self._filter_df("outlier")
            if not data_to_add:
                self.clear_selection()

    def update_normal(self, data_to_add: Sequence | None = None, base_obj=None):
        """
        Updates normal points either from passed data or from stored selected data.
        """
        obj = base_obj or st

        _data = (
            {self.value_as_datetime(e) for e in set(data_to_add)} if data_to_add else self.selection
        )

        if _data.intersection(self.data["outlier"]):
            obj.warning(
                "Some of the selected points have already been marked as outliers and were overwritten."
            )
            self.data["outlier"] = self.data["outlier"] - _data
        if not _data.issubset(self.normal):
            self.data["normal"].update(_data)
            self.df_normal = self._filter_df("normal")
            if not data_to_add:
                self.clear_selection()

    def update_plot(self, start_time: datetime.datetime, end_time: datetime.datetime):
        if (
            (not self.start and not self.end)
            or (start_time != self.start)
            or (end_time != self.end)
        ):
            self.df_plot = self.df[self.df.index.to_series().between(start_time, end_time)]
            self.start = start_time
            self.end = end_time

    def clear_selection(self):
        self.data["selected"].clear()

    def clear_all(self):
        self.clear_selection()
        self.data["outlier"].clear()
        self.data["normal"].clear()

    def _filter_df(self, key: str):
        if key not in self.data:
            raise ValueError(f"Key {key} not found.")

        return self.df[self.df.index.isin(self.data[key])]

    @staticmethod
    def value_as_datetime(value: str | int | datetime.datetime) -> datetime.datetime:
        # Plotly sometimes returns selected points as timestamp
        if isinstance(value, int):
            return datetime.datetime.fromtimestamp(value / 1000)
        # also for some reason, as a string for single point selection
        elif isinstance(value, str):
            return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M")
        else:
            return value
