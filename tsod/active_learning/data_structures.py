from collections import defaultdict
import datetime
import pickle
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
        self.df_test_outlier = pd.DataFrame()
        self.df_test_normal = pd.DataFrame()
        self.df_plot = pd.DataFrame()
        self.start = None
        self.end = None
        self._download_data = {}

    @classmethod
    def from_other_state(cls, other):
        state = cls(other.df)
        state.__dict__ = other.__dict__

        return state

    @property
    def download_data(self):
        return pickle.dumps(self._download_data)

    @property
    def selection(self):
        return self.data["selected"]

    @property
    def outlier(self):
        return self.data["outlier"]

    @property
    def normal(self):
        return self.data["normal"]

    @property
    def test_outlier(self):
        return self.data["test_outlier"]

    @property
    def test_normal(self):
        return self.data["test_normal"]

    def update_selected(self, data: Sequence):
        to_add = {self.value_as_datetime(e) for e in set(data)}
        if not to_add.issubset(self.data["selected"]):
            self.data["selected"].update(to_add)
            self._update_df("selected")
            self._update_plot_df("selected")
            st.experimental_rerun()

    def update_data(self, key: str, data_to_add: Sequence | None = None, base_obj=None):
        obj = base_obj or st

        _data = (
            {self.value_as_datetime(e) for e in set(data_to_add)} if data_to_add else self.selection
        )

        for k, stored_data in self.data.items():
            if (k == key) or (k == "selected"):
                continue
            if _data.intersection(stored_data):
                obj.warning(
                    f"Some of the selected points have already been marked as {k} and were overwritten."
                )
                self.data[k] = self.data[k] - _data
                self._update_df(k)
                self._update_plot_df(k)

        if not _data.issubset(self.data[key]):
            self.data[key].update(_data)
            self._update_df(key)
            self._update_plot_df(key)
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

            for key in self.data:
                self._update_plot_df(key)

    def _update_plot_df(self, key: str):
        if key not in self.data:
            raise ValueError(f"Key {key} not found.")

        setattr(self, f"df_plot_{key}", self.df_plot[self.df_plot.index.isin(self.data[key])])

    def clear_selection(self):
        self.data["selected"].clear()
        self._update_df("selected")
        self._update_plot_df("selected")

    def clear_all(self):
        for key in self.data:
            self.data[key].clear()
            self._update_df(key)
            self._update_plot_df(key)

    def _update_df(self, key: str):
        if key not in self.data:
            raise ValueError(f"Key {key} not found.")

        new_df = self.df[self.df.index.isin(self.data[key])]

        setattr(self, f"df_{key}", new_df)
        self._download_data[key] = new_df

    @staticmethod
    def value_as_datetime(value: str | int | datetime.datetime) -> datetime.datetime:
        # Plotly sometimes returns selected points as timestamp
        if isinstance(value, int):
            return datetime.datetime.fromtimestamp(value / 1000)
        # also for some reason, as a string for single point selection
        elif isinstance(value, str):
            try:
                return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M")  # Plotly return format
            except ValueError:
                pass
            return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")  # Pyecharts return format
        else:
            return value
