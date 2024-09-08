import logging
from abc import ABC, abstractmethod
from typing import Literal, Union

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MissingValuesHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, axis: Literal[0, 1] = 0, thresh: Union[int, None] = None):
        if axis not in [0, 1, "index", "columns"]:
            raise ValueError(
                f"Invalid axis value: {axis}. Must be 0, 1, 'index', or 'columns'."
            )

        if thresh is not None and not isinstance(thresh, int):
            raise ValueError(
                f"Invalid thresh value: {thresh}. Must be None or an integer."
            )

        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"\nDropping Missing values with axis={self.axis} and thres={self.thresh}"
        )

        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)  # type: ignore
        logging.info("Missing Values Dropped")
        return df_cleaned


class FillMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"\nFilling missing values using the method {self.method}")
        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include=["number"]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df_cleaned[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include=["number"]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df_cleaned[numeric_columns].median()
            )
        elif self.method == "mode":
            numeric_columns = df_cleaned.select_dtypes(include=["number"]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df_cleaned[numeric_columns].mode()
            )
        elif self.method == "constant":
            numeric_columns = df_cleaned.select_dtypes(include=["number"]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                self.fill_value
            )
        else:
            logging.warning(f"Unknown method {self.method}")
        logging.info("Missing Values filled")
        return df_cleaned


class MissingValuesHandler:
    def __init__(self, strategy: MissingValuesHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValuesHandlingStrategy):
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing Handling Missing Values Strategy")
        return self._strategy.handle(df)
