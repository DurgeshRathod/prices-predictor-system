import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataSplitterStrategy(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, target: str, test_size: float = 0.2) -> tuple:
        pass


class SimpleTrainTestSplitStrategy(DataSplitterStrategy):

    def split(self, df: pd.DataFrame, target: str, test_size: float = 0.2) -> tuple:
        logging.info(f"Splitting data with test size {test_size} and target {target}")

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        logging.info("Data splitting completed")
        return X_train, X_test, y_train, y_test


class DataSplitter:
    def __init__(self, strategy: DataSplitterStrategy) -> None:
        self.strategy = strategy

    def set_strategy(self, strategy: DataSplitterStrategy) -> None:
        self.strategy = strategy

    def apply_split(
        self, df: pd.DataFrame, target: str, test_size: float = 0.2
    ) -> tuple:
        return self.strategy.split(df, target, test_size)
