import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RegressorMixin:
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Pipeline:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pd.DataFrame")
        if not isinstance(y_train, pd.Series):
            raise TypeError("X_train must be a pd.Series")
        logging.info("Initializing linear regression model ")
        pipeline = Pipeline(
            [("scalar", StandardScaler()), ("model", LinearRegression())]
        )
        logging.info("Training linear regression model ")
        pipeline.fit(X_train, y_train)
        return pipeline


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy) -> None:
        self.strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy) -> None:
        self.strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        return self.strategy.build_and_train_model(X_train, y_train)
