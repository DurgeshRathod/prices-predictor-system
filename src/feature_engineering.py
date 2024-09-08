import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureEngineerStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LogTransformation(FeatureEngineerStrategy):
    def __init__(self, features) -> None:
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log transformation to features {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log Transformation Completed")
        return df_transformed


class StandardScaling(FeatureEngineerStrategy):
    def __init__(self, features) -> None:
        self.features = features
        self.scalar = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Standard Scaling to features {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scalar.fit_transform(
            df[self.features].to_xarray()
        )
        logging.info("Standard Scaling Completed")
        return df_transformed


class MinMaxSclaing(FeatureEngineerStrategy):
    def __init__(self, features, feature_range=(0, 1)) -> None:
        self.features = features
        self.scalar = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Min Max Scaling to features {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scalar.fit_transform(
            df[self.features].to_xarray()
        )
        logging.info("Min Max Scaling Completed")
        return df_transformed


class OneHotEncoding(FeatureEngineerStrategy):
    def __init__(self, features) -> None:
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying One hot encoding to features {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features].to_xarray()),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One hot encoding Completed")
        return df_transformed


class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineerStrategy) -> None:
        self.strategy = strategy

    def set_strategey(self, strategy: FeatureEngineerStrategy) -> None:
        self.strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.apply_transformation(df)
