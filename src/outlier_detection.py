import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3) -> None:
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outlier using Z Score method")
        z_scores = np.abs((df - df.mean)) / df.std()
        outliers = z_scores > self.threshold
        logging.info("Outlier detected with Z score threshold")
        return pd.DataFrame(outliers)


class IQROutlierDetection(OutlierDetectionStrategy):

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outlier using IQR method")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q1 - Q3
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Outlier detected with IQR threshold")
        return pd.DataFrame(outliers)


class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy) -> None:
        self.strategy = strategy

    def set_strategey(self, strategy: OutlierDetectionStrategy) -> None:
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method="remove"):
        outliers = self.detect_outliers(df)
        if method == "remove":
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            df_cleaned = df.clip(
                lower=df.to_xarray().quantile(0.01),
                upper=df.to_xarray().quantile(0.99),
                axis=1,
            )
        else:
            logging.warning(f"Unknown method {method} for outlier detection")
            return df

        logging.info("Outlier detection completed")
        return df_cleaned
