from typing import Tuple

import pandas as pd
from zenml import step

from data_splitter import DataSplitter, SimpleTrainTestSplitStrategy


@step
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.apply_split(df, target_column)
    return X_train, X_test, y_train, y_test
