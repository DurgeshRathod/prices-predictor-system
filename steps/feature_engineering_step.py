from typing import List

import pandas as pd
from zenml import step

from feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxSclaing,
    OneHotEncoding,
    StandardScaling,
)


@step
def feature_engineering_step(
    df: pd.DataFrame, strategy="log", features: List[str] = []
) -> pd.DataFrame:
    if strategy == "log":
        handler = FeatureEngineer(LogTransformation(features=features))
    elif strategy == "standard_scalar":
        handler = FeatureEngineer(StandardScaling(features=features))
    elif strategy == "min_max_scalar":
        handler = FeatureEngineer(MinMaxSclaing(features=features))
    elif strategy == "one_hot_encoding":
        handler = FeatureEngineer(OneHotEncoding(features=features))

    else:
        raise ValueError(f"Unsupported Feature Engineering strategy {strategy}")
    cleaned_df = handler.apply_feature_engineering(df)
    return cleaned_df
