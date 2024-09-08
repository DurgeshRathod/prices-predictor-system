import pandas as pd
from zenml import step

from outlier_detection import (
    IQROutlierDetection,
    OutlierDetector,
    ZScoreOutlierDetection,
)


@step
def outlier_detection_step(
    df: pd.DataFrame, column_name: str, strategy
) -> pd.DataFrame:

    if strategy == "zscore":
        outlier_detector = OutlierDetector(ZScoreOutlierDetection())
    elif strategy == "iqr":
        outlier_detector = OutlierDetector(IQROutlierDetection())

    else:
        raise ValueError(f"Unsupported outlier detection strategy {strategy}")
    df_numeric = df.select_dtypes(include=[int, float])
    cleaned_df = outlier_detector.handle_outliers(df_numeric, "remove")
    return cleaned_df
