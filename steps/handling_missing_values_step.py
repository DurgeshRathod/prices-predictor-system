import pandas as pd
from zenml import step

from handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValuesHandler,
)


@step
def handle_missing_values_step(
    df: pd.DataFrame, strategy: str = "mean"
) -> pd.DataFrame:
    if strategy == "drop":
        handler = MissingValuesHandler(DropMissingValuesStrategy(axis=0))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValuesHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Unsupported Missing Value handling strategy {strategy}")
    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df
