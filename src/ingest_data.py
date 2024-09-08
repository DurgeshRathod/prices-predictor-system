import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass


class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError(
                f"{file_path.split('.')[0]} is not a supported extension for zip ingestor"
            )
        # TODO IMPLEMENT THIS. should support a single csv from extracted zip
        output_dir = "extracted_data"
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        extracted_files = os.listdir(output_dir)
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        if len(csv_files) == 0:
            raise ValueError("NO CSV FOUND in extracted data")
        if len(csv_files) > 1:
            raise ValueError("more than 1 csv file found in the extracted data")
        return pd.read_csv(csv_files[0])


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        match file_extension:
            case ".zip":
                return ZipDataIngestor()
            case _:
                raise ValueError(
                    f"{file_extension} is not a supported extension for data ingestion"
                )
