import pandas as pd
from log.looger import logging
from excep.exception import customexception
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path: Path = Path("artifacts") / "raw.csv"
    train_data_path: Path = Path("artifacts") / "train.csv"
    test_data_path: Path = Path("artifacts") / "test.csv"
    dataset_path: Path = Path("dataset") / "gemstone.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            data = pd.read_csv(self.ingestion_config.dataset_path)
            logging.info("reading dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("data saved in artifacts")

            train_data, test_data = train_test_split(data, test_size=0.20)
            logging.info("train test complete")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("data ingestion complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise customexception(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
