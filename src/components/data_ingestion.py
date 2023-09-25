import os, sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.model_train import ModelTrain
from src.components.data_transformer import DataTransformer
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info(f"Data Ingestion is started.")

        try:
            data = pd.read_csv('research/data/passwords.csv')
            data.dropna(inplace=True)
            print(f"Data Shape: {data.shape}")
            print(data.duplicated().sum())
            logging.info(f"Read the dataset as DataFrame")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Train and Test dataset Split initialized.")
            training_dataset, testing_dataset = train_test_split(data, test_size=0.3, random_state=42)
            training_dataset.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            testing_dataset.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Ingestion of the data ins completed!")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path, self.ingestion_config.raw_data_path)
        except Exception as error:
            raise CustomException(error, sys)


if __name__ == '__main__':
    data_ingesiton = DataIngestion()
    train_path, test_path, raw_path = data_ingesiton.initiate_data_ingestion()
    data_transformer = DataTransformer()
    X_train_tfidr, X_test_tfidr, y_train, y_test = data_transformer.get_data_transformer(raw_path=raw_path)
    model_trainer = ModelTrain()
    model_trainer.train_model(
        X_train_tfidr=X_train_tfidr,
        X_test_tfidr=X_test_tfidr,
        y_train=y_train,
        y_test=y_test
    )
