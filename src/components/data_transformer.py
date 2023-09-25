import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


@dataclass
class DataTransformerConfig:
    preprocessor_path: str = os.path.join('artifacts', "preprocessor.pkl")
    tfidr_vectorizer_path: str = os.path.join('artifacts', "tfidr_vectorizer.pkl")

class DataTransformer:
    def __init__(self) -> None:
        self.data_transformer_config = DataTransformerConfig()

    def get_transformer(self):
        imputer = SimpleImputer(strategy='most_frequent', fill_value='')
        return imputer


    def get_data_transformer(self, raw_path: str):
        
        try:
            # Load the Training and Testing Datasets.
            train_data = pd.read_csv(raw_path)
            # test_data = pd.read_csv(test_path)

            # Pick the desired columns to train model.
            x = train_data[["password"]]
            y = train_data[["strength"]]

            # Split the train and test from dataset.
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=np.random.seed(42))

            print(f"X_train Shape is: {X_train.shape}")
            print(f"y_train Shape is: {y_train.shape}")

            # Call transformer to fit and transforms the training and testing dataset.
            column_transformer = self.get_transformer()
            X_train = column_transformer.fit_transform(X_train.values.reshape(-1, 1)).ravel()
            X_test = column_transformer.transform(X_test.values.reshape(-1, 1)).ravel()

            print(f"After applying fit_transform Shape is: {X_train.shape}")
            print(f"After applying transform Shape is: {X_test.shape}")

            """
            We will use Analyser as 'char' because we don't have any
            duplicated password words, in our training dataset.
            """
            Vectorizer = TfidfVectorizer(analyzer='char')
            X_train_tfidr = Vectorizer.fit_transform(X_train)
            X_test_tfidr = Vectorizer.transform(X_test)

            logging.info("Saving Preprocessor pickle.")

            save_object(
                file_path=self.data_transformer_config.preprocessor_path,
                obj=column_transformer
            )

            logging.info("Saved Preprocessor pickle.")
            logging.info("Saving TFIDR Vectorizer pickle.")

            save_object(
                file_path=self.data_transformer_config.tfidr_vectorizer_path,
                obj=Vectorizer
            )

            logging.info("Saving TFIDR Vectorizer pickle.")

            return (
                X_train_tfidr,
                X_test_tfidr,
                y_train,
                y_test
            )
        except Exception as error:
            raise CustomException(error, sys)







