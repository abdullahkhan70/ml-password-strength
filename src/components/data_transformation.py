import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_object_path: str = os.path.join('artifacts', 'preprocessor.pkl')

@dataclass
class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation(self):
        try:
            numerical_features = ["rank", "value", "rank_alt", "font_size", "offline_crack_sec"]
            # numerical_features = ["value"]
            categorical_features = ["category"]
            # categorical_features = ["password"]
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])
            logging.info(f"Numberical Columns: {numerical_features}")
            logging.info(f"Categorical Columns: {categorical_features}")
            preprocessor = ColumnTransformer(transformers=[
                ("numerical_pipeline", numerical_pipeline, numerical_features),
                ("categorical_pipeline", categorical_pipeline, categorical_features)
            ])
            return (preprocessor, numerical_features)
        except Exception as error:
            raise CustomException(error, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            if len(train_path) > 0 and len(test_path) > 0:
                train_data = pd.read_csv(train_path)
                # train_data = np.transpose(train_data)
                print(f"Training Dataset Shape: {train_data.shape}")
                test_data = pd.read_csv(test_path)
                # test_data = np.transpose(test_data)
                print(f"Testing Dataset Shape: {train_data.shape}")
                logging.info(f"Read the train and test dataset completed.")
                logging.info(f"Obtaining Preprocessing Object.")

                preprocessor_obj, _ = self.get_transformation()

                target_column = "strength"

                input_feature_train_data = train_data.drop(columns=[target_column], axis=1)
                target_feature_train_data = train_data[target_column]

                input_feature_test_data = test_data.drop(columns=[target_column], axis=1)
                target_feature_test_data = test_data[target_column]

                logging.info(f"Applying Preprocessing object in training DataFrame and Testing DataFrame.")

                input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
                input_feature_test_arr = preprocessor_obj.transform(input_feature_test_data)
                
                print(f"Train Array Shape: {input_feature_train_arr.shape}")
                print(f"Test Array Shape: {input_feature_test_arr.shape}")

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_data)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_data)
                ]

                print(f"Test Arr Shape: {test_arr.shape}")

                logging.info(f"Successfully, saved the Preprocessor object.")

                save_object(
                    file_path = self.data_transformation_config.preprocessor_object_path,
                    obj = preprocessor_obj
                )

                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_object_path
                )

        except Exception as error:
            raise CustomException(error, sys)