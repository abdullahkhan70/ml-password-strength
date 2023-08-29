import os, sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict_pipeline(self, features):
        try:
            model_path = "artifacts/model.pkl"
            tfidr_vecortizer_path = "artifacts/tfidr_vectorizer.pkl"
            model = load_object(model_path)
            preprocessor = load_object(tfidr_vecortizer_path)
            data_scaled = preprocessor.transform(features)
            predict = model.predict(data_scaled)[0]
            if predict == 1:
                return "Strong Password"
            else:
                return "Weak Password"
        except Exception as error:
            raise CustomException(error, sys)

class CustomDataClass:
    def __init__(
        self,
        password: str,) -> None:
        self.password = password

    def get_custom_data(self):
        try:
            custom_data_input_dict = {
                'password': [self.password]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as error:
            raise CustomException(error, sys)