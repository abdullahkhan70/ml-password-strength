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
            # model_path = os.path.dirname(os.path.join('artifacts', 'model.pkl'))
            model_path = "artifacts/model.pkl"
            # preprocessor_path = os.path.dirname(os.path.join('artifacts', 'preprocessor.pkl'))
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)
        except Exception as error:
            raise CustomException(error, sys)

class CustomDataClass:
    def __init__(
        self,
        password: str,
        rank: int,
        value: int,
        rank_alt: int,
        font_size: int,
        offline_crack_sec: int) -> None:
        self.password = password
        self.rank = rank
        self.rank_alt = rank_alt
        self.value = value
        self.font_size = font_size
        self.offline_crack_sec = offline_crack_sec

    def get_custom_data(self):
        try:
            custom_data_input_dict = {
                'password': [self.password],
                'rank': [self.rank],
                'rank_alt': [self.rank_alt],
                'value': [self.value],
                'font_size': [self.font_size],
                'offline_crack_sec': [self.offline_crack_sec]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as error:
            raise CustomException(error, sys)