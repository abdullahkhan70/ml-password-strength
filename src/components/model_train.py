import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluates_model, save_object, get_accuracy_score
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


@dataclass
class ModelTrainConfig:
    model_train_path: str = os.path.join('artifacts', 'model.pkl')

@dataclass
class ModelTrain:
    def __init__(self) -> None:
        self.model_train_config = ModelTrainConfig()

    def train_model(self, X_train_tfidr, X_test_tfidr, y_train, y_test):
        try:
            models = {
                "random_forest": RandomForestClassifier(),
                "adaboost_classifier": AdaBoostClassifier(),
                "gradient_boost_classifier": GradientBoostingClassifier(),
                'knn_neighbour_classifier': KNeighborsClassifier(),
                'xgb_classifier': XGBClassifier(),
                'cat_boost_classifier': CatBoostClassifier()
            }
            model_list: dict = evaluates_model(
                X_train_tfidr=X_train_tfidr,
                X_test_tfidr=X_test_tfidr,
                y_train=y_train,
                y_test=y_test,
                models=models
            )

            best_model_score = max(sorted(model_list.values()))
            print(f"Best Model Score: {best_model_score}")

            best_model_name = list(model_list.keys())[
                list(model_list.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Best Model Name: {best_model_name}")

            save_object(
                file_path=self.model_train_config.model_train_path,
                obj=best_model
            )

            print(f"Predicted of Best Model: {best_model.predict(X_test_tfidr)}")
            logging.info(f"Predicted of Best Model: {best_model.predict(X_test_tfidr)}")

            print(f"R2 Score of the Best Model: {get_accuracy_score(y_test, best_model.predict(X_test_tfidr))}")
            logging.info(f"R2 Score of the Best Model: {get_accuracy_score(y_test, best_model.predict(X_test_tfidr))}")

            return get_accuracy_score(y_test, best_model.predict(X_test_tfidr))

        except Exception as error:
            raise CustomException(error, sys)