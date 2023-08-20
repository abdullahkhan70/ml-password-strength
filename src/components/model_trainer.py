import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model, evaluate_preds
from dataclasses import dataclass
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass 
class ModelTrainerConfig:
    train_model_path: str = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr: list, test_arr: list):
        try:
            logging.info(f"Splitting the Training and Testing datasets.")
            
            # X_train, y_train, X_test, y_test = train_test_split(train_arr, test_arr, test_size=0.2, random_state=42)
            
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "random_forest": RandomForestRegressor(),
                "decision_tree": DecisionTreeRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "linear_regression": LinearRegression(),
                "k_neighbour": KNeighborsRegressor(),
                "xgb_regressor": XGBRegressor(),
                "catboosting_regressor": CatBoostRegressor(),
                "adaboost_regressor": AdaBoostRegressor()
            }
            
            model_list: dict = evaluate_model(
                    X_train=X_train, 
                    y_train=y_train, 
                    X_test=X_test, 
                    y_test=y_test,
                    models=models
            )
            
            best_model_score = max(sorted(model_list.values()))
            print(f"Best Model Score: {best_model_score}")

            best_model_name = list(model_list.keys())[
                list(model_list.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Best Model Found: {best_model}")

            save_object(
                file_path=self.model_trainer_config.train_model_path,
                obj=best_model
            )

            print(f"Predicted of Best Model: {best_model.predict(X_test)}")

            print(f"R2 Score of the Best Model: {evaluate_preds(y_test, best_model.predict(X_test))}")
            
            return evaluate_preds(y_test, best_model.predict(X_test))

        except Exception as error:
            raise CustomException(error, sys)