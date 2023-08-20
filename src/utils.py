import os, sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        print(f"File Path: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as error:
        raise CustomException(error, sys)

def load_object(file_path: str):
    try:
        dir_path = os.path.dirname(file_path)
        print(f"File Path: {dir_path}")

        with open(file_path, "rb") as file_obj:
            data = dill.load(file_obj)
            
        return data

    except Exception as error:
        raise CustomException(error, sys)

def evaluate_preds(true, pred):
    r2_scores = r2_score(true, pred)
    return r2_scores

def evaluate_model(X_train, y_train, X_test, y_test, models: dict):
    model_list: dict = {}
    for i in range(len(list(models))):
        model = list(models.values())[i]

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Matrics
        r2_score_y_test = evaluate_preds(y_test, y_test_pred)

        model_list[list(models.keys())[i]] = r2_score_y_test
        
    print(f"Model List: {model_list}")
    return model_list