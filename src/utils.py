import os, sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

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
            data = dill.load(file=file_obj)
            
        return data

    except Exception as error:
        raise CustomException(error, sys)

def evaluate_preds(true, pred):
    r2_scores = r2_score(true, pred)
    return r2_scores

def get_accuracy_score(true, pred):
    return accuracy_score(true, pred)

def evaluates_model(X_train_tfidr, X_test_tfidr, y_train, y_test, models: dict):
    try:
        model_list: dict = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            if list(models.keys())[i] == "xgb_classifier":
                le = LabelEncoder()
                le_y_train = le.fit_transform(y_train)

                model.fit(X_train_tfidr, le_y_train)
            else:
                model.fit(X_train_tfidr, y_train)
            
            y_test_pred = model.predict(X_test_tfidr)

            r2_scores = get_accuracy_score(y_test, y_test_pred)

            model_list[list(models.keys())[i]] = r2_scores
            
        return model_list
    except Exception as error:
        raise CustomException(error, sys)