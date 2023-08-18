import os, sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException

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