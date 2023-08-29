from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomDataClass, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        custom_data_class = CustomDataClass(
            password=request.form.get('password')
        )
        features = custom_data_class.get_custom_data()
        predict_pipeline = PredictPipeline()
        answer = predict_pipeline.predict_pipeline(features=features)
        return render_template('index.html', results = answer)


if __name__ == '__main__':
    app.run('0.0.0.0', port=8081, debug=True)