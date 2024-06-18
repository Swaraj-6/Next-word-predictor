from app import app
from flask import request
from model.training_pipeline import TrainingPipeline
from model.prediction_pipeline import Prediction


@app.route("/training", methods=["POST"])
def training_controller():
    trainingPipeline = TrainingPipeline(request.json["data"])
    return trainingPipeline.training_model()


@app.route("/prediction", methods=["POST"])
def prediction_controller():
    prediction = Prediction()
    return prediction.predict(request.json["data"], request.json["no_of_pred"])
