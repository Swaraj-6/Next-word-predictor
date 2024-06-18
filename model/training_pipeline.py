from model.component.data_processing import DataProcessing
from model.component.model_training import ModelTraining
from flask import make_response

class TrainingPipeline():

    def __init__(self, data):
        self.data = data

    def training_model(self):

        dataProcessing = DataProcessing(self.data)
        X, y, max_len = dataProcessing.data_pre_processing()

        modelTraining = ModelTraining()
        final_training_accuracy = modelTraining.model_training(X, y, max_len)

        return make_response({"message": "Training Completed Successfully", "accuracy": final_training_accuracy}, 200)