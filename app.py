from flask import Flask, make_response, request
from model.component.data_processing import DataProcessing
from model.component.model_training import ModelTraining

app = Flask(__name__)

@app.route("/")
def home():
    return make_response({"data":"This is home"}, 200)

from controller import *

if __name__ == "__main__":
    app.run(debug=True)

