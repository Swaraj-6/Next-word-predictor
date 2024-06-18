from dataclasses import dataclass
import os
from model.utils import load_object
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from flask import make_response



@dataclass
class DataTransformationConfig:
    tokenizer_obj_file_path: str = os.path.join("artifact", "tokenizer.pkl")
    model_obj_file_path: str = os.path.join("artifact", "model.keras")
    len_file_path: str = os.path.join("artifact", "max_len.txt")

class Prediction():

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        with open(self.data_transformation_config.len_file_path, 'r') as f:
            self.max_len = int(f.read())

    def predict(self, text, no_of_pred):
        tokenizer = load_object(self.data_transformation_config.tokenizer_obj_file_path)
        model = load_model(self.data_transformation_config.model_obj_file_path)

        result = []
        for i in range(no_of_pred):
            # Tokenize
            tokenized_text = tokenizer.texts_to_sequences([text])[0]
            # Padding
            tokenized_text = pad_sequences([tokenized_text], maxlen=self.max_len-1, padding='pre')
            # Predict
            predicted_classes = np.argmax(model.predict(tokenized_text))

            for word, index in tokenizer.word_index.items():
                if index == predicted_classes:
                    text = text + " " + word
                    result.append(text)

        return make_response({"Result": result}, 200)

