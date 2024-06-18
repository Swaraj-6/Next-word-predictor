from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from dataclasses import dataclass
import os
from model.utils import load_object

@dataclass
class DataTransformationConfig:
    tokenizer_obj_file_path: str = os.path.join("artifact", "tokenizer.pkl")
    model_obj_file_path: str = os.path.join("artifact", "model.keras")
    len_file_path: str = os.path.join("artifact", "max_len.txt")

class ModelTraining():

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        with open(self.data_transformation_config.len_file_path, 'r') as f:
            self.max_len = int(f.read())

    def model_training(self, X, y):

        tokenizer = load_object(self.data_transformation_config.tokenizer_obj_file_path)

        model = Sequential()

        model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=self.max_len-1))
        model.add(Bidirectional(LSTM(units=150, return_sequences=True)))
        model.add(Bidirectional(LSTM(units=150)))
        model.add(Dense(units=len(tokenizer.word_index)+1, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        history = model.fit(X, y, epochs=100, verbose=1)
        final_training_accuracy = history.history['accuracy'][-1]

        model.save(self.data_transformation_config.model_obj_file_path)

        return final_training_accuracy