import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from dataclasses import dataclass
import os
from model.utils import save_object


@dataclass
class DataTransformationConfig:
    tokenizer_obj_file_path: str = os.path.join("artifact", "tokenizer.pkl")
    len_file_path: str = os.path.join("artifact", "max_len.txt")

class DataProcessing():
    
    def __init__(self, data):
        self.data = data
        self.tokenizer = Tokenizer()
        self.data_transformation_config = DataTransformationConfig()

    def data_pre_processing(self):
        self.data = [x for x in self.data.split("\n") if x != ""]

        self.tokenizer.fit_on_texts(self.data)
        input_sequences = []

        for sentence in self.data:
            tokenized_sent = self.tokenizer.texts_to_sequences([sentence])[0]

            for i in range(1, len(tokenized_sent)):
                n_gram_seq = tokenized_sent[:i+1]
                input_sequences.append(n_gram_seq)                                                          
        
        max_len = max([len(x) for x in input_sequences])

        padded_input_seq = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
        X = padded_input_seq[:, :-1]
        y = to_categorical(padded_input_seq[:, -1], num_classes=len(self.tokenizer.word_index)+1)

        save_object(self.data_transformation_config.tokenizer_obj_file_path, self.tokenizer)
        with open(self.data_transformation_config.len_file_path, 'w') as f:
            f.write(str(max_len))

        return X, y