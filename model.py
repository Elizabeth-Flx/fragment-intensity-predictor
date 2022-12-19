from keras import layers
from keras import Input
from keras.models import Model
from data_formatting import df

import numpy as np

class Eval_Model:

    def __init__(self, max_precursor=3):

        self.MAX_PRECURSOR = max_precursor

        #* === Peptide sequence === *#
        self.seq_input = Input(shape=(30,), name="sequence" , dtype='int32')

        #* Embedding layer
        self.embedding = layers.Embedding(input_dim=22, output_dim=8, input_length=30)(self.seq_input)

        #* CNN layers
        self.cnn = layers.Conv1D(64, 3, activation='relu', padding='same') (self.embedding)
        #self.cnn = layers.MaxPooling1D(2)                   (self.cnn)
        self.cnn = layers.Conv1D(64, 5, activation='relu', padding='same') (self.cnn)
        #self.cnn = layers.MaxPooling1D(2)                   (self.cnn)
        self.cnn = layers.Conv1D(64, 7, activation='relu', padding='same') (self.cnn)
        #self.cnn = layers.MaxPooling1D(2)                   (self.cnn)

        self.cnn_flatten = layers.Flatten()(self.cnn)

        #* === Precursor charge === *#
        self.pre_input = Input(shape=(self.MAX_PRECURSOR,), name="precursor")

        self.dense = layers.Dense(5, activation='relu')(self.pre_input)

        #* === Concatenate layers === *#
        self.concat = layers.concatenate([self.cnn_flatten, self.dense], axis=-1)

        #* Dense layers
        self.dense  = layers.Dense(128, activation='relu')(self.concat)
        #self.dense  = layers.Dense(64,  activation='relu')(self.dense)
        self.output = layers.Dense(56,  activation='sigmoid')(self.dense)

        self.model = Model([self.seq_input, self.pre_input], self.output)

        self.model.compile(
            optimizer = 'adam',
            #loss = 'cosine_similarity',
            #loss = 'mean_squared_error',
            loss = Eval_Model.spectral_angle_loss,
            metrics=[
                'cosine_similarity',
                'mean_squared_error'
            ]
        )
        print(self.model.summary())


    #* Loss Functions/Metrics

    @staticmethod
    def spectral_angle_loss(y_true, y_pred):
        import keras.backend as k
        import tensorflow
        # Normalize the vectors
        x = k.l2_normalize(y_true, axis=-1)
        y = k.l2_normalize(y_pred, axis=-1)

        # Calculate the dot product between the vectors
        dot_product = k.sum(x * y, axis=-1)

        # Return the spectral angle
        return -(1 - 2 * tensorflow.acos(dot_product) / np.pi )



    def train_model(self, training_x, training_y, n_epochs, n_batch_size, validation_data):
        training_hist = self.model.fit(
            x=training_x,
            y=training_y,
            epochs=n_epochs,
            batch_size=n_batch_size,
            validation_data=validation_data
        )
        return training_hist.history

    def predict(self, prediction_data):
        return self.model(prediction_data)
