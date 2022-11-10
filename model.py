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
        self.embedding = layers.Embedding(input_dim=22, output_dim=64, input_length=30)(self.seq_input)

        #* CNN layers
        self.cnn = layers.Conv1D(128, 3, activation='relu', padding='same') (self.embedding)
        #self.cnn = layers.MaxPooling1D(2)                   (self.cnn)
        self.cnn = layers.Conv1D(128, 5, activation='relu', padding='same') (self.cnn)
        #self.cnn = layers.MaxPooling1D(2)                   (self.cnn)
        self.cnn = layers.Conv1D(128, 7, activation='relu', padding='same') (self.cnn)
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
            loss = 'mean_squared_error',
            #loss = Eval_Model.masked_spectral_distance,
            metrics=[
                'cosine_similarity',
                'mean_squared_error'
            ]
        )
        print(self.model.summary())



    #* Loss Functions/Metrics

    @staticmethod
    def spectral_angle_loss(y_true, y_pred):
        sum = 0
        for i in range(len(y_true)):
            sum += df.spectral_angle(y_true[i], y_pred[i])-1
        return -sum/len(y_true)

    # ! FROM PROSIT, DONT LEAVE HERE FOR DRAFTS
    @staticmethod
    def masked_spectral_distance(true, pred):
        # Note, fragment ions that cannot exists (i.e. y20 for a 7mer) must have the value  -1.
        import tensorflow
        import keras.backend as k

        epsilon = k.epsilon()
        pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
        true_masked = ((true + 1) * true) / (true + 1 + epsilon)
        pred_norm = k.l2_normalize(true_masked, axis=-1)
        true_norm = k.l2_normalize(pred_masked, axis=-1)
        product = k.sum(pred_norm * true_norm, axis=1)
        arccos = tensorflow.acos(product)
        return 2 * arccos / np.pi


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


