# In here we try to predict the value of stock using a deep neural network.
# We use Keras with Tensorflow as back end to create the model.

import keras as keras
from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K
from keras import callbacks
import tensorflow as tf
import numpy as np
import pandas as pd

def model(input_shape):
##First input
    X_input = Input(input_shape)

    ## Dense Layer 1
    X = Dense(10, activation='relu', name = 'dense_1')(X_input)

    ## Dense layer 2
    X = Dense(10, activation='relu', name = 'dense_2')(X)

    ## Dense layer 3
    X = Dense(5, activation='relu', name = 'dense_3')(X)

    ## Debse layer 4
    X = Dense(1, activation='relu', name ='dense_4')(X)

    ##The model object
    model = Model(inputs = X_input, outputs = X, name='finModel')

    return model



def main(unused_argv):

    ##Using the GPU
    with tf.device('/device:GPU:0'):
        ##Loading the data, this is incorrect way, read these are numpy array or else this clips the first row everytime.
        X_train = np.genfromtxt("..\data\X_Train_scaled.csv" , delimiter=",")  # Returns np.array
        Y_train = np.genfromtxt("..\data\Y_Train_scaled.csv" , delimiter=",") # Returns np.array
        X_eval = np.genfromtxt("..\data\X_Validation_scaled.csv" , delimiter=",")  # Returns np.array
        Y_eval = np.genfromtxt("..\data\Y_Validation_scaled.csv" , delimiter=",") # Returns np.array

        ## Initializing the model
        Model = model(train_data.shape[1:]);

        ## Compling the model
        Model.compile(optimizer = "Adam" , loss = "mean_squared_logarithmic_error", metrics = ['mean_squared_error','cosine', 'mae']);

        ## Printing the modle summary
        Model.summary()

        ## Adding the callback for TensorBoard
        tensorboard = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True);

        ##fitting the model
        Hist =  Model.fit(x = X_train, y = Y_train, epochs = 300, batch_size=100, callbacks=[tensorboard], validation_data=(X_eval,Y_eval) );

        ##Evaluating the model
        score = Model.evaluate(X_eval, Y_eval, batch_size=100);

        ## Saving the model
        Model.save('../dnn/model/stock_prediction.h5');
##Running the app
if __name__ == "__main__":
  tf.app.run()
