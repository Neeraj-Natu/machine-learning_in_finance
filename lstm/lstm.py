import numpy as np
import pandas as pd
import keras 
from keras import callbacks
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, RepeatVector
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle


def model(input_shape):

    ## Initializing the model
    X_input = Input(input_shape);

    ## First LSTM Layer
    X = LSTM(10, name = 'LSTM_1', return_sequences=True)(X_input);


    ## Second LSTM Layer
    X = LSTM(5, name = 'LSTM_2', return_sequences=True)(X);
    
    #X = Flatten() (X)
    ##Dense Layer
    X = Dense(1, activation='relu', name = 'dense_1')(X)


    ##The model object
    model = Model(inputs = X_input, outputs = X, name='LSTMModel')

    return model



def main (unused_argv):

    ##Using the GPU
    with tf.device('/device:GPU:0'):
        ##Loading the data, this is incorrect way, read these are numpy array or else this clips the first row everytime.
        X_train = pickle.load(open("..\data\X_train.p","rb"));
        Y_train = pickle.load(open("..\data\Y_train.p","rb"));
        X_eval = pickle.load(open("..\data\X_eval.p","rb"));
        Y_eval = pickle.load(open("..\data\Y_eval.p","rb"));

        ## Building the model
        Model = model(X_train.shape[1:]);

        ## Compling the model
        Model.compile(optimizer = "Adam" , loss = "mean_squared_logarithmic_error", metrics = ['mean_squared_error','cosine', 'mae']);

        ## Printing the model summary
        Model.summary()

        ## Adding the callback for TensorBoard
        tensorboard = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True);

        ##fitting the model
        Hist =  Model.fit(x = X_train, y = Y_train, epochs = 300,batch_size=10, callbacks=[tensorboard], validation_data=(X_eval,Y_eval));

        ##Evaluating the model
        score = Model.evaluate(X_eval, Y_eval, batch_size=10);

        ## Saving the model
        Model.save('../lstm/model/stock_prediction.h5');
##Running the app
if __name__ == "__main__":
  tf.app.run()
