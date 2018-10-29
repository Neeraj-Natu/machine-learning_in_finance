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
    X = Dense(15, activation='relu', name = 'dense_1')(X_input)

    ## Dense layer 2
    X = Dense(10, activation='relu', name = 'dense_2')(X)

    ## Dense layer 3
    X = Dense(5, activation='relu', name = 'dense_3')(X)

    ##dense 2 layer
    X = Dense(2, activation='relu', name ='dense_4')(X)

    ##The model object
    model = Model(inputs = X_input, outputs = X, name='finModel')

    return model



def main(unused_argv):

    ##Using the GPU
    with tf.device('/device:GPU:0'):
        ##Loading the data
        train_data = pd.read_csv("..\data\Train_data.csv").values  # Returns np.array
        train_labels = pd.read_csv("..\data\Train_labels.csv").values # Returns np.array
        eval_data = pd.read_csv("..\data\Validation_data.csv").values  # Returns np.array
        eval_labels = pd.read_csv("..\data\Validation_labels.csv").values # Returns np.array

        ##Pre processing the data
        #train_labels = keras.utils.np_utils.to_categorical(train_labels, 10)
        #eval_labels = keras.utils.np_utils.to_categorical(eval_labels, 10)
        #train_data = np.reshape(train_data, [-1, 28, 28, 1])
        #eval_data = np.reshape(eval_data, [-1,28,28,1])

        ## Initializing the model
        Model = model(train_data.shape[1:]);

        ## Compling the model
        Model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = ["accuracy"]);

        ## Printing the modle summary
        Model.summary()

        ## Adding the callback for TensorBoard
        tensorboard = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True);

        ##fitting the model
        Model.fit(x = train_data, y = train_labels, epochs = 20, batch_size=100, callbacks=[tensorboard], validation_data=(eval_data,eval_labels) );

##Running the app
if __name__ == "__main__":
  tf.app.run()
