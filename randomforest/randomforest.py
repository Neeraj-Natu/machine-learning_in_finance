# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestRegressor
import pickle
# Load pandas
import pandas as pd
# Load numpy
import numpy as np

X_train = np.genfromtxt("..\data\X_Train_scaled.csv" , delimiter=",")  # Returns np.array
Y_train = np.genfromtxt("..\data\Y_Train_scaled.csv" , delimiter=",") # Returns np.array

model = RandomForestRegressor(n_estimators=250, max_depth=10);
model.fit(X_train, Y_train);

pickle.dump( model , open( "../randomforest/model/stock_prediction.p", "wb" ) );