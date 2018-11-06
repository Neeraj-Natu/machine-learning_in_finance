# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

# Load pandas
import pandas as pd

# Load numpy
import numpy as np


train_data = np.genfromtxt("..\data\Train_data_scaled.csv" , delimiter=",")  # Returns np.array
train_labels = np.genfromtxt("..\data\Train_labels_scaled.csv" , delimiter=",") # Returns np.array

model = RandomForestRegressor(n_estimators=250, max_depth=10);
model.fit(train_data, train_labels);


joblib.dump(model, '.\model\stock_prediction.sav');