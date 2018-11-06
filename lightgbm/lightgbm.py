import lightgbm as lgb
# Load pandas
import pandas as pd
# Load numpy
import numpy as np

train_data = np.genfromtxt("..\data\Train_data_scaled.csv" , delimiter=",")  # Returns np.array
train_labels = np.genfromtxt("..\data\Train_labels_scaled.csv" , delimiter=",") # Returns np.array

train_dataset = lgb.Dataset(train_data, label=train_labels);

params = {}
params['learning_rate'] = 0.0003
params['boosting_type'] = 'gbdt'
params['metric'] = 'mse'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 5

model = lgb.train(params, train_dataset, 250);

model.save_model('.\model\stock_prediction.txt');
