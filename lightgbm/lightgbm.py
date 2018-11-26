import lightgbm as lgb
# Load pandas
import pandas as pd
# Load numpy
import numpy as np

train_data = np.genfromtxt("..\data\X_Train_scaled.csv" , delimiter=",")  # Returns np.array
train_labels = np.genfromtxt("..\data\Y_Train_scaled.csv" , delimiter=",") # Returns np.array
eval_data = np.genfromtxt("..\data\X_Validation_scaled.csv" , delimiter=",")  # Returns np.array
eval_labels = np.genfromtxt("..\data\Y_Validation_scaled.csv" , delimiter=",") # Returns np.array

train_dataset = lgb.Dataset(train_data, label=train_labels);
eval_dataset = lgb.Dataset(eval_data, label=eval_labels);

params = {}
params['learning_rate'] = 0.0003
params['boosting_type'] = 'gbdt'
params['metric'] = 'mse'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 5

model = lgb.train(params, train_dataset, 250, early_stopping_rounds=150, valid_sets=[eval_dataset],verbose_eval=20);

model.save_model('../lightgbm/model/stock_prediction.txt');
