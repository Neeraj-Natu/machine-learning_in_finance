## Machine Learning in Finance

Predicting the future has always been one of the fancied dreams.
In this repository i try to predict the value of stock 10 days into future based on some popular technical analysis parameters.
The data is collected from Alpha Vantage APIs. The historical data is collected for past 20 years. (1998 till 2018).
And the stock for which the prices are predicted is **_"AAPL"_**

The technical indicators used are :
  - ADX
  - RSI
  - SMA
  - MACD

Along with time series data that includes the daily :

  - high
  - low
  - close
  - adjusted close
  - dividend
  - split cofficient

There are a total of 4 different models used namely **Deep neural Network**, **Light-GBM**, **Random-Forest** and **LSTM** for training on same data and their performace is measured on the test data the metric used is R2 score for all the models to be consistent.

The data prepration and Comparing of the models have been done using jupyter notebooks so that these processes are self explanatory and each step is explained in detail.

###### The repository structure is as:
```

machine-learning_in_finance
-data
-dnn
--dnn.py (the DNN model)
--model 
---stock-prediction.h5 (The saved DNN model after training)
--Graph (Folder for tensor board graphs).
-lightgbm
--model
---stock_prediction.txt (The saved lightgbm model after training)
--lightgbm.py (the lightgbm model)
-lstm
--lstm.py (the lstm model)
--model
---stock-prediction.h5 (The saved LSTM model after training)
--Graph (Folder for tensor board graphs)
-randomforest
--model
---stock_prediction.sav (The saved Random-Forest model after training)
--randomforest.py (the Random Forest model)
-Utils
--Data-prep.ipynb (The data prep notebook)
--Stock_Prediction.ipynb (The final prediction and model comparision notebook)

```

