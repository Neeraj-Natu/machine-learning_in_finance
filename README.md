## Machine Learning in Finance

In this repository I try to predict the value of stock 10 days into future based on some popular technical analysis parameters.
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

The repository is meant to be followed along with this [blog.](https://medium.com/@natu.neeraj/exploring-machine-learning-in-finance-fe1c7ab45ca5).

To run the jupyter notebook please go to utils folder and start the jupyter notebook.
below are the commands to do so:

For Windows users
Open command prompt

`cd .\utils`

`python jupyter notebook`

For Mac users 
Open terminal and cd to the location of repository


`cd ./utils`

`python jupyter notebook`


The models are pre built for ease of use and following along but please feel free to play around with them.
If you wish to train the models please navigate to related model folder and run below command

`python .\model-name.py`

Please note replace model name (lstm,dnn or lightgbm) instead of model-name in above command.

