# The paper mentions to prepare the data from S&P 500 although 
# This repository uses 10 popular stocks that have been around for 20 years instead of entire S&P 
# to avoid complexity in preparing the data and avoid surviorship bias

# We use the free Data from Alpha Vantage on below stocks 


import requests
import json


url = "https://www.alphavantage.co/query"
apikey = " V1SZPLE0U8CCXSFW."


