import pandas as pd;

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&outputsize=full&datatype=csv&apikey=V1SZPLE0U8CCXSFW.";

series = pd.read_json(path_or_buf=url);


print(series);