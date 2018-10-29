## Machine Learning in Finance
Stock market prediction is one of the most fancied dreams and many people earn a good living being good at it.
In this code i would attempt at stock prediction 10 days in advance for Apple.
To collect the data we use alpha vantage which is a reliable and a rich source for historical data.
Also we use 5 technical factors that are most used in the real world by traders for trading in short term.
If future i would try to fundamentals and sentiment data into the analysis and would check how well the models perform with additional data.

Currently we have 20 years data from 1998 till 2018.
The data prepration is one of the most difficult step for any machine learning project thus i have done it using a jupyter notebook to be interactive and easy to follow.


To predict the stocks 10 days in future we would use a simple deep neural network, a Random Forest, a Gradient Boosted Tree and a LSTM network and see which one of these gives the better performance.

Also once we have the best model, we will use zipline and backtrader popular backtesting libraries to perform backtesting and see how much our best model would have made us using these libraries for just one stock.


