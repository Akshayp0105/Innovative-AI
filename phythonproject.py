Python 3.11.7 (tags/v3.11.7:fa7a6f2, Dec  4 2023, 19:24:49) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error  # Corrected import
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import yfinance as yf

# Function to preprocess the data
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    return data

# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Print the mean squared error on the test set
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
...     print("Mean Squared Error:", mse)
...     
...     return pipeline
... 
... # Function to predict stock price
... def predict_stock_price(model, data):
...     last_data = data.iloc[-1].values.reshape(1, -1)
...     prediction = model.predict(last_data)
...     return prediction[0]
... 
... # Load historical stock data
... def load_stock_data(symbol, start_date, end_date):
...     data = yf.download(symbol, start=start_date, end=end_date)
...     return data
... 
... # Main function
... def main():
...     # Load historical data
...     symbol = 'AAPL'  # Example stock symbol (Apple)
...     start_date = '2020-01-01'
...     end_date = '2022-01-01'
...     data = load_stock_data(symbol, start_date, end_date)
...     
...     # Preprocess data
...     data = preprocess_data(data)
...     
...     # Define features and target
...     X = data[['Year', 'Month', 'Day']]
...     y = data['Close']
...     
...     # Train the model
...     model = train_model(X, y)
...     
...     # Predict stock price
...     prediction = predict_stock_price(model, data)
...     print("Predicted stock price:", prediction)
... 
... if __name__ == "__main__":
...     main()
