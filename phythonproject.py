import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import yfinance as yf

def preprocess_data(data):
    data['YEAR'] = data.index.year
    data['MONTH'] = data.index.month
    data['DAY'] = data.index.day
    return data


def train_model(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    pipeline.fit(X, y)


    y_pred = pipeline.predict(X)
    mse = mean_squared_error(y, y_pred)
    print("Mean Squared Error:", mse)

    return pipeline


def predict_stock_price(model, data):
    last_data = data[['YEAR', 'MONTH', 'DAY']].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(last_data)
    return prediction[0]


def load_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data


def main():

    companies = ['RELIANCE.NS', 'MSFT', 'GOOGL', 'ACN', 'HCLTECH.NS', 'BAJAJ-AUTO.NS', 'TATAMOTORS.NS', 'ADANIPORTS.NS']

    for symbol in companies:
        print("stock predicion", symbol)
        start_date = '2000-01-01'
        end_date = '2024-02-02'
        try:
            data = load_stock_data(symbol, start_date, end_date)
            data = preprocess_data(data)
            X = data[['YEAR', 'MONTH', 'DAY']]
            y = data['Close']
            model = train_model(X, y)
            predicted_price = predict_stock_price(model, data)
            print("Predicted stock price for", symbol, ":", predicted_price)
        except Exception as e:
            print("Error predicting stock price for", symbol, ":", e)

if __name__ == "__main__":
    main()
