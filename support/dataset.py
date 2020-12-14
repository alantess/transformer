import pandas as pd


def retrieve_data():
    train_dataset = "/mnt/d/dataset/binance_BTCUSDT_5m.csv"
    df = pd.read_csv(train_dataset)
    df = df.drop(columns=['Date', 'Time', 'Volume'])
    # Columns are set at close, high, low and open.
    df = df.dropna()
    data = df.values
    return data
