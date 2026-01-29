import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class NASDAQ_FCM:
    def __init__(self):
        self.nodes = [
            'Large_Deficits', 'Foreign_Demand', 'Low_Unemployment', 'Supply_Increase',
            'Energy_Price_Increase', 'Consumer_Demand', 'Business_Investment',
            'Market_Volatility', 'Equity_Inflows', 'AI_Technology_Advances',
            'Monetary_Policy_Interest_Rates', 'Inflation', 'Corporate_Earnings',
            'Investor_Sentiment', 'NASDAQ'
        ]
        self.n_nodes = len(self.nodes)
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.W = np.zeros((self.n_nodes, self.n_nodes))
        self._initialize_weights()

    def _initialize_weights(self):
        def set_weight(a, b, w):
            self.W[self.node_index[a], self.node_index[b]] = w

        set_weight('Large_Deficits', 'Monetary_Policy_Interest_Rates', 0.7)
        set_weight('Foreign_Demand', 'Monetary_Policy_Interest_Rates', 0.5)
        set_weight('Low_Unemployment', 'Monetary_Policy_Interest_Rates', 0.6)
        set_weight('Monetary_Policy_Interest_Rates', 'Inflation', 0.8)
        set_weight('Low_Unemployment', 'Inflation', 0.6)
        set_weight('Supply_Increase', 'Inflation', 0.6)
        set_weight('Energy_Price_Increase', 'Inflation', 0.7)
        set_weight('Consumer_Demand', 'Inflation', 0.5)
        set_weight('Business_Investment', 'Corporate_Earnings', 0.7)
        set_weight('Consumer_Demand', 'Corporate_Earnings', 0.6)
        set_weight('Corporate_Earnings', 'Investor_Sentiment', 0.8)
        set_weight('Equity_Inflows', 'Investor_Sentiment', 0.7)
        set_weight('Investor_Sentiment', 'NASDAQ', 0.9)
        set_weight('Corporate_Earnings', 'NASDAQ', 0.8)
        set_weight('AI_Technology_Advances', 'NASDAQ', 0.7)
        set_weight('Monetary_Policy_Interest_Rates', 'NASDAQ', -0.5)
        set_weight('Inflation', 'NASDAQ', -0.4)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_state(self, outer):
        state = np.zeros(self.n_nodes)
        for k, v in outer.items():
            state[self.node_index[k]] = v

        idx = [self.node_index[n] for n in [
            'Monetary_Policy_Interest_Rates',
            'Inflation',
            'Corporate_Earnings',
            'Investor_Sentiment',
            'NASDAQ'
        ]]

        for _ in range(40):
            prev = state.copy()
            for i in idx:
                state[i] = self.sigmoid(np.dot(state, self.W[:, i]))
            if np.max(np.abs(prev - state)) < 1e-4:
                break

        return state[self.node_index['NASDAQ']]

def fetch_nasdaq_data(start, end):
    end = min(pd.to_datetime(end), pd.Timestamp.today())
    df = yf.download("^IXIC", start=start, end=end, progress=False)

    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Fix KeyError for 'Adj Close'
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']

    df['Return'] = df['Adj Close'].pct_change()
    df.dropna(inplace=True)
    return df


def run_fcm(fcm, df):
    preds = []
    outer = {k: 0.5 for k in [
        'Large_Deficits','Foreign_Demand','Low_Unemployment',
        'Supply_Increase','Energy_Price_Increase','Consumer_Demand',
        'Business_Investment','Market_Volatility',
        'Equity_Inflows','AI_Technology_Advances'
    ]}

    for _ in range(len(df)):
        preds.append(fcm.compute_state(outer))

    df['FCM'] = pd.Series(preds, index=df.index)
    df['FCM_Return'] = df['FCM'].diff().fillna(0)
    return df

def create_sequences(X, y, window=10):
    if len(X) <= window:
        raise ValueError("Not enough data for LSTM sequences.")
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

def train_lstm(df):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[['FCM_Return', 'Return']])

    residual = df['Return'] - df['FCM_Return']
    X, y = create_sequences(features, residual.values)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=16,
              validation_split=0.2,
              callbacks=[EarlyStopping(patience=5)],
              verbose=0)

    return model, scaler

def hybrid_predict(df, model, scaler):
    features = scaler.transform(df[['FCM_Return', 'Return']])
    X, _ = create_sequences(features, df['Return'].values)

    residual_pred = model.predict(X, verbose=0).flatten()
    df = df.iloc[-len(residual_pred):].copy()
    df['Hybrid_Pred'] = df['FCM_Return'] + residual_pred
    return df

def plot_metrics(df):
    corr_fcm = np.nan_to_num(pearsonr(df['FCM_Return'], df['Return'])[0])
    corr_h = np.nan_to_num(pearsonr(df['Hybrid_Pred'], df['Return'])[0])

    rmse_fcm = np.sqrt(mean_squared_error(df['Return'], df['FCM_Return']))
    rmse_h = np.sqrt(mean_squared_error(df['Return'], df['Hybrid_Pred']))

    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Return'], label='Actual')
    plt.plot(df.index, df['FCM_Return'], label=f'FCM (corr={corr_fcm:.2f})')
    plt.plot(df.index, df['Hybrid_Pred'], label=f'Hybrid (corr={corr_h:.2f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("NASDAQ FCM + LSTM Accuracy")
    plt.tight_layout()
    plt.savefig("fcm_lstm_accuracy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    start_date = "2025-01-29"
    end_date = "2026-01-29"

    fcm = NASDAQ_FCM()
    df = fetch_nasdaq_data(start_date, end_date)
    df = run_fcm(fcm, df)

    model, scaler = train_lstm(df)
    final_df = hybrid_predict(df, model, scaler)

    plot_metrics(final_df)