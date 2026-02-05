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
            if k in self.node_index:
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


class DynamicNASDAQ_FCM(NASDAQ_FCM):
    def __init__(self, learning_rate=0.002):  # Reduced LR for stability with real inputs
        super().__init__()
        self.learning_rate = learning_rate
        self.nasdaq_idx = self.node_index['NASDAQ']

        # Add realistic causal edges for dynamic inputs (negative impacts where appropriate)
        def set_weight(a, b, w):
            self.W[self.node_index[a], self.node_index[b]] = w

        # High volatility dampens sentiment and NASDAQ
        set_weight('Market_Volatility', 'Investor_Sentiment', -0.65)
        set_weight('Market_Volatility', 'NASDAQ', -0.55)

        # High energy prices (inflationary) can indirectly hurt via existing paths, but add direct if needed
        set_weight('Energy_Price_Increase', 'Investor_Sentiment', -0.4)

        # AI advances positive, already connected

    def compute_state_with_update(self, outer, target):
        state = np.zeros(self.n_nodes)
        for k, v in outer.items():
            if k in self.node_index:
                state[self.node_index[k]] = v

        idx = [self.node_index[n] for n in [
            'Monetary_Policy_Interest_Rates',
            'Inflation',
            'Corporate_Earnings',
            'Investor_Sentiment',
            'NASDAQ'
        ]]

        for _ in range(60):  # Increased iterations for better convergence with varying inputs
            prev = state.copy()
            for i in idx:
                state[i] = self.sigmoid(np.dot(state, self.W[:, i]))
            if np.max(np.abs(prev - state)) < 1e-5:
                break

        pred = state[self.nasdaq_idx]
        error = target - pred

        s_prime = pred * (1 - pred)
        delta = error * s_prime

        # Update incoming weights to NASDAQ with clipping for stability
        for k in range(self.n_nodes):
            self.W[k, self.nasdaq_idx] += self.learning_rate * delta * state[k]
            self.W[k, self.nasdaq_idx] = np.clip(self.W[k, self.nasdaq_idx], -3.0, 3.0)

        return pred


def fetch_nasdaq_data(start, end):
    end = min(pd.to_datetime(end), pd.Timestamp.today())
    df = yf.download("^IXIC", start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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


def run_dynamic_fcm(dynamic_fcm, df, start_date, end_date):
    preds = []
    outer_base = {k: 0.5 for k in [
        'Large_Deficits', 'Foreign_Demand', 'Low_Unemployment',
        'Supply_Increase', 'Consumer_Demand',
        'Business_Investment', 'Equity_Inflows'
        # Note: Market_Volatility, Energy_Price_Increase, AI_Technology_Advances will be overridden
    ]}

    # Fetch real-world proxy data using yfinance
    proxy_tickers = {
        'Market_Volatility': '^VIX',              # VIX level
        'Energy_Price_Increase': 'CL=F',          # Crude oil futures (use pct change)
        'AI_Technology_Advances': 'NVDA'          # NVIDIA as AI proxy (level)
    }

    proxy_data = {}
    for name, ticker in proxy_tickers.items():
        try:
            d = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if 'Adj Close' not in d.columns:
                d['Adj Close'] = d['Close']
            # Reindex to NASDAQ dates, forward/backward fill for missing
            proxy_data[name] = d['Adj Close'].reindex(df.index).ffill().bfill()
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}. Using constant 0.5.")
            proxy_data[name] = pd.Series(0.5, index=df.index)

    # Precompute energy pct change
    energy_pct = proxy_data['Energy_Price_Increase'].pct_change().fillna(0)

    # Running min/max for level-based proxies (causal)
    level_proxies = ['Market_Volatility', 'AI_Technology_Advances']
    running_proxies = {name: {'min': None, 'max': None} for name in level_proxies}

    # Running min/max for NASDAQ target (causal)
    running_min = df.iloc[0]['Adj Close']
    running_max = df.iloc[0]['Adj Close']

    for i, date in enumerate(df.index):
        outer = outer_base.copy()

        # Set dynamic real-world inputs
        for name in proxy_tickers:
            value = proxy_data[name].loc[date]

            if name == 'Energy_Price_Increase':
                pct = energy_pct.loc[date]
                # Sigmoid on scaled pct change (negative changes -> low activation)
                outer[name] = 1 / (1 + np.exp(-pct * 25))  # *25 scales typical daily % to ~ -5 to 5
            else:
                # Level-based: causal running normalization
                r = running_proxies[name]
                if r['min'] is None:  # First day
                    r['min'] = value
                    r['max'] = value
                    norm = 0.5
                else:
                    denom = r['max'] - r['min']
                    norm = (value - r['min']) / denom if denom > 0 else 0.5
                    norm = np.clip(norm, 0, 1)
                outer[name] = norm

                # Update running AFTER setting (causal)
                r['min'] = min(r['min'], value)
                r['max'] = max(r['max'], value)

        # Causal normalized NASDAQ target (price level tracking)
        close = df.loc[date, 'Adj Close']
        if running_max == running_min:
            target = 0.5
        else:
            target = (close - running_min) / (running_max - running_min)
        target = np.clip(target, 0, 1)

        # Compute & update
        pred = dynamic_fcm.compute_state_with_update(outer, target)
        preds.append(pred)

        # Update NASDAQ running min/max AFTER
        running_min = min(running_min, close)
        running_max = max(running_max, close)

    df['Dynamic_FCM'] = pd.Series(preds, index=df.index)
    df['Dynamic_FCM_Return'] = df['Dynamic_FCM'].diff().fillna(0)
    return df


# The rest of the functions remain the same as previous version
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
    cols = ['FCM_Return', 'Return']
    if 'Dynamic_FCM_Return' in df.columns:
        cols.insert(1, 'Dynamic_FCM_Return')
    features = scaler.fit_transform(df[cols])

    if 'Dynamic_FCM_Return' in df.columns:
        residual = df['Return'] - (df['FCM_Return'] + df['Dynamic_FCM_Return']) / 2
    else:
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
    cols = ['FCM_Return', 'Return']
    if 'Dynamic_FCM_Return' in df.columns:
        cols.insert(1, 'Dynamic_FCM_Return')
    features = scaler.transform(df[cols])
    X, _ = create_sequences(features, df['Return'].values)

    residual_pred = model.predict(X, verbose=0).flatten()
    df = df.iloc[-len(residual_pred):].copy()
    df['Hybrid_Pred'] = df['FCM_Return'] + residual_pred

    if 'Dynamic_FCM_Return' in df.columns:
        df['Hybrid_Dynamic_Pred'] = df['Dynamic_FCM_Return'] + residual_pred

    return df


def plot_metrics(df):
    corr_fcm = np.nan_to_num(pearsonr(df['FCM_Return'], df['Return'])[0])
    rmse_fcm = np.sqrt(mean_squared_error(df['Return'], df['FCM_Return']))

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Return'], label='Actual', linewidth=1.5)

    plt.plot(df.index, df['FCM_Return'], label=f'Static FCM (corr={corr_fcm:.2f}, RMSE={rmse_fcm:.4f})',
             linestyle='--', alpha=0.7)

    if 'Dynamic_FCM_Return' in df.columns:
        corr_dyn = np.nan_to_num(pearsonr(df['Dynamic_FCM_Return'], df['Return'])[0])
        rmse_dyn = np.sqrt(mean_squared_error(df['Return'], df['Dynamic_FCM_Return']))
        plt.plot(df.index, df['Dynamic_FCM_Return'], label=f'Dynamic FCM (corr={corr_dyn:.2f}, RMSE={rmse_dyn:.4f})',
                 linestyle='-.', alpha=0.7)

    if 'Hybrid_Pred' in df.columns:
        corr_h = np.nan_to_num(pearsonr(df['Hybrid_Pred'], df['Return'])[0])
        rmse_h = np.sqrt(mean_squared_error(df['Return'], df['Hybrid_Pred']))
        plt.plot(df.index, df['Hybrid_Pred'], label=f'Hybrid (corr={corr_h:.2f}, RMSE={rmse_h:.4f})',
                 linewidth=2)

    if 'Hybrid_Dynamic_Pred' in df.columns:
        corr_hd = np.nan_to_num(pearsonr(df['Hybrid_Dynamic_Pred'], df['Return'])[0])
        rmse_hd = np.sqrt(mean_squared_error(df['Return'], df['Hybrid_Dynamic_Pred']))
        plt.plot(df.index, df['Hybrid_Dynamic_Pred'], label=f'Hybrid Dynamic (corr={corr_hd:.2f}, RMSE={rmse_hd:.4f})',
                 linewidth=2, linestyle=':')

    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("NASDAQ FCM + LSTM Accuracy (Static vs Dynamic with Real Data)")
    plt.tight_layout()
    plt.savefig("fcm_lstm_accuracy.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    start_date = "2025-01-29"
    end_date = "2026-01-29"

    fcm_static = NASDAQ_FCM()
    fcm_dynamic = DynamicNASDAQ_FCM(learning_rate=0.002)

    df = fetch_nasdaq_data(start_date, end_date)

    df = run_fcm(fcm_static, df)
    df = run_dynamic_fcm(fcm_dynamic, df, start_date, end_date)

    model, scaler = train_lstm(df)
    final_df = hybrid_predict(df, model, scaler)

    plot_metrics(final_df)