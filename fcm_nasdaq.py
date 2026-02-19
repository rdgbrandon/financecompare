# fcm_nasdaq.py
import os
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
import copy
# Try to import Polygon RESTClient; fall back gracefully if not available
try:
    from polygon.rest import RESTClient # type: ignore
    POLYGON_AVAILABLE = True
except Exception:
    RESTClient = None
    POLYGON_AVAILABLE = False
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
    def __init__(self, learning_rate=0.002): # Reduced LR for stability with real inputs
        super().__init__()
        self.nodes.append('News_Sentiment')
        self.n_nodes += 1
        self.node_index['News_Sentiment'] = self.n_nodes - 1
        old_W = self.W.copy()
        self.W = np.zeros((self.n_nodes, self.n_nodes))
        self.W[:old_W.shape[0], :old_W.shape[1]] = old_W
        def set_weight(a, b, w):
            self.W[self.node_index[a], self.node_index[b]] = w
        set_weight('News_Sentiment', 'Investor_Sentiment', 0.6)
        set_weight('News_Sentiment', 'NASDAQ', 0.4)
        self.learning_rate = learning_rate
        self.nasdaq_idx = self.node_index['NASDAQ']
        # Add realistic causal edges for dynamic inputs (negative impacts where appropriate)
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
        for _ in range(60): # Increased iterations for better convergence with varying inputs
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
        # Backpropagate to previous layer
        direct_preds = [n for n in self.nodes if self.W[self.node_index[n], self.nasdaq_idx] != 0]
        for pred_node in direct_preds:
            pred_idx = self.node_index[pred_node]
            if pred_node not in [self.nodes[i] for i in idx]: continue  # skip if not computed node
            error_pred = delta * self.W[pred_idx, self.nasdaq_idx]
            s_prime_pred = state[pred_idx] * (1 - state[pred_idx])
            delta_pred = error_pred * s_prime_pred
            for k in range(self.n_nodes):
                self.W[k, pred_idx] += self.learning_rate * delta_pred * state[k]
                self.W[k, pred_idx] = np.clip(self.W[k, pred_idx], -3.0, 3.0)
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
# Robust function: attempt to use Polygon RESTClient if available; otherwise fallback to yfinance
def fetch_data_polygon(ticker, start, end):
    """
    Attempt to fetch daily adjusted close prices for `ticker` using Polygon RESTClient if installed.
    If Polygon is not installed or if the call fails, fall back to yfinance.
    Returns a DataFrame with an 'Adj Close' column indexed by Timestamp.
    """
    if POLYGON_AVAILABLE and RESTClient is not None:
        try:
            # RESTClient supports get_aggs(ticker, multiplier, timespan, from_, to)
            client = RESTClient() # If you need to pass api_key, set env var per polygon docs or supply here
            aggs = client.get_aggs(ticker, 1, "day", start, end)
            if not aggs:
                # empty result -> fallback
                raise RuntimeError("Polygon returned no aggregates")
            data = {'Adj Close': [a.close for a in aggs]}
            # polygon aggregate objects often have `timestamp` in milliseconds
            try:
                index = [pd.to_datetime(a.timestamp, unit='ms') for a in aggs]
            except Exception:
                # last resort try to parse from `a` attributes
                index = [pd.to_datetime(a._raw.get('t', None), unit='ms') if hasattr(a, '_raw') else pd.to_datetime(start) for a in aggs]
            df = pd.DataFrame(data, index=index)
            df.index = pd.to_datetime(df.index)
            # normalize to daily freq and keep business days
            df = df.sort_index()
            return df
        except Exception as e:
            print(f"Polygon fetch failed ({e}); falling back to yfinance for ticker {ticker}.")
            # fall through to yfinance fallback
    # Fallback using yfinance
    # Some ticker formats (like 'I:VIX') may not work with yfinance; map a few common ones:
    ticker_fallback = ticker
    if isinstance(ticker, str) and ticker.startswith('I:'):
        # Index-like prefix; map to common symbol for VIX
        sym = ticker.split(':', 1)[1]
        if sym.upper() == 'VIX':
            ticker_fallback = '^VIX'
        else:
            ticker_fallback = sym
    try:
        d = yf.download(ticker_fallback, start=start, end=end, progress=False)
        if d.empty:
            raise RuntimeError("yfinance returned empty DataFrame")
        if 'Adj Close' not in d.columns and 'Close' in d.columns:
            d['Adj Close'] = d['Close']
        return d[['Adj Close']].copy()
    except Exception as e:
        print(f"yfinance fallback failed for {ticker_fallback} ({e}). Returning constant 0.5 series.")
        # return constant series indexed by business days between start and end
        idx = pd.date_range(start=start, end=end, freq='B')
        return pd.DataFrame({'Adj Close': pd.Series(0.5, index=idx)})
def run_fcm(fcm, df):
    preds = []
    outer = {k: 0.5 for k in [
        'Large_Deficits', 'Foreign_Demand', 'Low_Unemployment',
        'Supply_Increase', 'Energy_Price_Increase', 'Consumer_Demand',
        'Business_Investment', 'Market_Volatility',
        'Equity_Inflows', 'AI_Technology_Advances'
    ]}
    for _ in range(len(df)):
        preds.append(fcm.compute_state(outer))
    df['FCM'] = pd.Series(preds, index=df.index)
    df['FCM_Return'] = df['FCM'].diff().fillna(0)
    return df
# Modified to accept source parameter for data fetching (Yahoo or "Google"/Polygon)
def run_dynamic_fcm(dynamic_fcm, df, start_date, end_date, source='yahoo'):
    preds = []
    outer_base = {k: 0.5 for k in [
        'Large_Deficits', 'Foreign_Demand', 'Low_Unemployment',
        'Supply_Increase', 'Consumer_Demand',
        'Business_Investment', 'Equity_Inflows',
        'News_Sentiment'
        # Note: Market_Volatility, Energy_Price_Increase, AI_Technology_Advances will be overridden
    ]}
    # NEW: Define tickers based on source
    if source == 'yahoo':
        proxy_tickers = {
            'Market_Volatility': '^VIX', # VIX level
            'Energy_Price_Increase': 'CL=F', # Crude oil futures (use pct change)
            'AI_Technology_Advances': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'] # Multiple tech stocks as AI proxy (level)
        }
    else: # 'google' using Polygon (or fallback)
        proxy_tickers = {
            'Market_Volatility': 'I:VIX',
            'Energy_Price_Increase': 'USO', # Use USO ETF as proxy for crude oil if polygon/yf works
            'AI_Technology_Advances': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        }
    proxy_data = {}
    for name, ticker in proxy_tickers.items():
        try:
            if isinstance(ticker, str):
                if source == 'yahoo':
                    d = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if 'Adj Close' not in d.columns and 'Close' in d.columns:
                        d['Adj Close'] = d['Close']
                    proxy_data[name] = d['Adj Close'].reindex(df.index).ffill().bfill()
                else:
                    # polygon or fallback via fetch_data_polygon
                    d = fetch_data_polygon(ticker, start_date, end_date)
                    # ensure 'Adj Close' column exists
                    if isinstance(d, pd.DataFrame):
                        if 'Adj Close' not in d.columns and 'Close' in d.columns:
                            d['Adj Close'] = d['Close']
                        proxy_data[name] = d['Adj Close'].reindex(df.index).ffill().bfill()
                    else:
                        # if fetch_data_polygon returns Series-like
                        proxy_data[name] = pd.Series(d, index=df.index).reindex(df.index).ffill().bfill()
            else:  # list of tickers
                data_list = []
                for t in ticker:
                    if source == 'yahoo':
                        d = yf.download(t, start=start_date, end=end_date, progress=False)
                        if 'Adj Close' not in d.columns and 'Close' in d.columns:
                            d['Adj Close'] = d['Close']
                        data_list.append(d['Adj Close'])
                    else:
                        d = fetch_data_polygon(t, start_date, end_date)
                        if 'Adj Close' not in d.columns and 'Close' in d.columns:
                            d['Adj Close'] = d['Close']
                        data_list.append(d['Adj Close'])
                avg = pd.concat(data_list, axis=1).mean(axis=1)
                proxy_data[name] = avg.reindex(df.index).ffill().bfill()
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
    # Add news sentiment
    news_sentiment_dict = {}
    if POLYGON_AVAILABLE:
        try:
            client = RESTClient()
            positive_words = ['rise', 'gain', 'positive', 'bull', 'growth', 'up']
            negative_words = ['fall', 'loss', 'negative', 'bear', 'decline', 'down']
            for date in df.index:
                date_str = date.strftime('%Y-%m-%d')
                news = client.list_ticker_news('^IXIC', published_utc_gte=date_str, published_utc_lte=date_str, limit=100)
                pos_count = 0
                neg_count = 0
                for n in news:
                    title = n.title.lower() if n.title else ''
                    desc = n.description.lower() if n.description else ''
                    text = title + ' ' + desc
                    pos_count += sum(text.count(w) for w in positive_words)
                    neg_count += sum(text.count(w) for w in negative_words)
                total = pos_count + neg_count
                if total > 0:
                    score = (pos_count - neg_count) / total
                else:
                    score = 0
                sentiment = dynamic_fcm.sigmoid(score * 5)  # scale to ~ -5 to 5
                news_sentiment_dict[date] = sentiment
        except Exception as e:
            print(f"News sentiment fetch failed ({e}). Using constant 0.5.")
            news_sentiment_dict = {d: 0.5 for d in df.index}
    else:
        news_sentiment_dict = {d: 0.5 for d in df.index}
    for i, date in enumerate(df.index):
        outer = outer_base.copy()
        # Set dynamic real-world inputs
        for name in proxy_tickers:
            value = proxy_data[name].loc[date]
            if name == 'Energy_Price_Increase':
                pct = energy_pct.loc[date]
                # Sigmoid on scaled pct change (negative changes -> low activation)
                outer[name] = 1 / (1 + np.exp(-pct * 25)) # *25 scales typical daily % to ~ -5 to 5
            else:
                # Level-based: causal running normalization
                r = running_proxies[name]
                if r['min'] is None: # First day
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
        outer['News_Sentiment'] = news_sentiment_dict[date]
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
    # NEW: Use source-specific column names for Google/Polygon
    col_prefix = 'Dynamic_FCM' if source == 'yahoo' else 'Dynamic_FCM_Google'
    df[col_prefix] = pd.Series(preds, index=df.index)
    df[col_prefix + '_Return'] = df[col_prefix].diff().fillna(0)
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
    cols = ['FCM_Return', 'Return']
    if 'Dynamic_FCM_Return' in df.columns:
        cols.insert(1, 'Dynamic_FCM_Return')
    # NEW: Add Google FCM return
    if 'Dynamic_FCM_Google_Return' in df.columns:
        cols.insert(2, 'Dynamic_FCM_Google_Return')
    # drop rows with NaN in those columns before fitting scaler
    feats_df = df[cols].dropna()
    features = scaler.fit_transform(feats_df)
    # Align residual computation with features_df's index to avoid shape mismatch
    if 'Dynamic_FCM_Return' in df.columns and 'Dynamic_FCM_Google_Return' in df.columns:
        residual = df.loc[feats_df.index, 'Return'] - (df.loc[feats_df.index, 'FCM_Return'] + df.loc[feats_df.index, 'Dynamic_FCM_Return'] + df.loc[feats_df.index, 'Dynamic_FCM_Google_Return']) / 3
    elif 'Dynamic_FCM_Return' in df.columns:
        residual = df.loc[feats_df.index, 'Return'] - (df.loc[feats_df.index, 'FCM_Return'] + df.loc[feats_df.index, 'Dynamic_FCM_Return']) / 2
    else:
        residual = df.loc[feats_df.index, 'Return'] - df.loc[feats_df.index, 'FCM_Return']
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
    # NEW: Add Google FCM return
    if 'Dynamic_FCM_Google_Return' in df.columns:
        cols.insert(2, 'Dynamic_FCM_Google_Return')
    # Need to drop NaNs and align so scaler.transform will accept shape
    feats_df = df[cols].dropna()
    if feats_df.empty:
        raise ValueError("No rows available for hybrid prediction after dropping NaNs.")
    features = scaler.transform(feats_df)
    X, _ = create_sequences(features, df.loc[feats_df.index, 'Return'].values)
    residual_pred = model.predict(X, verbose=0).flatten()
    df_pred = df.loc[feats_df.index].iloc[-len(residual_pred):].copy()
    df_pred['Hybrid_Pred'] = df_pred['FCM_Return'] + residual_pred
    if 'Dynamic_FCM_Return' in df_pred.columns:
        df_pred['Hybrid_Dynamic_Pred'] = df_pred['Dynamic_FCM_Return'] + residual_pred
    # NEW: Add hybrid for Google FCM
    if 'Dynamic_FCM_Google_Return' in df_pred.columns:
        df_pred['Hybrid_Dynamic_Google_Pred'] = df_pred['Dynamic_FCM_Google_Return'] + residual_pred
    return df_pred
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
        plt.plot(df.index, df['Dynamic_FCM_Return'], label=f'Dynamic FCM Yahoo (corr={corr_dyn:.2f}, RMSE={rmse_dyn:.4f})',
                 linestyle='-.', alpha=0.7)
    # NEW: Plot for Google FCM
    if 'Dynamic_FCM_Google_Return' in df.columns:
        corr_dyn_g = np.nan_to_num(pearsonr(df['Dynamic_FCM_Google_Return'], df['Return'])[0])
        rmse_dyn_g = np.sqrt(mean_squared_error(df['Return'], df['Dynamic_FCM_Google_Return']))
        plt.plot(df.index, df['Dynamic_FCM_Google_Return'], label=f'Dynamic FCM Google (corr={corr_dyn_g:.2f}, RMSE={rmse_dyn_g:.4f})',
                 linestyle=':', alpha=0.7)
    if 'Hybrid_Pred' in df.columns:
        corr_h = np.nan_to_num(pearsonr(df['Hybrid_Pred'], df['Return'])[0])
        rmse_h = np.sqrt(mean_squared_error(df['Return'], df['Hybrid_Pred']))
        plt.plot(df.index, df['Hybrid_Pred'], label=f'Hybrid (corr={corr_h:.2f}, RMSE={rmse_h:.4f})',
                 linewidth=2)
    if 'Hybrid_Dynamic_Pred' in df.columns:
        corr_hd = np.nan_to_num(pearsonr(df['Hybrid_Dynamic_Pred'], df['Return'])[0])
        rmse_hd = np.sqrt(mean_squared_error(df['Return'], df['Hybrid_Dynamic_Pred']))
        plt.plot(df.index, df['Hybrid_Dynamic_Pred'], label=f'Hybrid Dynamic Yahoo (corr={corr_hd:.2f}, RMSE={rmse_hd:.4f})',
                 linewidth=2, linestyle=':')
    # NEW: Plot for Hybrid Google
    if 'Hybrid_Dynamic_Google_Pred' in df.columns:
        corr_hdg = np.nan_to_num(pearsonr(df['Hybrid_Dynamic_Google_Pred'], df['Return'])[0])
        rmse_hdg = np.sqrt(mean_squared_error(df['Return'], df['Hybrid_Dynamic_Google_Pred']))
        plt.plot(df.index, df['Hybrid_Dynamic_Google_Pred'], label=f'Hybrid Dynamic Google (corr={corr_hdg:.2f}, RMSE={rmse_hdg:.4f})',
                 linewidth=2, linestyle='--')
    plt.legend()
    plt.grid(alpha=0.3)
    # NEW: Updated title to include comparison
    plt.title("NASDAQ FCM + LSTM Accuracy (Static vs Dynamic Yahoo vs Dynamic Google)")
    plt.tight_layout()
    plt.savefig("fcm_lstm_accuracy.png", dpi=300)
    plt.show()
if __name__ == "__main__":
    # Example dates - adjust as needed
    start_date = "2025-01-29"
    end_date = "2026-01-29"
    fcm_static = NASDAQ_FCM()
    fcm_dynamic = DynamicNASDAQ_FCM(learning_rate=0.002)
    # NEW: Create another dynamic FCM for "Google" data source
    fcm_dynamic_google = DynamicNASDAQ_FCM(learning_rate=0.002)
    df = fetch_nasdaq_data(start_date, end_date)
    df = run_fcm(fcm_static, df)
    df = run_dynamic_fcm(fcm_dynamic, df, start_date, end_date)
    # Run Google/Polygon-backed dynamic FCM (falls back to yfinance if polygon not installed)
    df_google = df.copy()
    df_google = run_dynamic_fcm(fcm_dynamic_google, df_google, start_date, end_date, source='google')
    # merge Google dynamic columns back into main df if present
    if 'Dynamic_FCM_Google' in df_google.columns:
        df['Dynamic_FCM_Google'] = df_google['Dynamic_FCM_Google']
        df['Dynamic_FCM_Google_Return'] = df_google['Dynamic_FCM_Google_Return']
    model, scaler = train_lstm(df)
    final_df = hybrid_predict(df, model, scaler)
    plot_metrics(final_df)