# visualize_fcm.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yfinance as yf   
import pandas as pd
from datetime import datetime, timedelta
import copy
from matplotlib.animation import FuncAnimation

# NEW: Try to import Polygon for Google proxies
try:
    from polygon.rest import RESTClient
    POLYGON_AVAILABLE = True
except Exception:
    RESTClient = None
    POLYGON_AVAILABLE = False

# -------------------------
# --- Canvas & axes setup
# -------------------------
# NEW: Changed to 1x3 subplots for adding Google dynamic panel
fig, axs = plt.subplots(1, 3, figsize=(54, 14))
ax_static, ax_dyn_yahoo, ax_dyn_google = axs
for ax in axs:
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')

# -------------------------
# --- Nodes + layout 
# -------------------------
outer_nodes = [
    ('Large_Deficits', '^IRX', '13W Treasury Bill', 1.5, 12),
    ('Foreign_Demand', 'UUP', 'US Dollar Index', 1.5, 10),
    ('Low_Unemployment', 'N/A', 'No proxy (uses 0.5)', 1.5, 8),
    ('Supply_Increase', 'XLI', 'Industrial ETF', 1.5, 6),
    ('Energy_Price_Increase', 'XLE', 'Energy ETF', 1.5, 4),
    ('Consumer_Demand', 'XLY', 'Consumer Discr. ETF', 6, 1),
    ('Business_Investment', 'XLI', 'Industrial ETF (DUPLICATE)', 9, 1),
    ('Market_Volatility', '^VIX', 'Volatility Index', 12, 1),
    ('Equity_Inflows', 'SPY', 'S&P 500 ETF', 15, 1),
    ('AI_Technology', 'NVDA', 'NVIDIA', 18, 4),
]

inner_nodes = [
    ('Monetary_Policy', 8, 10),
    ('Inflation', 8, 7),
    ('Corporate_Earnings', 12, 10),
    ('Investor_Sentiment', 12, 7),
    ('NASDAQ', 15, 8.5),
]

# canonical node order (outer -> inner -> target)
node_order = [n[0] for n in outer_nodes] + [n[0] for n in inner_nodes]

# -------------------------
# --- Drawing helper funcs 
# -------------------------
def draw_outer_node(ax, name, ticker, desc, x, y):
    box = mpatches.FancyBboxPatch((x-1.4, y-0.6), 2.8, 1.2,
                                   boxstyle="round,pad=0.05",
                                   facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y+0.2, name.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x, y-0.35, f'{ticker}', ha='center', va='center', fontsize=7, color='darkblue')

def draw_inner_node(ax, name, x, y):
    circle = plt.Circle((x, y), 0.9, facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, name.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')

def draw_target_node(ax, name, x, y):
    circle = plt.Circle((x, y), 1.1, facecolor='lightgreen', edgecolor='darkgreen', linewidth=3)
    ax.add_patch(circle)
    ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')

# draw base layout onto a given axis (nodes + legends + proxy text)
def draw_base_layout(ax, show_legend=True):
    for name, ticker, desc, x, y in outer_nodes:
        draw_outer_node(ax, name, ticker, desc, x, y)
    for name, x, y in inner_nodes[:-1]:
        draw_inner_node(ax, name, x, y)
    draw_target_node(ax, 'NASDAQ', 15, 8.5)
    if show_legend:
        legend_elements = [
            mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Outer Nodes (Yahoo Finance proxies)'),
            mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Inner Nodes (computed)'),
            mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Target Node (NASDAQ)'),
            plt.Line2D([0], [0], color='green', linewidth=3, label='Positive weight'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Negative weight'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.text(10, 13.5, 'FCM Node Structure & Proxy Assignments', ha='center', fontsize=16, fontweight='bold')
    ax.text(10, 13, 'Blue boxes = Input nodes with Yahoo Finance ticker proxies', ha='center', fontsize=10)
    proxy_text = """
PROXY ISSUES:
• XLI used for BOTH Supply_Increase AND Business_Investment
• Most proxies are stock prices, not economic indicators
• Low_Unemployment has NO proxy (defaults to 0.5)
• All data is same-day (no prediction lag)
"""
    ax.text(0.5, 2.5, proxy_text, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# draw base layout on static side
draw_base_layout(ax_static, show_legend=True)

# -------------------------
# --- Connections 
# -------------------------
connections = [
    # To Monetary Policy (green = positive)
    (1.5, 12, 8, 10, 0.7, 'green'),      # Large_Deficits
    (1.5, 10, 8, 10, 0.5, 'green'),      # Foreign_Demand
    (1.5, 8, 8, 10, 0.6, 'green'),       # Low_Unemployment

    # To Inflation
    (8, 10, 8, 7, 0.8, 'green'),         # Monetary_Policy
    (1.5, 8, 8, 7, 0.6, 'green'),        # Low_Unemployment
    (1.5, 6, 8, 7, 0.6, 'green'),        # Supply_Increase
    (1.5, 4, 8, 7, 0.7, 'green'),        # Energy_Price_Increase
    (6, 1, 8, 7, 0.5, 'green'),          # Consumer_Demand

    # To Corporate Earnings
    (9, 1, 12, 10, 0.7, 'green'),        # Business_Investment
    (8, 7, 12, 10, 0.4, 'green'),        # Inflation
    (6, 1, 12, 10, 0.6, 'green'),        # Consumer_Demand
    (8, 10, 12, 10, -0.6, 'red'),        # Monetary_Policy (negative)

    # To Investor Sentiment
    (12, 10, 12, 7, 0.8, 'green'),       # Corporate_Earnings
    (8, 7, 12, 7, -0.5, 'red'),          # Inflation (negative)
    (12, 1, 12, 7, 0.3, 'green'),        # Market_Volatility
    (15, 1, 12, 7, 0.7, 'green'),        # Equity_Inflows

    # To NASDAQ
    (12, 7, 15, 8.5, 0.9, 'green'),      # Investor_Sentiment
    (12, 10, 15, 8.5, 0.8, 'green'),     # Corporate_Earnings
    (8, 7, 15, 8.5, -0.4, 'red'),        # Inflation (negative)
    (8, 10, 15, 8.5, -0.5, 'red'),       # Monetary_Policy (negative)
    (18, 4, 15, 8.5, 0.7, 'green'),      # AI_Technology
]

# Build pos->name mapping for convenience
pos_to_name = {}
for name, _, _, x, y in outer_nodes:
    pos_to_name[(x, y)] = name
for name, x, y in inner_nodes:
    pos_to_name[(x, y)] = name

# Build static weights dict (from connections)
weights_static_from_connections = {}
for x1, y1, x2, y2, weight, _ in connections:
    src = pos_to_name.get((x1, y1))
    tgt = pos_to_name.get((x2, y2))
    if src and tgt:
        weights_static_from_connections.setdefault(src, {})[tgt] = weight

# Draw arrows on static axis using original connection weights (exact parity)
def draw_arrows_on_axis(ax, weights_dict_local, annotate_weights=True):
    for x1, y1, x2, y2, weight_orig, _ in connections:
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        offset1 = 1.0 if x1 < 5 else 0.9
        offset2 = 1.1 if (x2, y2) == (15, 8.5) else 0.9
        start_x = x1 + (dx/dist) * offset1
        start_y = y1 + (dy/dist) * offset1
        end_x = x2 - (dx/dist) * offset2
        end_y = y2 - (dy/dist) * offset2

        src = pos_to_name.get((x1, y1))
        tgt = pos_to_name.get((x2, y2))
        w = weight_orig
        if src and tgt:
            w = weights_dict_local.get(src, {}).get(tgt, weight_orig)

        color = 'green' if w >= 0 else 'red'
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=max(0.5, abs(w)*3.0), alpha=0.8))
        if annotate_weights:
            mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            ax.text(mid_x, mid_y, f'{w:.2f}', fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.8))

draw_arrows_on_axis(ax_static, weights_static_from_connections, annotate_weights=True)

plt.savefig('fcm_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved static diagram to fcm_structure.png")

# -------------------------
# --- Data fetching (robust)
# -------------------------
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=5*365 + 100)).strftime('%Y-%m-%d')

tickers = {
    'Large_Deficits': '^IRX',
    'Foreign_Demand': 'UUP',
    'Supply_Increase': 'XLI',
    'Energy_Price_Increase': 'XLE',
    'Consumer_Demand': 'XLY',
    'Business_Investment': 'XLI',
    'Market_Volatility': '^VIX',
    'Equity_Inflows': 'SPY',
    'AI_Technology': 'NVDA',
    'NASDAQ': '^IXIC'  # real target
}

print("Fetching data...")
tickers_list = list(set(tickers.values()))
try:
    raw_all = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True, progress=False)
except Exception as e:
    print("yfinance download error:", e)
    raw_all = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))

# Normalize download result to a single-level series of prices
# If yfinance returned a DataFrame with MultiIndex columns choose 'Close' or auto_adjust ensured Close exists
if isinstance(raw_all.columns, pd.MultiIndex):
    # prefer 'Close' if available in top-level, else try to collapse second level
    if 'Close' in raw_all.columns.levels[0]:
        raw = raw_all['Close']
    else:
        # Collapse by taking top-level first available price column
        raw = raw_all.xs(raw_all.columns.levels[0][0], axis=1, level=0, drop_level=True)
elif 'Close' in raw_all.columns:
    raw = raw_all['Close']
elif 'Adj Close' in raw_all.columns:
    raw = raw_all['Adj Close']
else:
    # fallback: if raw_all itself is single column series or similar, try to use it
    raw = raw_all.copy()

# ensure all tickers exist as columns
for t in tickers_list:
    if t not in raw.columns:
        raw[t] = np.nan

raw = raw.sort_index().ffill().bfill()

# compute returns; use fill_method=None to avoid FutureWarning
returns = raw.pct_change(fill_method=None).fillna(0)

# Build input proxies (safe)
inputs = pd.DataFrame(index=raw.index, columns=[n for n in node_order[:10]], dtype=float)

def safe_series(sym, default=0.5):
    if sym in raw.columns:
        return raw[sym].copy()
    else:
        print(f"Warning: ticker {sym} missing — using constant proxy {default}")
        return pd.Series(default, index=raw.index)

def safe_return(sym):
    if sym in returns.columns:
        return returns[sym].copy()
    else:
        print(f"Warning: returns for {sym} missing — using 0.0")
        return pd.Series(0.0, index=returns.index)

inputs['Large_Deficits'] = safe_series('^IRX') / 100.0
inputs['Foreign_Demand'] = safe_return('UUP')
inputs['Low_Unemployment'] = 0.5
inputs['Supply_Increase'] = safe_return('XLI')
inputs['Energy_Price_Increase'] = safe_return('XLE')
inputs['Consumer_Demand'] = safe_return('XLY')
inputs['Business_Investment'] = safe_return('XLI')
inputs['Market_Volatility'] = safe_series('^VIX') / 50.0
inputs['Equity_Inflows'] = safe_return('SPY')
inputs['AI_Technology'] = safe_return('NVDA')

# Normalize to [0,1]
inputs_norm = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
inputs_norm = inputs_norm.clip(0, 1).fillna(0.5)

# NEW: For Google (Polygon) proxies
google_ticker_map = {
    '^IRX': 'I:IRX',
    'UUP': 'UUP',
    'XLI': 'XLI',
    'XLE': 'USO',
    'XLY': 'XLY',
    '^VIX': 'I:VIX',
    'SPY': 'SPY',
    'NVDA': 'NVDA',
}

def fetch_data_polygon(ticker, start, end):
    if POLYGON_AVAILABLE and RESTClient is not None:
        try:
            client = RESTClient()
            start_ts = int(pd.to_datetime(start).timestamp() * 1000)
            end_ts = int(pd.to_datetime(end).timestamp() * 1000)
            aggs = client.get_aggs(ticker, 1, "day", start_ts, end_ts)
            if not aggs:
                raise RuntimeError("Polygon returned no aggregates")
            data = {'Adj Close': [a.close for a in aggs]}
            index = [pd.to_datetime(a.timestamp, unit='ms').date() for a in aggs]
            df = pd.DataFrame(data, index=index)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Polygon fetch failed ({e}); falling back to yfinance for {ticker}.")
    # Fallback to yfinance
    try:
        d = yf.download(ticker, start=start, end=end, progress=False)
        if 'Adj Close' not in d.columns:
            d['Adj Close'] = d['Close']
        return d[['Adj Close']]
    except Exception as e:
        print(f"yfinance fallback failed ({e}). Returning constant 0.5.")
        idx = pd.date_range(start, end, freq='B')
        return pd.DataFrame({'Adj Close': [0.5] * len(idx)}, index=idx)

def safe_series_google(sym, default=0.5):
    sym_g = google_ticker_map.get(sym, sym)
    d = fetch_data_polygon(sym_g, start_date, end_date)
    return d['Adj Close'].reindex(raw.index).ffill().bfill()

def safe_return_google(sym):
    series = safe_series_google(sym)
    return series.pct_change().fillna(0)

inputs_google = pd.DataFrame(index=raw.index, columns=[n for n in node_order[:10]], dtype=float)
inputs_google['Large_Deficits'] = safe_series_google('^IRX') / 100.0
inputs_google['Foreign_Demand'] = safe_return_google('UUP')
inputs_google['Low_Unemployment'] = 0.5
inputs_google['Supply_Increase'] = safe_return_google('XLI')
inputs_google['Energy_Price_Increase'] = safe_return_google('XLE')
inputs_google['Consumer_Demand'] = safe_return_google('XLY')
inputs_google['Business_Investment'] = safe_return_google('XLI')
inputs_google['Market_Volatility'] = safe_series_google('^VIX') / 50.0
inputs_google['Equity_Inflows'] = safe_return_google('SPY')
inputs_google['AI_Technology'] = safe_return_google('NVDA')

inputs_norm_google = (inputs_google - inputs_google.min()) / (inputs_google.max() - inputs_google.min() + 1e-8)
inputs_norm_google = inputs_norm_google.clip(0, 1).fillna(0.5)

# Actual NASDAQ returns & normalized actual used as target
actual_returns = returns['^IXIC'] if '^IXIC' in returns.columns else pd.Series(0.0, index=returns.index)
actual_norm = (actual_returns - actual_returns.min()) / (actual_returns.max() - actual_returns.min() + 1e-8)

# -------------------------
# --- Weights dict from connections (static baseline)
# -------------------------
weights_dict = {}
for x1, y1, x2, y2, weight, _ in connections:
    src = pos_to_name.get((x1, y1))
    tgt = pos_to_name.get((x2, y2))
    if src and tgt:
        weights_dict.setdefault(src, {})[tgt] = weight

# -------------------------
# --- Numeric helpers
# -------------------------
def sigmoid(x, c=2.0):
    x = np.clip(np.asarray(x, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-c * x))

def compute_state_from_outer(outer_series, wdict, node_order_local, max_iter=60, tol=1e-5):
    # outer_series: pandas Series of the outer nodes (10)
    state = {n: 0.5 for n in node_order_local}
    for k in outer_series.index:
        state[k] = float(outer_series[k])
    inner_idx = ['Monetary_Policy', 'Inflation', 'Corporate_Earnings', 'Investor_Sentiment', 'NASDAQ']
    for _ in range(max_iter):
        prev_state = state.copy()
        for tgt in inner_idx:
            s = 0.0
            for src in node_order_local:
                s += wdict.get(src, {}).get(tgt, 0.0) * state[src]
            state[tgt] = float(sigmoid(s, c=2.0))
        if max(abs(prev_state[n] - state[n]) for n in inner_idx) < tol:
            break
    return state

# -------------------------
# --- Static activations (exact parity)
# -------------------------
activations = pd.DataFrame(index=inputs_norm.index, columns=node_order, dtype=float)
activations.update(inputs_norm)

print("Computing static activations...")
for i in range(1, len(activations)):
    prev_outer = inputs_norm.iloc[i-1]
    state = compute_state_from_outer(prev_outer, weights_dict, node_order)
    activations.loc[activations.index[i]] = pd.Series(state)

activations['NASDAQ_pred'] = activations['NASDAQ']
valid_idx = activations.index[1:]
static_pred = activations['NASDAQ_pred'].loc[valid_idx]
static_pred_norm = (static_pred - static_pred.min()) / (static_pred.max() - static_pred.min() + 1e-8)

static_corr = static_pred_norm.corr(actual_norm.loc[valid_idx])
static_mse = ((static_pred_norm - actual_norm.loc[valid_idx]) ** 2).mean()
print(f"\nStatic FCM metrics (lagged): corr={static_corr:.4f}, normalized MSE={static_mse:.6f}")

# -------------------------
# --- Dynamic FCM (online updates)
# -------------------------
dynamic_weights = copy.deepcopy(weights_dict)
learning_rate = 0.02
weight_clip = 3.0

dates = list(inputs_norm.index)
n_steps = len(dates)
max_frames = min(200, n_steps)
skip = max(1, n_steps // max_frames)

weights_history = []
pred_history = []
date_history = []

print("Running online updates to build dynamic weight history...")
for idx in range(1, n_steps):
    date = dates[idx]
    prev_date = dates[idx-1]
    outer_prev = inputs_norm.loc[prev_date]

    # compute state using current dynamic_weights (so updates affect future)
    state = compute_state_from_outer(outer_prev, dynamic_weights, node_order)
    pred = state['NASDAQ']

    # target using normalized actual at current date (if exists)
    target = float(actual_norm.loc[date]) if date in actual_norm.index else 0.0

    error = target - pred
    grad = pred * (1 - pred)
    delta = error * grad

    # update incoming NASDAQ weights only
    for src in node_order:
        if src in dynamic_weights and 'NASDAQ' in dynamic_weights[src]:
            dynamic_weights[src]['NASDAQ'] += learning_rate * delta * state[src]
            dynamic_weights[src]['NASDAQ'] = np.clip(dynamic_weights[src]['NASDAQ'], -weight_clip, weight_clip)

    # snapshot
    if (idx % skip) == 0 or idx == n_steps - 1:
        weights_history.append(copy.deepcopy(dynamic_weights))
        pred_history.append(pred)
        date_history.append(date)

# replay dynamic to produce dyn_series aligned for metrics
dynamic_weights_replay = copy.deepcopy(weights_dict)
dyn_series = pd.Series(index=inputs_norm.index, dtype=float)
for idx in range(1, n_steps):
    date = dates[idx]
    prev_date = dates[idx-1]
    state = compute_state_from_outer(inputs_norm.loc[prev_date], dynamic_weights_replay, node_order)
    dyn_series.loc[date] = state['NASDAQ']
    target = float(actual_norm.loc[date]) if date in actual_norm.index else 0.0
    error = target - state['NASDAQ']
    grad = state['NASDAQ'] * (1 - state['NASDAQ'])
    delta = error * grad
    for src in node_order:
        if src in dynamic_weights_replay and 'NASDAQ' in dynamic_weights_replay[src]:
            dynamic_weights_replay[src]['NASDAQ'] += learning_rate * delta * state[src]
            dynamic_weights_replay[src]['NASDAQ'] = np.clip(dynamic_weights_replay[src]['NASDAQ'], -weight_clip, weight_clip)

dyn_norm = (dyn_series - dyn_series.min()) / (dyn_series.max() - dyn_series.min() + 1e-8)
valid_idx_dyn = dyn_norm.dropna().index.intersection(valid_idx)
dynamic_corr = dyn_norm.loc[valid_idx_dyn].corr(actual_norm.loc[valid_idx_dyn])
dynamic_mse = ((dyn_norm.loc[valid_idx_dyn] - actual_norm.loc[valid_idx_dyn]) ** 2).mean()
print(f"Dynamic FCM Yahoo metrics (lagged): corr={dynamic_corr:.4f}, normalized MSE={dynamic_mse:.6f}")

# NEW: Dynamic FCM for Google
dynamic_weights_google = copy.deepcopy(weights_dict)

weights_history_google = []
pred_history_google = []
date_history_google = []

print("Running online updates to build dynamic Google weight history...")
for idx in range(1, n_steps):
    date = dates[idx]
    prev_date = dates[idx-1]
    outer_prev = inputs_norm_google.loc[prev_date]

    state = compute_state_from_outer(outer_prev, dynamic_weights_google, node_order)
    pred = state['NASDAQ']

    target = float(actual_norm.loc[date]) if date in actual_norm.index else 0.0

    error = target - pred
    grad = pred * (1 - pred)
    delta = error * grad

    for src in node_order:
        if src in dynamic_weights_google and 'NASDAQ' in dynamic_weights_google[src]:
            dynamic_weights_google[src]['NASDAQ'] += learning_rate * delta * state[src]
            dynamic_weights_google[src]['NASDAQ'] = np.clip(dynamic_weights_google[src]['NASDAQ'], -weight_clip, weight_clip)

    if (idx % skip) == 0 or idx == n_steps - 1:
        weights_history_google.append(copy.deepcopy(dynamic_weights_google))
        pred_history_google.append(pred)
        date_history_google.append(date)

# replay for Google
dynamic_weights_replay_google = copy.deepcopy(weights_dict)
dyn_series_google = pd.Series(index=inputs_norm_google.index, dtype=float)
for idx in range(1, n_steps):
    date = dates[idx]
    prev_date = dates[idx-1]
    state = compute_state_from_outer(inputs_norm_google.loc[prev_date], dynamic_weights_replay_google, node_order)
    dyn_series_google.loc[date] = state['NASDAQ']
    target = float(actual_norm.loc[date]) if date in actual_norm.index else 0.0
    error = target - state['NASDAQ']
    grad = state['NASDAQ'] * (1 - state['NASDAQ'])
    delta = error * grad
    for src in node_order:
        if src in dynamic_weights_replay_google and 'NASDAQ' in dynamic_weights_replay_google[src]:
            dynamic_weights_replay_google[src]['NASDAQ'] += learning_rate * delta * state[src]
            dynamic_weights_replay_google[src]['NASDAQ'] = np.clip(dynamic_weights_replay_google[src]['NASDAQ'], -weight_clip, weight_clip)

dyn_norm_google = (dyn_series_google - dyn_series_google.min()) / (dyn_series_google.max() - dyn_series_google.min() + 1e-8)
valid_idx_dyn_google = dyn_norm_google.dropna().index.intersection(valid_idx)
dynamic_corr_google = dyn_norm_google.loc[valid_idx_dyn_google].corr(actual_norm.loc[valid_idx_dyn_google])
dynamic_mse_google = ((dyn_norm_google.loc[valid_idx_dyn_google] - actual_norm.loc[valid_idx_dyn_google]) ** 2).mean()
print(f"Dynamic FCM Google metrics (lagged): corr={dynamic_corr_google:.4f}, normalized MSE={dynamic_mse_google:.6f}")

# -------------------------
# --- Prepare dynamic panel drawing utilities
# -------------------------
def draw_nodes_on_axis(ax):
    for name, ticker, desc, x, y in outer_nodes:
        box = mpatches.FancyBboxPatch((x-1.4, y-0.6), 2.8, 1.2,
                                       boxstyle="round,pad=0.05",
                                       facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.2, name.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, y-0.35, f'{ticker}', ha='center', va='center', fontsize=7, color='darkblue')
    for name, x, y in inner_nodes[:-1]:
        circle = plt.Circle((x, y), 0.9, facecolor='lightyellow', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')
    circle = plt.Circle((15, 8.5), 1.1, facecolor='lightgreen', edgecolor='darkgreen', linewidth=3)
    ax.add_patch(circle)
    ax.text(15, 8.5, 'NASDAQ', ha='center', va='center', fontsize=12, fontweight='bold')

def draw_arrows_dynamic(ax, wdict, annotate_weights=True):
    # clear and redraw nodes
    ax.cla()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')
    draw_nodes_on_axis(ax)
    for x1, y1, x2, y2, orig_w, _ in connections:
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        offset1 = 1.0 if x1 < 5 else 0.9
        offset2 = 1.1 if (x2, y2) == (15, 8.5) else 0.9
        start_x = x1 + (dx/dist) * offset1
        start_y = y1 + (dy/dist) * offset1
        end_x = x2 - (dx/dist) * offset2
        end_y = y2 - (dy/dist) * offset2

        src = pos_to_name.get((x1, y1))
        tgt = pos_to_name.get((x2, y2))
        w = orig_w
        if src and tgt:
            w = wdict.get(src, {}).get(tgt, orig_w)

        color = 'green' if w >= 0 else 'red'
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=max(0.5, abs(w)*3.0), alpha=0.85))
        if annotate_weights:
            mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            ax.text(mid_x, mid_y, f'{w:.2f}', fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.9))
    return ax

# initialize dynamic panel with first snapshot (or baseline)
initial_weights = weights_history[0] if weights_history else weights_dict
draw_arrows_dynamic(ax_dyn_yahoo, initial_weights)
ax_dyn_yahoo.text(10, 13.2, 'Dynamic FCM Yahoo (weights update over time)', ha='center', fontsize=12, fontweight='bold')

# NEW: Initialize Google dynamic panel
initial_weights_google = weights_history_google[0] if weights_history_google else weights_dict
draw_arrows_dynamic(ax_dyn_google, initial_weights_google)
ax_dyn_google.text(10, 13.2, 'Dynamic FCM Google (weights update over time)', ha='center', fontsize=12, fontweight='bold')

# -------------------------
# --- Animation function
# -------------------------
def update_frame(idx):
    wdict = weights_history[idx]
    date = date_history[idx] if idx < len(date_history) else None
    draw_arrows_dynamic(ax_dyn_yahoo, wdict, annotate_weights=True)
    txt = f"Dynamic Yahoo weights snapshot\nDate: {date.strftime('%Y-%m-%d') if date is not None else 'N/A'}"
    ax_dyn_yahoo.text(10, 13.2, txt, ha='center', fontsize=12, fontweight='bold')

    # NEW: Update Google panel
    wdict_g = weights_history_google[idx]
    draw_arrows_dynamic(ax_dyn_google, wdict_g, annotate_weights=True)
    txt_g = f"Dynamic Google weights snapshot\nDate: {date.strftime('%Y-%m-%d') if date is not None else 'N/A'}"
    ax_dyn_google.text(10, 13.2, txt_g, ha='center', fontsize=12, fontweight='bold')

# NEW: Use min frames for both histories
frames = min(len(weights_history), len(weights_history_google))
if frames == 0:
    print("No dynamic weight history produced; skipping animation.")
else:
    ani = FuncAnimation(fig, update_frame, frames=frames, interval=300, repeat=False)

# -------------------------
# --- Final console outputs + show
# -------------------------
print("\nFINAL METRICS:")
print(f"Static FCM corr: {static_corr:.4f}, static MSE: {static_mse:.6f}")
print(f"Dynamic Yahoo FCM corr: {dynamic_corr:.4f}, dynamic MSE: {dynamic_mse:.6f}")
print(f"Dynamic Google FCM corr: {dynamic_corr_google:.4f}, dynamic MSE: {dynamic_mse_google:.6f}")

final_w = weights_history[-1] if weights_history else weights_dict
print("\nFinal incoming NASDAQ weights (dynamic Yahoo snapshot):")
for src in node_order:
    if src in final_w and 'NASDAQ' in final_w[src]:
        print(f"{src:25s} -> NASDAQ : {final_w[src]['NASDAQ']:.4f}")

# NEW: Print final Google weights
final_w_g = weights_history_google[-1] if weights_history_google else weights_dict
print("\nFinal incoming NASDAQ weights (dynamic Google snapshot):")
for src in node_order:
    if src in final_w_g and 'NASDAQ' in final_w_g[src]:
        print(f"{src:25s} -> NASDAQ : {final_w_g[src]['NASDAQ']:.4f}")

plt.tight_layout()
plt.show()