import os
import sys
import time
import math
import json
import copy
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import nltk
from nltk.tokenize import word_tokenize

# Deep Learning imports
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# Ensure punkt tokenizer
nltk.download('punkt', quiet=True)

# -----------------------
# Basic logging
# -----------------------
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger("fcm_nasdaq")

# -----------------------
# Configuration (editable)
# -----------------------
CONFIG = {
    "start_date": None,  # set to None => default to 5 years back
    "end_date": None,    # set to None => default to today
    "lookback_days_for_norm": 252,  # rolling window for normalization (1 year)
    "max_news_pages": 100,  # pagination limit when pulling news
    "sentiment_shift_days": 1,  # shift news forward by 1 trading day by default
    "polygon_api_key_env": "POLYGON_API_KEY",
    "newsapi_key_env": "NEWSAPI_KEY",
    "news_query": "NASDAQ OR stock market OR technology OR AI",
    "lexicon_urls": [
        "https://drive.google.com/uc?export=download&id=1cfg_w3USlRFS97wo7XQmYnuzhpmzboAY",
    ],
    "fallback_positive": ["rise","gain","positive","bull","growth","strong","beat","surge","rally","profit","boost","record","high","better","outperform"],
    "fallback_negative": ["fall","loss","negative","bear","decline","down","weak","miss","drop","plunge","cut","warn","risk","low","worse","underperform"],
    # node->tickers mapping: outer node names map to either a single ticker or a list of tickers to aggregate
    "node_ticker_map": {
        "Large_Deficits": ["^IRX"],
        "Foreign_Demand": ["UUP"],
        "Low_Unemployment": [],  # no ticker: will default to constant
        "Supply_Increase": ["XLI"],
        "Energy_Price_Increase": ["XLE"],
        "Consumer_Demand": ["XLY"],
        "Business_Investment": ["XLI"],
        "Market_Volatility": ["^VIX"],
        "Equity_Inflows": ["SPY"],
        "AI_Technology": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"],
        # News_Sentiment_Yahoo and News_Sentiment_Google handled separately (source-specific)
    },
    "target_ticker": "^IXIC",
}

# -----------------------
# Utilities: date window
# -----------------------
def get_date_range(start_date=None, end_date=None, years_back=5, extra_days=100):
    if end_date is None:
        end = datetime.now(timezone.utc).date()
    else:
        end = pd.to_datetime(end_date).date()
    if start_date is None:
        start = end - timedelta(days=years_back*365 + extra_days)
    else:
        start = pd.to_datetime(start_date).date()
    return start.isoformat(), end.isoformat()

# -----------------------
# Lexicon loader (robust)
# -----------------------
def load_loughran_lexicon(try_urls=None):
    """
    Attempt to download a Loughran–McDonald style lexicon.
    Return two sets: positive_words, negative_words.
    Falls back to CONFIG fallback lists if download fails.
    """
    if try_urls is None:
        try_urls = CONFIG["lexicon_urls"]
    for url in try_urls:
        try:
            logger.info(f"Attempting to download lexicon from {url}")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                text = r.text
                # Attempt to parse CSV; we only need the word column and positive/negative columns if present
                csvf = StringIO(text)
                df = pd.read_csv(csvf, dtype=str, keep_default_na=False, na_values=[])
                # heuristics: find column that looks like "Word" or first column;
                word_col = None
                pos_col = None
                neg_col = None
                for c in df.columns:
                    lc = c.strip().lower()
                    if "word" in lc or "token" in lc or "term" in lc:
                        word_col = c
                    if "positive" in lc or "pos" == lc:
                        pos_col = c
                    if "negative" in lc or "neg" == lc:
                        neg_col = c
                if word_col is None:
                    word_col = df.columns[0]
                positive = set()
                negative = set()
                for _, row in df.iterrows():
                    w = str(row[word_col]).strip().lower()
                    if not w:
                        continue
                    # If pos/neg columns exist interpret them, otherwise use simple heuristics
                    try:
                        if pos_col and int(float(row[pos_col])) > 0:
                            positive.add(w)
                        if neg_col and int(float(row[neg_col])) > 0:
                            negative.add(w)
                    except Exception:
                        pass
                if len(positive) == 0 and len(negative) == 0:
                    # fallback: try to parse lists in raw text
                    lines = text.splitlines()
                    for L in lines:
                        parts = L.strip().split()
                        if len(parts) == 1 and parts[0].isalpha():
                            # crude
                            pass
                logger.info(f"Loaded lexicon from {url}: +{len(positive)} / -{len(negative)}")
                if len(positive) == 0:
                    positive = set(CONFIG["fallback_positive"])
                if len(negative) == 0:
                    negative = set(CONFIG["fallback_negative"])
                return positive, negative
            else:
                logger.warning(f"Lexicon download returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.exception(f"Lexicon download failed for {url}: {e}")
    # final fallback
    logger.warning("Lexicon download failed for all sources; using embedded fallback.")
    return set(CONFIG["fallback_positive"]), set(CONFIG["fallback_negative"])

# -----------------------
# Price fetching (yfinance primary, polygon optional)
# -----------------------
POLYGON_KEY = os.getenv(CONFIG["polygon_api_key_env"], None)
POLYGON_AVAILABLE = bool(POLYGON_KEY)

def fetch_prices_yfinance(tickers, start, end, auto_adjust=True):
    """
    Return a DataFrame with 'Adj Close' prices for provided tickers (columns).
    """
    logger.info("Fetching price data from yfinance for %d tickers", len(tickers))
    # yfinance returns MultiIndex columns. Use 'Adj Close' if available.
    raw = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    # Normalize to single-level columns with Adj Close values
    if isinstance(raw.columns, pd.MultiIndex):
        # prefer 'Adj Close' or 'Close' level
        if 'Adj Close' in raw.columns.levels[0]:
            df = raw['Adj Close']
        elif 'Close' in raw.columns.levels[0]:
            df = raw['Close']
        else:
            # take first available second-level
            lvl0 = raw.columns.levels[0][0]
            df = raw.xs(lvl0, axis=1, level=0, drop_level=True)
    else:
        # raw likely contains Adj Close column if single ticker
        if 'Adj Close' in raw.columns:
            df = raw['Adj Close'].to_frame() if raw['Adj Close'].ndim == 1 else raw['Adj Close']
        elif 'Close' in raw.columns:
            df = raw['Close']
        else:
            df = raw.copy()
    # If df is Series (single ticker) make DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    df = df.rename_axis('Date').sort_index()
    return df

# Polygon fetch helper (optional)
def fetch_prices_polygon(ticker, start, end):
    """
    Fetch daily aggregate prices from Polygon for a single ticker.
    Returns DataFrame indexed by date with 'Adj Close' column.
    Note: Polygon requires ms timestamps for some endpoints; library usage can vary.
    """
    try:
        if not POLYGON_AVAILABLE:
            raise RuntimeError("Polygon key not available")
        from polygon.rest import RESTClient
        client = RESTClient(api_key=POLYGON_KEY)
        # polygon accept yyyy-mm-dd format via get_aggs when using timestamp in ms
        start_ts = int(pd.to_datetime(start).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end).timestamp() * 1000)
        aggs = client.get_aggs(ticker, 1, "day", start_ts, end_ts)
        if not aggs:
            raise RuntimeError("Polygon returned no aggregates")
        data = {'Adj Close': [a.close for a in aggs]}
        idx = [pd.to_datetime(a.timestamp, unit='ms').date() for a in aggs]
        df = pd.DataFrame(data, index=pd.to_datetime(idx))
        return df
    except Exception as e:
        logger.debug("Polygon fetch failed (%s) for %s - falling back to yfinance", e, ticker)
        # fallback to yfinance for the same ticker
        return fetch_prices_yfinance([ticker], start, end)

def fetch_prices_google(tickers, start, end):
    if not POLYGON_AVAILABLE:
        logger.info("Polygon not available; using yfinance for Google prices")
        return fetch_prices_yfinance(tickers, start, end)
    logger.info("Fetching price data from Polygon for %d tickers", len(tickers))
    price_dict = {}
    for t in tickers:
        d = fetch_prices_polygon(t, start, end)
        price_dict[t] = d['Adj Close']
    df = pd.DataFrame(price_dict).rename_axis('Date').sort_index()
    return df

# -----------------------
# Bulk news fetchers
# -----------------------
NEWSAPI_KEY = os.getenv(CONFIG["newsapi_key_env"], None)
HAS_NEWSAPI = bool(NEWSAPI_KEY)

def fetch_news_newsapi(query, start, end, page_size=100, max_pages=5, domains=None):
    """
    Fetch news using NewsAPI historical endpoints (if available).
    Returns a list of dicts with keys: title, description, publishedAt, url, source.
    """
    if not HAS_NEWSAPI:
        logger.debug("NewsAPI key not found; skipping NewsAPI fetch for query=%s", query)
        return []
    results = []
    base = "https://newsapi.org/v2/everything"
    headers = {"Authorization": NEWSAPI_KEY}
    # NewsAPI free tier has limitations — be conservative; page through a few pages
    for page in range(1, min(max_pages, CONFIG["max_news_pages"]) + 1):
        params = {
            "q": query,
            "from": start,
            "to": end,
            "language": "en",
            "pageSize": min(page_size, 100),
            "page": page,
            "sortBy": "publishedAt",
        }
        if domains:
            params['domains'] = domains
        try:
            r = requests.get(base, params=params, headers=headers, timeout=10)
            if r.status_code != 200:
                logger.warning("NewsAPI returned %d: %s", r.status_code, r.text[:200])
                break
            payload = r.json()
            articles = payload.get("articles", [])
            if not articles:
                break
            for a in articles:
                results.append({
                    "title": a.get("title") or "",
                    "description": a.get("description") or "",
                    "publishedAt": a.get("publishedAt"),
                    "url": a.get("url"),
                    "source": a.get("source", {}).get("name")
                })
            if len(articles) < params["pageSize"]:
                break
            time.sleep(0.2)
        except Exception as e:
            logger.exception("NewsAPI fetch error: %s", e)
            break
    logger.info("NewsAPI fetched %d articles for query=%s domains=%s", len(results), query, domains)
    return results

def fetch_news_polygon_bulk(tickers_or_query_list, start, end, per_ticker_limit=200):
    """
    If Polygon is available, fetch news for multiple tickers in bulk (paginated) and return list.
    This is written to use polygon REST API if installed. Falls back to [].
    """
    if not POLYGON_AVAILABLE:
        logger.debug("Polygon key not available - skipping polygon news fetch.")
        return []
    try:
        from polygon.rest import RESTClient
        client = RESTClient(api_key=POLYGON_KEY)
        all_items = []
        for t in tickers_or_query_list:
            # polygon: list_ticker_news expects a ticker (not ^IXIC) — call per ticker but with a batched approach
            # Note: polygon rate limits apply, perform limited calls
            try:
                aggr = client.list_ticker_news(t, published_utc_gte=start, published_utc_lte=end, limit=per_ticker_limit)
                for n in aggr:
                    all_items.append({
                        "title": getattr(n, 'title', '') or '',
                        "description": getattr(n, 'description', '') or '',
                        "publishedAt": getattr(n, 'published_utc', None),
                        "url": getattr(n, 'article_url', None),
                        "source": getattr(n, 'publisher', None)
                    })
                time.sleep(0.05)
            except Exception as e:
                logger.debug("Polygon news fetch fail for %s: %s", t, e)
                continue
        logger.info("Polygon news fetch returned %d articles across %d tickers", len(all_items), len(tickers_or_query_list))
        return all_items
    except Exception as e:
        logger.exception("Polygon news bulk error: %s", e)
        return []

# -----------------------
# Text cleaning & sentiment
# -----------------------
import re
_token_re = re.compile(r"[^\w\s!]")
_excl_re = re.compile(r"!+")

def clean_and_tokenize(text):
    # minimal cleaning preserving exclamation marks and ALL CAPS
    if not text:
        return []
    tmp = text.replace("\n", " ").strip()
    tmp = _token_re.sub(" ", tmp)  # remove punctuation except '!' preserved by not including it
    tokens = word_tokenize(tmp)
    return tokens

def compute_article_sentiment(tokens, pos_set, neg_set):
    """
    Compute a score that preserves polarity. Returns (polarity_score, intensity_sum, token_count).
    polarity_score is sum(pos*intensity - neg*intensity) / token_count (range roughly unconstrained),
    we'll scale later using hyperbolic tangent to [-1,1].
    """
    pos_sum = 0.0
    neg_sum = 0.0
    tokens_count = 0
    for t in tokens:
        if not t.strip():
            continue
        intensity = 1.0
        if t.isupper() and len(t) > 1:  # ALL CAPS signal
            intensity *= 1.5
        excl_count = _excl_re.findall(t)
        if excl_count:
            # if token itself contains exclamation marks (rare), increase intensity
            intensity *= 1.0 + 0.15 * sum(len(x) for x in excl_count)
        tok_clean = _excl_re.sub("", t).lower()
        if tok_clean in pos_set:
            pos_sum += intensity
        elif tok_clean in neg_set:
            neg_sum += intensity
        tokens_count += 1
    raw = pos_sum - neg_sum
    return raw, (pos_sum + neg_sum), tokens_count

def aggregate_daily_sentiment(articles, pos_set, neg_set, date_shift_days=1, decay=0.8):
    """
    articles: list of dicts with 'title', 'description', 'publishedAt' (ISO str)
    Returns: pandas.Series indexed by date with polarity in [-1,1] and intensity series
    Steps:
      - dedupe by url/title
      - compute per-article raw polarity and tokens
      - aggregate by date: sum raw polarity and total tokens
      - compute daily_raw = sum_raw / max(1,total_tokens)
      - running_score = daily_raw + decay * previous_running
      - polarity = tanh(running_score * 5.0)
      - shift forward by date_shift_days to model market effect next day
    """
    # Deduplicate by url/title
    seen = set()
    grouped = defaultdict(list)
    for a in articles:
        key = (a.get("url") or "")[:200] + "::" + (a.get("title") or "")[:200]
        if key in seen:
            continue
        seen.add(key)
        # parse published date day
        p = a.get("publishedAt")
        try:
            if p is None:
                continue
            dt = pd.to_datetime(p)
            day = dt.date()
        except Exception:
            continue
        text = (a.get("title") or "") + " " + (a.get("description") or "")
        tokens = clean_and_tokenize(text)
        raw, intensity, tcount = compute_article_sentiment(tokens, pos_set, neg_set)
        grouped[pd.to_datetime(day)].append({"raw": raw, "intensity": intensity, "tokens": tcount})
    if not grouped:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rows = []
    running_score = 0.0
    for day in sorted(grouped.keys()):
        items = grouped[day]
        total_raw = sum(i["raw"] for i in items)
        total_tokens = sum(i["tokens"] for i in items)
        total_intensity = sum(i["intensity"] for i in items)
        if total_tokens == 0:
            daily_raw = 0.0
        else:
            daily_raw = total_raw / max(1.0, total_tokens)  # normalized raw polarity
        running_score = daily_raw + decay * running_score
        # scale via tanh to [-1,1] and preserve intensity separately
        polarity = math.tanh(running_score * 5.0)  # scale factor tuned - adjustable
        rows.append((day, polarity, total_intensity, total_tokens))
    df = pd.DataFrame(rows, columns=["date", "polarity", "intensity", "tokens"]).set_index("date").sort_index()
    # shift polarity forward into the market day by default
    if date_shift_days != 0:
        df["polarity"] = df["polarity"].shift(date_shift_days)
        df["intensity"] = df["intensity"].shift(date_shift_days)
    polarity_series = df["polarity"].fillna(0.0)
    intensity_series = df["intensity"].fillna(0.0)
    return polarity_series, intensity_series

# -----------------------
# Build input proxies dynamically
# -----------------------
def build_proxies_from_prices(price_df, node_ticker_map, target_index):
    """
    price_df: DataFrame with Adj Close columns for tickers
    node_ticker_map: mapping node-> list of tickers (possibly empty)
    Returns: inputs DataFrame (outer nodes) and returns Series for target.
    - For nodes with multiple tickers, we aggregate by mean returns.
    - Low_Unemployment (no ticker) returned as constant 0.5
    """
    # compute returns (daily pct change)
    returns = price_df.pct_change().fillna(0)
    inputs = {}
    for node, tickers in node_ticker_map.items():
        if not tickers:
            # no proxy available -> constant (for now)
            inputs[node] = pd.Series(0.5, index=price_df.index)
            continue
        # for multi-ticker nodes compute mean returns series
        available = [t for t in tickers if t in price_df.columns]
        if not available:
            # fallback constant but log
            logger.warning("No price columns available for node %s -> constant 0.5", node)
            inputs[node] = pd.Series(0.5, index=price_df.index)
            continue
        # Use returns except for pure rate tickers like ^IRX where we scale
        if any(t.startswith("^") for t in available) and not any(t == '^VIX' for t in available):
            # treat these as level proxies and scale them (example: ^IRX is percent)
            series_list = []
            for t in available:
                if t in price_df.columns:
                    series_list.append(price_df[t] / 100.0)
            if series_list:
                s = pd.concat(series_list, axis=1).mean(axis=1)
                inputs[node] = s.reindex(price_df.index).ffill(limit=5).bfill(limit=5)
        elif any(t == '^VIX' for t in available):
            # scale VIX
            s = price_df[available[0]] / 50.0
            inputs[node] = s.reindex(price_df.index).ffill(limit=5).bfill(limit=5)
        else:
            # average returns across tickers
            series_list = [returns[t] for t in available if t in returns.columns]
            if series_list:
                mat = pd.concat(series_list, axis=1)
                inputs[node] = mat.mean(axis=1).reindex(price_df.index).fillna(0.0)
            else:
                inputs[node] = pd.Series(0.0, index=price_df.index)
    # News_Sentiment will be added externally as columns per source
    inputs_df = pd.DataFrame(inputs).reindex(price_df.index)
    target = None
    if target_index in returns.columns:
        target = returns[target_index]
    else:
        logger.warning("Target ticker %s not found in returns; target set to zeros", target_index)
        target = pd.Series(0.0, index=price_df.index)
    return inputs_df, target

# -----------------------
# Initial weight estimation (dynamic replacement for static weights)
# -----------------------
def estimate_initial_weights(inputs_df, target_series, node_order, scale=1.0):
    """
    Estimate initial weights from correlation between each outer node and target.
    Returns a dict-of-dicts like weights[src][tgt] for edges into inner nodes (we keep the original topology shape but we
    will initialize incoming edges into inner nodes using correlations).
    NOTE: This is a simple heuristic: correlation -> weight. We keep sign.
    """
    weights = {}
    # compute correlation of each outer node with target (lagged by 1 day: outer at t-1 -> target at t)
    lagged_inputs = inputs_df.shift(1).dropna()
    common_idx = lagged_inputs.index.intersection(target_series.index)
    if common_idx.empty:
        logger.warning("No overlap between lagged inputs and target for weight estimation; using small random weights")
        # small random initialization
        for src in node_order:
            weights[src] = {}
            for tgt in node_order[-5:]:  # assume last 5 are inner/tgt
                weights[src][tgt] = np.random.uniform(-0.1, 0.1)
        return weights
    t = target_series.reindex(common_idx)
    for src in node_order:
        if src in lagged_inputs.columns:
            s = lagged_inputs[src].reindex(common_idx)
            if s.std() == 0 or t.std() == 0:
                corr = 0.0
            else:
                corr = s.corr(t)
            if pd.isna(corr):
                corr = 0.0
            # Scale down initial weights significantly to allow learning
            w_init = float(np.clip(corr * scale * 0.3, -0.5, 0.5))
        else:
            w_init = 0.0
        weights.setdefault(src, {})
        # initialize only edges to inner nodes - we keep full connectivity based on provided connections in visualizer
        # For now we create entries for NASDAQ and inner nodes for safety:
        for tgt in ["Monetary_Policy","Inflation","Corporate_Earnings","Investor_Sentiment","NASDAQ"]:
            weights[src][tgt] = w_init * np.random.uniform(0.8,1.2)  # small perturbation
    logger.info("Estimated initial incoming weights from correlations (sample): %s", {k: list(v.items())[:1] for k,v in list(weights.items())[:3]})
    return weights

# -----------------------
# FCM solver + dynamic learning (kept from your original, minimally changed)
# -----------------------
def sigmoid(x, c=2.0):
    x = np.clip(np.asarray(x, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-c * x))

def compute_state_from_outer(outer_series, wdict, node_order_local, max_iter=60, tol=1e-5):
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

def run_online_updates(inputs_norm, weights_init, actual_norm, node_order, learning_rate=0.02, weight_clip=1.0, max_frames=200):
    """
    Run your online update algorithm but using dynamic, estimated weights_init.
    Returns: weights_history (list of dict snapshots), pred_history (NASDaq preds), date_history
    """
    dynamic_weights = copy.deepcopy(weights_init)
    dates = list(inputs_norm.index)
    n_steps = len(dates)
    skip = max(1, n_steps // max_frames)
    weights_history = []
    pred_history = []
    date_history = []
    for idx in range(1, n_steps):
        date = dates[idx]
        prev_date = dates[idx-1]
        outer_prev = inputs_norm.loc[prev_date]
        state = compute_state_from_outer(outer_prev, dynamic_weights, node_order)
        pred = state['NASDAQ']
        target = float(actual_norm.loc[date]) if date in actual_norm.index else 0.0
        error = target - pred
        grad = pred * (1 - pred)
        delta = error * grad
        # update incoming NASDAQ weights only
        for src in node_order:
            if src in dynamic_weights and 'NASDAQ' in dynamic_weights[src]:
                dynamic_weights[src]['NASDAQ'] += learning_rate * delta * state[src]
                dynamic_weights[src]['NASDAQ'] = np.clip(dynamic_weights[src]['NASDAQ'], -weight_clip, weight_clip)
        # Backpropagate to previous layer
        direct_preds = [src for src in node_order if src in dynamic_weights and 'NASDAQ' in dynamic_weights[src] and dynamic_weights[src]['NASDAQ'] != 0]
        for pred_name in direct_preds:
            error_pred = delta * dynamic_weights[pred_name]['NASDAQ']
            grad_pred = state[pred_name] * (1 - state[pred_name])
            delta_pred = error_pred * grad_pred
            for src in node_order:
                if src in dynamic_weights and pred_name in dynamic_weights[src]:
                    dynamic_weights[src][pred_name] += learning_rate * delta_pred * state[src]
                    dynamic_weights[src][pred_name] = np.clip(dynamic_weights[src][pred_name], -weight_clip, weight_clip)
        # snapshot
        if (idx % skip) == 0 or idx == n_steps - 1:
            weights_history.append(copy.deepcopy(dynamic_weights))
            pred_history.append(pred)
            date_history.append(date)
    return weights_history, pred_history, date_history

# -----------------------
# Deep Learning: LSTM Model
# -----------------------
def build_lstm_model(input_shape, output_shape=1, lstm_units=64, dropout=0.2):
    """Build LSTM model for time series prediction."""
    if not KERAS_AVAILABLE:
        logger.warning("TensorFlow/Keras not available; LSTM model skipped")
        return None

    model = Sequential([
        LSTM(lstm_units, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units // 2, activation='relu'),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(output_shape)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def create_lstm_sequences(data, lookback=20):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm(inputs_norm, target_norm, lookback=20, epochs=50, batch_size=32):
    """Train LSTM model on normalized inputs."""
    if not KERAS_AVAILABLE:
        logger.warning("TensorFlow/Keras not available; LSTM training skipped")
        return None, None, None

    # Prepare data: use all input features
    data = inputs_norm.values
    target = target_norm.values

    # Create sequences
    X, _ = create_lstm_sequences(data, lookback)
    _, y = create_lstm_sequences(target, lookback)

    if len(X) == 0:
        logger.warning("Insufficient data for LSTM training")
        return None, None, None

    # Split train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build and train
    model = build_lstm_model((X.shape[1], X.shape[2]), output_shape=1)
    if model is None:
        return None, None, None

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1, verbose=0)

    # Predictions
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()

    lstm_metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train) if SKLEARN_AVAILABLE else 0,
        'test_mae': mean_absolute_error(y_test, y_pred_test) if SKLEARN_AVAILABLE else 0,
        'train_mse': mean_squared_error(y_train, y_pred_train) if SKLEARN_AVAILABLE else 0,
        'test_mse': mean_squared_error(y_test, y_pred_test) if SKLEARN_AVAILABLE else 0,
        'test_r2': r2_score(y_test, y_pred_test) if SKLEARN_AVAILABLE else 0,
    }

    logger.info(f"LSTM Training: Train MAE={lstm_metrics['train_mae']:.6f}, Test MAE={lstm_metrics['test_mae']:.6f}")

    return model, lstm_metrics, (y_test, y_pred_test)

def build_gru_model(input_shape, output_shape=1, gru_units=64, dropout=0.2):
    """Build GRU model for time series prediction (faster alternative to LSTM)."""
    if not KERAS_AVAILABLE:
        logger.warning("TensorFlow/Keras not available; GRU model skipped")
        return None

    from tensorflow.keras.layers import GRU

    model = Sequential([
        GRU(gru_units, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(dropout),
        GRU(gru_units // 2, activation='relu'),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(output_shape)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_gru(inputs_norm, target_norm, lookback=20, epochs=50, batch_size=32):
    """Train GRU model on normalized inputs (faster than LSTM)."""
    if not KERAS_AVAILABLE:
        logger.warning("TensorFlow/Keras not available; GRU training skipped")
        return None, None, None

    # Prepare data: use all input features
    data = inputs_norm.values
    target = target_norm.values

    # Create sequences
    X, _ = create_lstm_sequences(data, lookback)
    _, y = create_lstm_sequences(target, lookback)

    if len(X) == 0:
        logger.warning("Insufficient data for GRU training")
        return None, None, None

    # Split train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build and train
    model = build_gru_model((X.shape[1], X.shape[2]), output_shape=1)
    if model is None:
        return None, None, None

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1, verbose=0)

    # Predictions
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()

    gru_metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train) if SKLEARN_AVAILABLE else 0,
        'test_mae': mean_absolute_error(y_test, y_pred_test) if SKLEARN_AVAILABLE else 0,
        'train_mse': mean_squared_error(y_train, y_pred_train) if SKLEARN_AVAILABLE else 0,
        'test_mse': mean_squared_error(y_test, y_pred_test) if SKLEARN_AVAILABLE else 0,
        'test_r2': r2_score(y_test, y_pred_test) if SKLEARN_AVAILABLE else 0,
    }

    logger.info(f"GRU Training: Train MAE={gru_metrics['train_mae']:.6f}, Test MAE={gru_metrics['test_mae']:.6f}")

    return model, gru_metrics, (y_test, y_pred_test)

def compute_node_correlations(inputs_df, target_series, window=252):
    """
    Compute rolling correlations between all nodes and target over time.
    Returns: dict with correlation snapshots at key time points.
    """
    correlations_history = {}

    # Get indices at regular intervals (quarterly)
    step = max(1, len(inputs_df) // 20)  # 20 snapshots
    snapshot_dates = inputs_df.index[::step]

    for snapshot_date in snapshot_dates:
        # Get data up to this point
        idx = inputs_df.index.get_loc(snapshot_date)
        start_idx = max(0, idx - window)

        data_slice = inputs_df.iloc[start_idx:idx+1]
        target_slice = target_series.iloc[start_idx:idx+1]

        if len(data_slice) < 10:  # Need minimum data
            continue

        # Compute correlations
        correlations = data_slice.corrwith(target_slice)
        correlations_history[snapshot_date] = correlations

    return correlations_history

def ensemble_predictions(lstm_pred, gru_pred, fcm_pred, weights=None):
    """
    Combine predictions from multiple models using weighted averaging.

    Args:
        lstm_pred: LSTM predictions (1D array)
        gru_pred: GRU predictions (1D array)
        fcm_pred: FCM predictions (1D array)
        weights: Optional weights for each model (default: equal weights)

    Returns:
        Ensemble predictions (1D array), accuracy metrics dict
    """
    # Handle None predictions (if models didn't train)
    valid_preds = []
    if lstm_pred is not None:
        valid_preds.append(np.array(lstm_pred).flatten())
    if gru_pred is not None:
        valid_preds.append(np.array(gru_pred).flatten())
    if fcm_pred is not None:
        valid_preds.append(np.array(fcm_pred).flatten())

    if not valid_preds:
        logger.warning("No valid predictions for ensemble")
        return None, {}

    # Ensure all predictions have same length
    min_len = min(len(p) for p in valid_preds)
    valid_preds = [p[-min_len:] for p in valid_preds]  # Align to shortest

    # Compute ensemble (simple averaging)
    ensemble_pred = np.mean(valid_preds, axis=0)

    return ensemble_pred, {
        "num_models": len(valid_preds),
        "ensemble_avg": float(np.mean(ensemble_pred)),
        "ensemble_std": float(np.std(ensemble_pred)),
    }

# -----------------------
# Public high-level runner
# -----------------------
def prepare_data_and_run(config_override=None):
    conf = CONFIG.copy()
    if config_override:
        conf.update(config_override)
    start, end = get_date_range(conf["start_date"], conf["end_date"])
    # price tickers: flatten node_ticker_map + target
    tickers = set()
    for v in conf["node_ticker_map"].values():
        tickers.update(v)
    tickers.add(conf["target_ticker"])
    tickers = sorted([t for t in tickers if t])  # remove empty
    price_df_yahoo = fetch_prices_yfinance(tickers, start, end)
    price_df_google = fetch_prices_google(tickers, start, end)
    # build proxies inputs
    inputs_df_yahoo, target_yahoo = build_proxies_from_prices(price_df_yahoo, conf["node_ticker_map"], conf["target_ticker"])
    inputs_df_google, target_google = build_proxies_from_prices(price_df_google, conf["node_ticker_map"], conf["target_ticker"])
    # load lexicon
    pos_set, neg_set = load_loughran_lexicon()
    # fetch news: try source-specific with NewsAPI if available, else Polygon unified
    polarity_series_yahoo = None
    polarity_series_google = None
    polarity_series = None
    intensity_series = None
    articles = []
    if HAS_NEWSAPI:
        q = conf["news_query"]
        # Try fetching without domain restrictions first (more reliable)
        articles_all = fetch_news_newsapi(q, start, end)
        if articles_all:
            # Simulate split between yahoo and google for dual analysis
            mid = len(articles_all) // 2
            articles_yahoo = articles_all[:mid]
            articles_google = articles_all[mid:]
        else:
            articles_yahoo = []
            articles_google = []
        polarity_series_yahoo, _ = aggregate_daily_sentiment(articles_yahoo, pos_set, neg_set, conf["sentiment_shift_days"])
        polarity_series_google, _ = aggregate_daily_sentiment(articles_google, pos_set, neg_set, conf["sentiment_shift_days"])
        # If both are empty, fall through to create unified sentiment
        if polarity_series_yahoo.empty and polarity_series_google.empty:
            polarity_series_yahoo = None
            polarity_series_google = None

    if polarity_series_yahoo is None or polarity_series_yahoo.empty:
        # fallback: fetch without domain filter or use Polygon
        articles = []
        if HAS_NEWSAPI:
            q = conf["news_query"]
            articles = fetch_news_newsapi(q, start, end)  # No domain restriction
        if not articles and POLYGON_AVAILABLE:
            tickers_for_news = list(set([x for sub in conf["node_ticker_map"].values() for x in sub]))
            articles = fetch_news_polygon_bulk(tickers_for_news, start, end)
        polarity_series, intensity_series = aggregate_daily_sentiment(articles, pos_set, neg_set, conf["sentiment_shift_days"])
        # Create synthetic sentiment if no real data available
        if polarity_series.empty:
            logger.warning("No news articles found; creating synthetic sentiment based on price volatility")
            # Use returns volatility as sentiment proxy
            volatility = target_yahoo.rolling(window=20).std().fillna(0)
            # Normalize to [-1, 1] using tanh
            polarity_series = pd.Series(
                np.tanh((volatility - volatility.mean()) / (volatility.std() + 1e-8)),
                index=target_yahoo.index
            )
    # attach sentiment into inputs
    if polarity_series_yahoo is not None:
        inputs_df_yahoo["News_Sentiment_Yahoo"] = polarity_series_yahoo.reindex(inputs_df_yahoo.index).fillna(0.0)
        inputs_df_google["News_Sentiment_Yahoo"] = polarity_series_yahoo.reindex(inputs_df_google.index).fillna(0.0)
        inputs_df_yahoo["News_Sentiment_Google"] = polarity_series_google.reindex(inputs_df_yahoo.index).fillna(0.0)
        inputs_df_google["News_Sentiment_Google"] = polarity_series_google.reindex(inputs_df_google.index).fillna(0.0)
    else:
        inputs_df_yahoo["News_Sentiment"] = polarity_series.reindex(inputs_df_yahoo.index).fillna(0.0)
        inputs_df_google["News_Sentiment"] = polarity_series.reindex(inputs_df_google.index).fillna(0.0)
    # node order: outer followed by inner
    node_order = list(inputs_df_yahoo.columns) + ['Monetary_Policy','Inflation','Corporate_Earnings','Investor_Sentiment','NASDAQ']
    # estimate initial weights from correlations (use yahoo for estimation)
    weights_init = estimate_initial_weights(inputs_df_yahoo, target_yahoo, node_order, scale=1.0)
    # normalization: use rolling lookback to avoid lookahead
    inputs_norm_yahoo = inputs_df_yahoo.copy()
    inputs_norm_google = inputs_df_google.copy()
    lb = conf["lookback_days_for_norm"]
    for df_norm in [inputs_norm_yahoo, inputs_norm_google]:
        for col in df_norm.columns:
            series = df_norm[col]
            # compute rolling min/max on past data (expand as first values)
            roll_min = series.rolling(window=lb, min_periods=1).min().shift(1).fillna(series.min())
            roll_max = series.rolling(window=lb, min_periods=1).max().shift(1).fillna(series.max())
            denom = (roll_max - roll_min + 1e-12)
            df_norm[col] = ((series - roll_min) / denom).clip(0,1).fillna(0.5)
    # actual target normalized similarly (past-window), use yahoo target
    actual_returns = target_yahoo
    learning_rate = 0.02  # learning rate for weight updates
    roll_min = actual_returns.rolling(window=lb, min_periods=1).min().shift(1).fillna(actual_returns.min())
    roll_max = actual_returns.rolling(window=lb, min_periods=1).max().shift(1).fillna(actual_returns.max())
    # Avoid division by zero and extreme values
    denom = (roll_max - roll_min + 1e-12)
    # Clip denominator to prevent extreme scaling
    denom = denom.clip(lower=1e-6)
    actual_norm = ((actual_returns - roll_min) / denom).fillna(0.0)
    # Clip normalized values to [-5, 5] to remove outliers
    actual_norm = actual_norm.clip(-5, 5)
    # run online updates for yahoo and google
    weights_history_yahoo, pred_history_yahoo, date_history_yahoo = run_online_updates(inputs_norm_yahoo, weights_init, actual_norm, node_order)
    weights_history_google, pred_history_google, date_history_google = run_online_updates(inputs_norm_google, weights_init, actual_norm, node_order)
    # compute static activations (using yahoo)
    activations = pd.DataFrame(index=inputs_norm_yahoo.index, columns=node_order, dtype=float)
    activations.update(inputs_norm_yahoo)
    for i in range(1, len(activations)):
        prev_outer = inputs_norm_yahoo.iloc[i-1]
        state = compute_state_from_outer(prev_outer, weights_init, node_order)
        activations.loc[activations.index[i]] = pd.Series(state)
    static_pred = activations['NASDAQ'].loc[activations.index[1:]]
    static_norm = (static_pred - static_pred.min()) / (static_pred.max() - static_pred.min() + 1e-8)
    valid_idx = static_norm.index.intersection(actual_norm.index)
    static_corr = static_norm.corr(actual_norm.loc[valid_idx])
    static_mse = ((static_norm - actual_norm.loc[valid_idx]) ** 2).mean()
    # dynamic replay for yahoo
    dates = list(inputs_norm_yahoo.index)
    dynamic_weights_replay = copy.deepcopy(weights_init)
    dyn_series_yahoo = pd.Series(index=inputs_norm_yahoo.index, dtype=float)
    for idx in range(1, len(dates)):
        date = dates[idx]
        prev_date = dates[idx-1]
        state = compute_state_from_outer(inputs_norm_yahoo.loc[prev_date], dynamic_weights_replay, node_order)
        dyn_series_yahoo.loc[date] = state['NASDAQ']
        target = float(actual_norm.loc[date]) if date in actual_norm.index else 0.0
        error = target - state['NASDAQ']
        grad = state['NASDAQ'] * (1 - state['NASDAQ'])
        delta = error * grad
        for src in node_order:
            if src in dynamic_weights_replay and 'NASDAQ' in dynamic_weights_replay[src]:
                dynamic_weights_replay[src]['NASDAQ'] += learning_rate * delta * state[src]
                dynamic_weights_replay[src]['NASDAQ'] = np.clip(dynamic_weights_replay[src]['NASDAQ'], -1.0, 1.0)
        direct_preds = [src for src in node_order if src in dynamic_weights_replay and 'NASDAQ' in dynamic_weights_replay[src] and dynamic_weights_replay[src]['NASDAQ'] != 0]
        for pred_name in direct_preds:
            error_pred = delta * dynamic_weights_replay[pred_name]['NASDAQ']
            grad_pred = state[pred_name] * (1 - state[pred_name])
            delta_pred = error_pred * grad_pred
            for src in node_order:
                if src in dynamic_weights_replay and pred_name in dynamic_weights_replay[src]:
                    dynamic_weights_replay[src][pred_name] += learning_rate * delta_pred * state[src]
                    dynamic_weights_replay[src][pred_name] = np.clip(dynamic_weights_replay[src][pred_name], -1.0, 1.0)
    dyn_norm_yahoo = (dyn_series_yahoo - dyn_series_yahoo.min()) / (dyn_series_yahoo.max() - dyn_series_yahoo.min() + 1e-8)
    valid_idx_dyn = dyn_norm_yahoo.dropna().index.intersection(valid_idx)
    dynamic_corr_yahoo = dyn_norm_yahoo.loc[valid_idx_dyn].corr(actual_norm.loc[valid_idx_dyn])
    dynamic_mse_yahoo = ((dyn_norm_yahoo.loc[valid_idx_dyn] - actual_norm.loc[valid_idx_dyn]) ** 2).mean()
    # dynamic replay for google
    dynamic_weights_replay_google = copy.deepcopy(weights_init)
    dyn_series_google = pd.Series(index=inputs_norm_google.index, dtype=float)
    for idx in range(1, len(dates)):
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
                dynamic_weights_replay_google[src]['NASDAQ'] = np.clip(dynamic_weights_replay_google[src]['NASDAQ'], -1.0, 1.0)
        direct_preds = [src for src in node_order if src in dynamic_weights_replay_google and 'NASDAQ' in dynamic_weights_replay_google[src] and dynamic_weights_replay_google[src]['NASDAQ'] != 0]
        for pred_name in direct_preds:
            error_pred = delta * dynamic_weights_replay_google[pred_name]['NASDAQ']
            grad_pred = state[pred_name] * (1 - state[pred_name])
            delta_pred = error_pred * grad_pred
            for src in node_order:
                if src in dynamic_weights_replay_google and pred_name in dynamic_weights_replay_google[src]:
                    dynamic_weights_replay_google[src][pred_name] += learning_rate * delta_pred * state[src]
                    dynamic_weights_replay_google[src][pred_name] = np.clip(dynamic_weights_replay_google[src][pred_name], -1.0, 1.0)
    dyn_norm_google = (dyn_series_google - dyn_series_google.min()) / (dyn_series_google.max() - dyn_series_google.min() + 1e-8)
    valid_idx_dyn_google = dyn_norm_google.dropna().index.intersection(valid_idx)
    dynamic_corr_google = dyn_norm_google.loc[valid_idx_dyn_google].corr(actual_norm.loc[valid_idx_dyn_google])
    dynamic_mse_google = ((dyn_norm_google.loc[valid_idx_dyn_google] - actual_norm.loc[valid_idx_dyn_google]) ** 2).mean()

    # Train LSTM deep learning model
    logger.info("Training LSTM deep learning model...")
    lstm_model, lstm_metrics, lstm_results = train_lstm(inputs_norm_yahoo, actual_norm, lookback=20, epochs=30)
    if lstm_model is None:
        lstm_metrics = {'train_mae': 0, 'test_mae': 0, 'train_mse': 0, 'test_mse': 0, 'test_r2': 0}
        lstm_results = (np.array([]), np.array([]))

    # Train GRU deep learning model (faster alternative to LSTM)
    logger.info("Training GRU deep learning model...")
    gru_model, gru_metrics, gru_results = train_gru(inputs_norm_yahoo, actual_norm, lookback=20, epochs=30)
    if gru_model is None:
        gru_metrics = {'train_mae': 0, 'test_mae': 0, 'train_mse': 0, 'test_mse': 0, 'test_r2': 0}
        gru_results = (np.array([]), np.array([]))

    # Create ensemble predictions from LSTM, GRU, and Yahoo FCM
    logger.info("Creating ensemble predictions...")
    ensemble_pred, ensemble_info = ensemble_predictions(
        lstm_results[1] if lstm_results[0] is not None else None,
        gru_results[1] if gru_results[0] is not None else None,
        dyn_norm_yahoo.values if len(dyn_norm_yahoo) > 0 else None
    )
    if ensemble_pred is not None and len(ensemble_pred) > 0:
        ensemble_corr = pd.Series(ensemble_pred).corr(pd.Series(actual_norm.values[-len(ensemble_pred):]))
        ensemble_mse = np.mean((ensemble_pred - actual_norm.values[-len(ensemble_pred):]) ** 2)
    else:
        ensemble_corr = 0.0
        ensemble_mse = 0.0

    # Create metrics comparison dict
    metrics_comparison = {
        "Yahoo FCM": {"corr": dynamic_corr_yahoo, "mse": dynamic_mse_yahoo},
        "Google FCM": {"corr": dynamic_corr_google, "mse": dynamic_mse_google},
        "LSTM": {"mae": lstm_metrics.get('test_mae', 0), "mse": lstm_metrics.get('test_mse', 0), "r2": lstm_metrics.get('test_r2', 0)},
        "GRU": {"mae": gru_metrics.get('test_mae', 0), "mse": gru_metrics.get('test_mse', 0), "r2": gru_metrics.get('test_r2', 0)},
        "Ensemble": {"corr": ensemble_corr, "mse": ensemble_mse}
    }

    # Compute node correlations over time
    logger.info("Computing historical node correlations...")
    correlations_history_yahoo = compute_node_correlations(inputs_norm_yahoo, actual_norm, window=252)

    # Compute feature importance (correlation with target at end)
    feature_importance = inputs_norm_yahoo.corrwith(actual_norm).abs().sort_values(ascending=False)

    # return
    return {
        "price_df_yahoo": price_df_yahoo,
        "price_df_google": price_df_google,
        "inputs_df_yahoo": inputs_df_yahoo,
        "inputs_df_google": inputs_df_google,
        "inputs_norm_yahoo": inputs_norm_yahoo,
        "inputs_norm_google": inputs_norm_google,
        "actual_returns": actual_returns,
        "actual_norm": actual_norm,
        "weights_init": weights_init,
        "weights_history_yahoo": weights_history_yahoo,
        "weights_history_google": weights_history_google,
        "pred_history_yahoo": pred_history_yahoo,
        "pred_history_google": pred_history_google,
        "date_history_yahoo": date_history_yahoo,
        "date_history_google": date_history_google,
        "polarity_series_yahoo": polarity_series_yahoo,
        "polarity_series_google": polarity_series_google,
        "polarity_series": polarity_series,
        "intensity_series": intensity_series,
        "node_order": node_order,
        "static_corr": static_corr,
        "static_mse": static_mse,
        "dynamic_corr_yahoo": dynamic_corr_yahoo,
        "dynamic_mse_yahoo": dynamic_mse_yahoo,
        "dynamic_corr_google": dynamic_corr_google,
        "dynamic_mse_google": dynamic_mse_google,
        # Deep Learning Results
        "lstm_model": lstm_model,
        "lstm_metrics": lstm_metrics,
        "lstm_results": lstm_results,
        "gru_model": gru_model,
        "gru_metrics": gru_metrics,
        "gru_results": gru_results,
        "ensemble_predictions": ensemble_pred,
        "ensemble_metrics": ensemble_info,
        "metrics_comparison": metrics_comparison,
        # Node Correlations
        "correlations_history_yahoo": correlations_history_yahoo,
        "feature_importance": feature_importance,
    }

# If run as script, run default pipeline and write minimal output
if __name__ == "__main__":
    out = prepare_data_and_run()
    logger.info("Run finished. Price head:\n%s", out["price_df_yahoo"].head().to_string())
    if out["polarity_series_yahoo"] is not None:
        logger.info("Yahoo Sentiment head:\n%s", out["polarity_series_yahoo"].head().to_string())
        logger.info("Google Sentiment head:\n%s", out["polarity_series_google"].head().to_string())
    else:
        logger.info("Sentiment head:\n%s", out["polarity_series"].head().to_string())
    print("weights_history_yahoo length:", len(out["weights_history_yahoo"]))
    print("Static corr:", out["static_corr"], "MSE:", out["static_mse"])
    print("Dynamic Yahoo corr:", out["dynamic_corr_yahoo"], "MSE:", out["dynamic_mse_yahoo"])
    print("Dynamic Google corr:", out["dynamic_corr_google"], "MSE:", out["dynamic_mse_google"])