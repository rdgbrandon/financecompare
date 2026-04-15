import os
import sys

# Load .env file so API keys are available via os.getenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import time
import math
import json
import copy
import logging
import warnings
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

# Suppress TF / protobuf noise before tensorflow is imported
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*tf.reset_default_graph.*")
warnings.filterwarnings("ignore", message=".*Do not pass an.*input_shape.*")
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
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
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
# FRED API for Economic Data
# -----------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", None)
HAS_FRED = bool(FRED_API_KEY)

def fetch_unemployment_data_fred(start_date, end_date):
    """Fetch unemployment from FRED, VIX proxy, or dynamic fallback."""
    idx = pd.date_range(start=start_date, end=end_date, freq='B')
    
    if HAS_FRED:
        try:
            url = 'https://api.stlouisfed.org/fred/series/data'
            r = requests.get(url, params={'series_id': 'UNRATE', 'api_key': FRED_API_KEY, 'file_type': 'json'}, timeout=10)
            if r.status_code == 200:
                data_dict = {}
                for obs in r.json().get('observations', []):
                    try:
                        if obs.get('value') and obs.get('value') != '.':
                            data_dict[pd.to_datetime(obs['date'])] = float(obs['value']) / 100.0
                    except: pass
                if data_dict:
                    s = pd.Series(data_dict).sort_index().clip(0, 1)
                    logger.info(f'Fetched unemployment from FRED: {len(data_dict)} observations')
                    return s.reindex(idx).ffill().fillna(0.5)
        except: pass
    
    try:
        logger.info('Fetching unemployment via VIX volatility proxy...')
        vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True)
        vix = vix_data['Close'] if isinstance(vix_data, pd.DataFrame) and 'Close' in vix_data.columns else (vix_data.iloc[:, 0] if isinstance(vix_data, pd.DataFrame) else vix_data)
        unemployment = 0.5 + 0.25 * (vix.clip(10, 80) - 45) / 35.0
        logger.info(f'Using VIX proxy for unemployment: {len(unemployment)} days')
        return unemployment.clip(0.2, 0.8).reindex(idx).ffill().fillna(0.5)
    except: pass
    
    logger.info('Using dynamic fallback unemployment estimate')
    return pd.Series(0.5 + 0.1 * np.sin(2 * np.pi * idx.dayofyear.astype(float) / 365.0), index=idx).clip(0.3, 0.7)

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
    Attempt to download a Loughran-McDonald style lexicon.
    Returns two sets: positive_words, negative_words.
    Falls back immediately to the embedded fallback wordlist if the network
    request fails, times out, or returns HTML instead of CSV (e.g. Google
    Drive confirmation pages).
    """
    fallback_pos = set(CONFIG["fallback_positive"])
    fallback_neg = set(CONFIG["fallback_negative"])

    if try_urls is None:
        try_urls = CONFIG["lexicon_urls"]

    for url in try_urls:
        try:
            logger.info(f"Attempting to download lexicon from {url}")
            r = requests.get(url, timeout=5)  # short timeout — don't block startup
            if r.status_code != 200:
                logger.warning(f"Lexicon download returned {r.status_code}; using fallback")
                break

            text = r.text.strip()
            # Google Drive "confirm download" page is HTML, not CSV
            if text.lstrip().startswith("<"):
                logger.warning("Lexicon URL returned HTML (likely a Google Drive interstitial); using fallback")
                break

            csvf = StringIO(text)
            df = pd.read_csv(csvf, dtype=str, keep_default_na=False, na_values=[])

            word_col = pos_col = neg_col = None
            for c in df.columns:
                lc = c.strip().lower()
                if word_col is None and ("word" in lc or "token" in lc or "term" in lc):
                    word_col = c
                if pos_col is None and ("positive" in lc or lc == "pos"):
                    pos_col = c
                if neg_col is None and ("negative" in lc or lc == "neg"):
                    neg_col = c
            if word_col is None and len(df.columns) > 0:
                word_col = df.columns[0]

            positive, negative = set(), set()
            for _, row in df.iterrows():
                w = str(row[word_col]).strip().lower()
                if not w:
                    continue
                try:
                    if pos_col and int(float(row[pos_col])) > 0:
                        positive.add(w)
                    if neg_col and int(float(row[neg_col])) > 0:
                        negative.add(w)
                except Exception:
                    pass

            if not positive:
                positive = fallback_pos
            if not negative:
                negative = fallback_neg

            logger.info(f"Loaded lexicon: +{len(positive)} / -{len(negative)}")
            return positive, negative

        except requests.exceptions.Timeout:
            logger.warning(f"Lexicon download timed out for {url}; using fallback")
            break
        except Exception as e:
            logger.warning(f"Lexicon download failed ({e}); using fallback")
            break

    logger.info("Using embedded fallback lexicon.")
    return fallback_pos, fallback_neg

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
        if 'Adj Close' in d.columns:
            price_dict[t] = d['Adj Close']
        elif 'Close' in d.columns:
            price_dict[t] = d['Close']
        elif t in d.columns:
            price_dict[t] = d[t]
        else:
            price_dict[t] = d.iloc[:, 0]
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
    NewsAPI free/developer plans only allow articles from the last 30 days; cap start accordingly.
    """
    if not HAS_NEWSAPI:
        logger.debug("NewsAPI key not found; skipping NewsAPI fetch for query=%s", query)
        return []
    _limit_start = (datetime.now(timezone.utc) - timedelta(days=28)).strftime("%Y-%m-%d")
    try:
        if start < _limit_start:
            logger.debug("NewsAPI: capping start from %s to %s (plan limit)", start, _limit_start)
            start = _limit_start
    except Exception:
        start = _limit_start
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
# Google News RSS fetcher  (real separate source — no API key needed)
# -----------------------
def fetch_google_news_rss(query: str, limit: int = 50, timeout: int = 8) -> list:
    """
    Fetch articles from Google News RSS feed.
    This is a genuinely different source from Yahoo Finance / NewsAPI.
    Parses with stdlib xml.etree — no extra dependencies.
    query: can be a ticker symbol or free-text topic (e.g. "AAPL stock market")
    Returns list of dicts with same schema as fetch_news_newsapi.
    """
    clean_q = query.replace("^","").replace("=F","").replace("-USD","")
    url = (
        f"https://news.google.com/rss/search"
        f"?q={clean_q}+stock+market&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; research/1.0)"})
        if r.status_code != 200:
            logger.warning("Google News RSS returned %d for query=%s", r.status_code, query)
            return []
        root = ET.fromstring(r.content)
        channel = root.find("channel")
        if channel is None:
            return []
        articles = []
        for item in channel.findall("item")[:limit]:
            title   = item.findtext("title",       "") or ""
            link    = item.findtext("link",        "") or ""
            desc    = item.findtext("description", "") or ""
            pub_str = item.findtext("pubDate",     "") or ""

            # Google News puts "Title - Source" in the title element
            source = "Google News"
            if " - " in title:
                parts  = title.rsplit(" - ", 1)
                title  = parts[0].strip()
                source = parts[1].strip()

            # Parse publish date to ISO string
            published_at = pub_str
            pub_ts = int(time.time())
            if pub_str:
                try:
                    pub_ts = int(parsedate_to_datetime(pub_str).timestamp())
                    published_at = parsedate_to_datetime(pub_str).isoformat()
                except Exception:
                    pass

            articles.append({
                "title":       title,
                "description": desc[:400],
                "publishedAt": published_at,
                "url":         link,
                "source":      source,
                "providerPublishTime": pub_ts,
            })
        logger.info("Google News RSS: %d articles for query=%s", len(articles), query)
        return articles
    except Exception as e:
        logger.warning("Google News RSS error for query=%s: %s", query, e)
        return []

# -----------------------
# Google Finance current-quote scraper  (no API key needed)
# -----------------------

# Maps yfinance ticker symbols → Google Finance (symbol, exchange) pairs
_GF_EXCHANGE: dict = {
    "AAPL":"NASDAQ","MSFT":"NASDAQ","NVDA":"NASDAQ","GOOG":"NASDAQ",
    "GOOGL":"NASDAQ","AMZN":"NASDAQ","TSLA":"NASDAQ","META":"NASDAQ",
    "NFLX":"NASDAQ","ORCL":"NYSE","JPM":"NYSE","GS":"NYSE","BAC":"NYSE",
    "XOM":"NYSE","CVX":"NYSE","WMT":"NYSE","V":"NYSE","MA":"NYSE",
    "SPY":"NYSEARCA","QQQ":"NASDAQ","GLD":"NYSEARCA",
}

def scrape_google_finance_quote(ticker: str, timeout: int = 8) -> dict:
    """
    Scrape a real-time quote from Google Finance.
    Uses three extraction strategies in order of reliability:
      1. data-last-price HTML attribute
      2. JSON-LD structured-data <script> block
      3. Known CSS class selectors (fragile, last resort)
    Returns dict with keys: price, change_pct, source, url, timestamp.
    Returns empty dict on any failure — always fail gracefully.
    """
    clean = ticker.replace("^", "").replace("=F", "").replace("-USD", "")
    exchange = _GF_EXCHANGE.get(ticker, _GF_EXCHANGE.get(clean, "NASDAQ"))
    url = f"https://www.google.com/finance/quote/{clean}:{exchange}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    meta = {"source": "Google Finance", "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            logger.debug("Google Finance: HTTP %d for %s", r.status_code, ticker)
            return {}
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")

        # Strategy 1: data-last-price attribute (most reliable when present)
        elem = soup.find(attrs={"data-last-price": True})
        if elem:
            price = float(elem["data-last-price"])
            chg_e = soup.find(attrs={"data-last-normal-market-change-percent": True})
            chg   = (float(chg_e["data-last-normal-market-change-percent"]) * 100
                     if chg_e else 0.0)
            return {**meta, "price": price, "change_pct": chg}

        # Strategy 2: JSON-LD structured data
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                d = json.loads(script.string or "{}")
                if isinstance(d, dict) and "price" in d:
                    return {**meta, "price": float(d["price"]), "change_pct": 0.0}
            except Exception:
                pass

        # Strategy 3: CSS class selectors (brittle — class names change with Google deploys)
        for cls in ["YMlKec fxKbKc", "kf1m4", "AHmHk", "IsqQVc NprOob YMlKec"]:
            elem = soup.find(class_=cls)
            if elem:
                raw = elem.get_text(strip=True).replace("$","").replace(",","")
                try:
                    price = float(raw)
                    if price > 0:
                        return {**meta, "price": price, "change_pct": 0.0}
                except ValueError:
                    pass

        logger.debug("Google Finance: no price found for %s", ticker)
        return {}
    except Exception as e:
        logger.debug("Google Finance scrape error for %s: %s", ticker, e)
        return {}

# -----------------------
# Text cleaning & sentiment
# -----------------------
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
def build_proxies_from_prices(price_df, node_ticker_map, target_index, start_date=None, end_date=None):
    """
    price_df: DataFrame with Adj Close columns for tickers
    node_ticker_map: mapping node-> list of tickers (possibly empty)
    start_date, end_date: for FRED unemployment data fetching
    Returns: inputs DataFrame (outer nodes) and returns Series for target.
    - For nodes with multiple tickers, we aggregate by mean returns.
    - Low_Unemployment: fetch from FRED if available, else constant 0.5
    """
    # compute returns (daily pct change)
    returns = price_df.pct_change().fillna(0)
    inputs = {}

    # Fetch unemployment data if available
    unemployment_data = None
    if start_date and end_date:
        unemployment_data = fetch_unemployment_data_fred(start_date, end_date)

    for node, tickers in node_ticker_map.items():
        if not tickers:
            # no proxy available
            if node == "Low_Unemployment" and unemployment_data is not None:
                # Use FRED unemployment data
                inputs[node] = unemployment_data.reindex(price_df.index).ffill().fillna(0.5)
                logger.info(f"Using FRED unemployment data for {node}")
            else:
                # Fall back to constant
                inputs[node] = pd.Series(0.5, index=price_df.index)
                if node == "Low_Unemployment":
                    logger.info(f"FRED unemployment not available; using constant 0.5 for {node}")
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
    # Squeeze any 2-D arrays/DataFrames down to 1-D Series before constructing the DataFrame
    inputs_clean = {
        k: (v.squeeze() if hasattr(v, "squeeze") and getattr(v, "ndim", 1) > 1 else v)
        for k, v in inputs.items()
    }
    inputs_df = pd.DataFrame(inputs_clean).reindex(price_df.index)
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
        Input(shape=input_shape),
        LSTM(lstm_units, activation='relu', return_sequences=True),
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

    model = Sequential([
        Input(shape=input_shape),
        GRU(gru_units, activation='relu', return_sequences=True),
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

        # Drop zero-variance columns to avoid numpy divide-by-zero in corrwith
        varying_cols = data_slice.columns[data_slice.std() > 1e-10]
        data_slice = data_slice[varying_cols]
        if data_slice.empty:
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
    inputs_df_yahoo, target_yahoo = build_proxies_from_prices(price_df_yahoo, conf["node_ticker_map"], conf["target_ticker"], start_date=start, end_date=end)
    inputs_df_google, target_google = build_proxies_from_prices(price_df_google, conf["node_ticker_map"], conf["target_ticker"], start_date=start, end_date=end)
    # load lexicon
    pos_set, neg_set = load_loughran_lexicon()

    # ── Yahoo sentiment: NewsAPI (historical) → Polygon → yfinance news → synthetic ──
    logger.info("Fetching Yahoo/NewsAPI sentiment data…")
    articles_yahoo = []
    if HAS_NEWSAPI:
        articles_yahoo = fetch_news_newsapi(conf["news_query"], start, end)
    if not articles_yahoo and POLYGON_AVAILABLE:
        tickers_for_news = list({x for sub in conf["node_ticker_map"].values() for x in sub})
        articles_yahoo = fetch_news_polygon_bulk(tickers_for_news, start, end)

    polarity_series_yahoo, intensity_series_yahoo = aggregate_daily_sentiment(
        articles_yahoo, pos_set, neg_set, conf["sentiment_shift_days"]
    )

    # Synthetic Yahoo fallback: price-volatility proxy when no news found
    if polarity_series_yahoo.empty:
        logger.warning("No Yahoo/NewsAPI articles; using price-volatility proxy for Yahoo sentiment")
        vol_y = target_yahoo.rolling(window=20).std().fillna(0)
        polarity_series_yahoo = pd.Series(
            np.tanh((vol_y - vol_y.mean()) / (vol_y.std() + 1e-8)),
            index=target_yahoo.index,
        )
        intensity_series_yahoo = pd.Series(vol_y.values, index=target_yahoo.index)

    # ── Google sentiment: Google News RSS (real separate source) → synthetic ──
    logger.info("Fetching Google News RSS sentiment data…")
    # Use a broader market query so we get decent coverage even for recent news
    google_rss_query = conf.get("news_query", "NASDAQ OR stock market OR technology OR AI")
    articles_google = fetch_google_news_rss(google_rss_query, limit=200)

    polarity_series_google, intensity_series_google = aggregate_daily_sentiment(
        articles_google, pos_set, neg_set, conf["sentiment_shift_days"]
    )

    # Synthetic Google fallback: momentum-based proxy when RSS returns nothing
    if polarity_series_google.empty:
        logger.warning("No Google News RSS articles; using momentum proxy for Google sentiment")
        # Use 5-day vs 20-day return momentum as a sign-preserving proxy
        ret_5   = target_yahoo.rolling(5).mean().fillna(0)
        ret_20  = target_yahoo.rolling(20).mean().fillna(0)
        momentum = ret_5 - ret_20
        polarity_series_google = pd.Series(
            np.tanh(momentum * 50),
            index=target_yahoo.index,
        )
        intensity_series_google = pd.Series(momentum.abs().values, index=target_yahoo.index)

    # Keep references for return dict
    polarity_series  = polarity_series_yahoo    # backward compat
    intensity_series = intensity_series_yahoo

    # ── Attach both sentiment streams as separate FCM input nodes ────────────
    inputs_df_yahoo["News_Sentiment_Yahoo"]  = (
        polarity_series_yahoo.reindex(inputs_df_yahoo.index).fillna(0.0)
    )
    inputs_df_yahoo["News_Sentiment_Google"] = (
        polarity_series_google.reindex(inputs_df_yahoo.index).fillna(0.0)
    )
    inputs_df_google["News_Sentiment_Yahoo"]  = (
        polarity_series_yahoo.reindex(inputs_df_google.index).fillna(0.0)
    )
    inputs_df_google["News_Sentiment_Google"] = (
        polarity_series_google.reindex(inputs_df_google.index).fillna(0.0)
    )

    # ── Sentiment Divergence: (Yahoo − Google) → cross-source agreement signal ─
    # High divergence = sources disagree = heightened uncertainty
    # Low divergence  = sources agree    = higher confidence signal
    sentiment_divergence_yahoo = (
        polarity_series_yahoo.reindex(inputs_df_yahoo.index).fillna(0.0)
        - polarity_series_google.reindex(inputs_df_yahoo.index).fillna(0.0)
    )
    inputs_df_yahoo["Sentiment_Divergence"]  = sentiment_divergence_yahoo
    inputs_df_google["Sentiment_Divergence"] = sentiment_divergence_yahoo  # same divergence

    # ── Sentiment Agreement Score: 1 when both sources agree, 0 when they diverge ─
    # Used as a confidence multiplier in signal generation
    max_div = sentiment_divergence_yahoo.abs().rolling(20, min_periods=1).max().replace(0, 1e-8)
    agreement_score = 1.0 - (sentiment_divergence_yahoo.abs() / max_div).clip(0, 1)
    inputs_df_yahoo["Sentiment_Agreement"]  = agreement_score
    inputs_df_google["Sentiment_Agreement"] = agreement_score

    logger.info(
        "Sentiment sources — Yahoo articles: %d | Google RSS articles: %d",
        len(articles_yahoo), len(articles_google),
    )
    # backward-compat aliases for old callers that checked polarity_series_yahoo
    polarity_series_yahoo_out  = polarity_series_yahoo
    polarity_series_google_out = polarity_series_google
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

    # Compute feature importance (correlation with target at end) — skip constant columns
    _fi_cols = inputs_norm_yahoo.columns[inputs_norm_yahoo.std() > 1e-10]
    feature_importance = inputs_norm_yahoo[_fi_cols].corrwith(actual_norm).abs().sort_values(ascending=False)

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
        "dyn_norm_yahoo": dyn_norm_yahoo,
        "dyn_norm_google": dyn_norm_google,
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