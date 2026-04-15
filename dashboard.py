#!/usr/bin/env python3
"""
FCM AI Trading Terminal  v3.0
Professional-grade dashboard: live data · ML predictions · backtesting · diagnostics
"""

# ── Silence TF / protobuf BEFORE any tensorflow import ──────────────────────
import os, warnings
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS",   "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",    "3")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*oneDNN.*")
warnings.filterwarnings("ignore", message=".*tf.reset_default_graph.*")
warnings.filterwarnings("ignore", message=".*use_container_width.*")
# ────────────────────────────────────────────────────────────────────────────

import sys, time, logging, traceback

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import math
import re
import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ── FCM module ───────────────────────────────────────────────────────────────
_FCM_ERR: Optional[str] = None
_FCM_OK  = False
try:
    from fcm_nasdaq import prepare_data_and_run
    _FCM_OK = True
except Exception as e:
    _FCM_ERR = traceback.format_exc()

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("fcm_terminal")

# ============================================================================
# CONSTANTS
# ============================================================================

EQUITY_TICKERS = ["AAPL","MSFT","NVDA","GOOG","AMZN","TSLA","META","NFLX"]

INDEX_TICKERS: Dict[str, str] = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "DXY (USD)": "DX-Y.NYB",
}

COMMODITY_TICKERS: Dict[str, str] = {
    "S&P 500":          "^GSPC",
    "Gold":             "GC=F",
    "Crude Oil":        "CL=F",
    "Silver":           "SI=F",
    "Natural Gas":      "NG=F",
    "Copper":           "HG=F",
    "Bitcoin":          "BTC-USD",
    "Ethereum":         "ETH-USD",
    "Wheat":            "ZW=F",
}

SECTOR_ETFS: Dict[str, str] = {
    "Technology":    "XLK",
    "Financials":    "XLF",
    "Healthcare":    "XLV",
    "Energy":        "XLE",
    "Consumer Disc": "XLY",
    "Industrials":   "XLI",
    "Materials":     "XLB",
    "Utilities":     "XLU",
    "Real Estate":   "XLRE",
    "Comm. Svcs":    "XLC",
    "Consumer Stap": "XLP",
}

# Chart theme shared across all pages
CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(13,17,23,0)",
    plot_bgcolor="rgba(13,17,23,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#c9d1d9"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# Colour palette
C_GREEN  = "#00d68f"
C_RED    = "#ff4757"
C_BLUE   = "#58a6ff"
C_ORANGE = "#f0883e"
C_PURPLE = "#bc8cff"
C_YELLOW = "#e3b341"
C_GREY   = "#8b949e"

# ============================================================================
# HEALTH TRACKER
# ============================================================================

class HealthTracker:
    def __init__(self):
        self.errors:   List[Dict] = []
        self.warnings: List[Dict] = []
        self.info:     List[Dict] = []

    def _e(self, lvl, src, msg): return {"level":lvl,"source":src,"msg":msg,"ts":datetime.now().strftime("%H:%M:%S")}
    def error(self, src, msg):  self.errors.append(self._e("ERROR",src,msg)); logger.error("[%s] %s",src,msg)
    def warn(self,  src, msg):  self.warnings.append(self._e("WARN",src,msg)); logger.warning("[%s] %s",src,msg)
    def info_log(self, src, msg): self.info.append(self._e("INFO",src,msg))
    def clear(self): self.errors.clear(); self.warnings.clear(); self.info.clear()

    @property
    def ok(self): return len(self.errors) == 0

    def summary(self) -> str:
        """Return a summary string of system status."""
        if self.errors:
            error_count = len(self.errors)
            return f"Error: {error_count} error(s) detected"
        elif self.warnings:
            warning_count = len(self.warnings)
            return f"Warning: {warning_count} warning(s) detected"
        return "All Systems Operational"

_health = HealthTracker()
if not _FCM_OK: _health.error("FCM Import", _FCM_ERR or "Unknown error")

# ============================================================================
# CACHED DATA FETCHING  (Streamlit cache with TTL prevents redundant API calls)
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Returns OHLCV DataFrame; cached 60 s."""
    try:
        t = yf.Ticker(ticker)
        h = t.history(period=period)
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.droplevel(1)
        return h
    except Exception as e:
        _health.error("OHLCV", f"{ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_download(ticker: str, period: str = "1y", **kw) -> pd.DataFrame:
    """yf.download wrapper; cached 60 s."""
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True, **kw)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        return raw
    except Exception as e:
        _health.error("Download", f"{ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_cached(ticker: str) -> List[Dict]:
    """Fetch + normalise yfinance news; cached 10 min.
    Handles both the legacy flat format and the yfinance 1.x nested content format.
    """
    try:
        tick = yf.Ticker(ticker)
        raw  = tick.news or []
        out  = []
        for item in raw[:20]:
            # yfinance 1.x wraps everything inside item["content"]
            if "content" in item and isinstance(item["content"], dict):
                c       = item["content"]
                title   = c.get("title", "Untitled")
                summary = c.get("summary") or c.get("description") or ""
                source  = (c.get("provider") or {}).get("displayName", "Unknown")
                link    = (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url", "#")
                pub_raw = c.get("pubDate") or c.get("displayTime") or ""
                try:
                    ts = int(datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                             .astimezone(timezone.utc).timestamp()) if pub_raw else int(time.time())
                except Exception:
                    ts = int(time.time())
            else:
                # Legacy flat format
                title   = item.get("title", "Untitled")
                summary = item.get("summary") or item.get("description") or ""
                source  = item.get("publisher") or item.get("source") or "Unknown"
                link    = item.get("link", "#")
                ts      = item.get("providerPublishTime", int(time.time()))
            out.append({"title": title, "link": link, "source": source,
                        "ts": ts, "summary": summary})
        return out
    except Exception as e:
        _health.error("News", f"{ticker}: {e}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def fetch_sentiment_cached(ticker: str) -> Dict:
    articles = fetch_news_cached(ticker)
    base = {"polarity": 0.0, "intensity": 0.0, "articles_count": len(articles), "label": "NEUTRAL"}
    if not articles:
        return base
    try:
        from textblob import TextBlob
        pols = []
        for a in articles:
            text = ((a.get("title") or "") + " " + (a.get("summary") or "")).strip()
            if text:
                pols.append(TextBlob(text).sentiment.polarity)
        if not pols:
            return base
        avg = float(np.mean(pols))
        label = "BULLISH" if avg > 0.05 else ("BEARISH" if avg < -0.05 else "NEUTRAL")
        return {"polarity": avg, "intensity": float(np.std(pols)) if len(pols) > 1 else 0.0,
                "articles_count": len(articles), "label": label,
                "polarities": pols}
    except ImportError:
        return {**base, "label": "UNKNOWN (textblob missing)"}
    except Exception as e:
        _health.error("Sentiment", str(e))
        return base

@st.cache_data(ttl=300, show_spinner=False)
def fetch_multi_close(tickers: Tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    """Batch-download Close prices for multiple tickers; cached 5 min."""
    frames = {}
    for t in tickers:
        df = fetch_ohlcv(t, period)
        if not df.empty and "Close" in df.columns:
            frames[t] = df["Close"]
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).ffill().bfill()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str, limit: int = 400) -> pd.Series:
    """Fetch a FRED economic data series; cached 1 hour."""
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return pd.Series(dtype=float)
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": api_key,
                    "file_type": "json", "limit": limit, "sort_order": "desc"},
            timeout=12,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        data = {}
        for o in obs:
            try:
                data[pd.Timestamp(o["date"])] = float(o["value"])
            except (ValueError, KeyError):
                pass
        return pd.Series(data).sort_index()
    except Exception as e:
        _health.warn("FRED", f"{series_id}: {e}")
        return pd.Series(dtype=float)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub_recommendation(symbol: str) -> List[Dict]:
    """Fetch analyst consensus recommendations from Finnhub; cached 5 min."""
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.get("https://finnhub.io/api/v1/stock/recommendation",
                     params={"symbol": symbol, "token": api_key}, timeout=8)
        r.raise_for_status()
        return r.json()[:6]
    except Exception as e:
        _health.warn("Finnhub", f"rec {symbol}: {e}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def fetch_alpha_vantage_indicator(symbol: str, function: str, **params) -> pd.Series:
    """Fetch a technical indicator from Alpha Vantage; cached 10 min."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return pd.Series(dtype=float)
    try:
        p = {"function": function, "symbol": symbol, "interval": "daily",
             "apikey": api_key, "datatype": "json"}
        p.update({k: str(v) for k, v in params.items()})
        r = requests.get("https://www.alphavantage.co/query", params=p, timeout=15)
        r.raise_for_status()
        data = r.json()
        ts_key = next((k for k in data if "Technical" in k or "Analysis" in k), None)
        if not ts_key:
            return pd.Series(dtype=float)
        ts = data[ts_key]
        first_entry = next(iter(ts.values()), {})
        val_key = next(iter(first_entry.keys()), None)
        if not val_key:
            return pd.Series(dtype=float)
        s = pd.Series({pd.Timestamp(d): float(v[val_key]) for d, v in ts.items()})
        return s.sort_index()
    except Exception as e:
        _health.warn("AlphaVantage", f"{function} {symbol}: {e}")
        return pd.Series(dtype=float)

@st.cache_data(ttl=600, show_spinner=False)
def fetch_polygon_news_feed(limit: int = 20) -> List[Dict]:
    """Fetch latest market news from Polygon; cached 10 min."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.get("https://api.polygon.io/v2/reference/news",
                     params={"limit": limit, "order": "desc",
                             "sort": "published_utc", "apiKey": api_key}, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        _health.warn("Polygon", f"news: {e}")
        return []

# ── Regime detection ──────────────────────────────────────────────────────────

def detect_regime(close: pd.Series) -> Tuple[str, str]:
    """Return (regime_label, css_class) using SMA200 + RSI + realized vol."""
    if len(close) < 30:
        return "UNKNOWN", "regime-neutral"
    sma200 = close.rolling(min(200, len(close))).mean().iloc[-1]
    above  = close.iloc[-1] > sma200
    delta  = close.diff()
    gain   = delta.where(delta > 0, 0).rolling(14).mean()
    loss   = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi    = (100 - 100 / (1 + gain / (loss + 1e-8))).iloc[-1]
    rv     = close.pct_change().rolling(20).std().iloc[-1]
    if above and rsi > 55 and rv < 0.025:
        return "BULL", "regime-bull"
    elif not above and rsi < 45:
        return "BEAR", "regime-bear"
    elif rv > 0.035:
        return "HIGH VOL", "regime-volatile"
    else:
        return "NEUTRAL", "regime-neutral"

# ── Technical signals ─────────────────────────────────────────────────────────

def compute_tech(prices_df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    signals = {}
    for t in prices_df.columns:
        try:
            px = prices_df[t].dropna()
            if len(px) < lookback: signals[t] = 0.0; continue
            sma_s = px.rolling(lookback // 2).mean()
            sma_l = px.rolling(lookback).mean()
            delta = px.diff()
            gain  = delta.where(delta > 0, 0).rolling(lookback).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
            rsi   = (100 - 100 / (1 + gain / (loss + 1e-8))).iloc[-1]
            sma_s_v = sma_s.iloc[-1]; sma_l_v = sma_l.iloc[-1]
            sma_sig = 1.0 if sma_s_v > sma_l_v else -1.0
            rsi_sig = 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.0)
            signals[t] = (sma_sig + rsi_sig) / 2.0
        except Exception:
            signals[t] = 0.0
    return signals

# ============================================================================
# PORTFOLIO ENGINE
# ============================================================================

@dataclass
class Trade:
    date: object; ticker: str; action: str
    quantity: float; price: float; cost: float

@dataclass
class Position:
    ticker: str; quantity: float = 0.0
    entry_price: float = 0.0; current_price: float = 0.0

    def update(self, p: float): self.current_price = p

    @property
    def pnl(self): return (self.current_price - self.entry_price) * self.quantity

class Portfolio:
    def __init__(self, capital: float, tc: float = 0.001):
        self.capital = capital; self.cash = capital; self.tc = tc
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity: List[float] = []
        self.dates:  List[object] = []

    def step(self, date, prices: Dict[str, float], signals: Dict[str, int]):
        for t, p in self.positions.items():
            if t in prices: p.update(prices[t])
        buys = sum(1 for s in signals.values() if s == 1)
        for t, sig in signals.items():
            if t not in prices: continue
            px = prices[t]
            if sig == 1:
                alloc = self.cash / max(1, buys)
                qty   = int(alloc / px)
                cost  = qty * px * (1 + self.tc)
                if self.cash >= cost > 0 and qty > 0:
                    self.cash -= cost
                    pos = self.positions.setdefault(t, Position(t))
                    n   = pos.quantity + qty
                    pos.entry_price = (pos.entry_price * pos.quantity + px * qty) / n
                    pos.quantity    = n
                    self.trades.append(Trade(date, t, "BUY", qty, px, cost))
            elif sig == -1:
                pos = self.positions.get(t)
                if pos and pos.quantity > 0:
                    rev = pos.quantity * px * (1 - self.tc)
                    self.cash += rev
                    self.trades.append(Trade(date, t, "SELL", pos.quantity, px, -rev))
                    pos.quantity = 0
        self.equity.append(self._val(prices)); self.dates.append(date)

    def _val(self, prices):
        return self.cash + sum(p.quantity * prices.get(t, 0) for t, p in self.positions.items())

    def metrics(self) -> Dict:
        if len(self.equity) < 2:
            return dict(total_return=0.0, sharpe=0.0, sortino=0.0, calmar=0.0,
                        max_dd=0.0, win_rate=0.0, trades=0, final=self.capital)
        eq   = np.array(self.equity, dtype=float)
        rets = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1e-8)
        tr   = (eq[-1] - self.capital) / self.capital * 100
        std  = np.std(rets)
        sharpe = (np.mean(rets) / std * np.sqrt(252)) if std > 0 else 0.0
        down   = rets[rets < 0]
        sortino = (np.mean(rets) / np.std(down) * np.sqrt(252)) if len(down) > 0 and np.std(down) > 0 else 0.0
        cmx    = np.maximum.accumulate(eq)
        dd_arr = (eq - cmx) / np.where(cmx != 0, cmx, 1e-8) * 100
        max_dd = float(np.min(dd_arr))
        calmar = abs(tr / max_dd) if max_dd != 0 else 0.0
        sells  = [t for t in self.trades if t.action == "SELL"]
        wins   = sum(1 for t in sells if abs(t.cost) > 0)
        win_r  = wins / len(sells) * 100 if sells else 0.0
        return dict(total_return=float(tr), sharpe=float(sharpe), sortino=float(sortino),
                    calmar=float(calmar), max_dd=float(max_dd),
                    win_rate=float(win_r), trades=len(self.trades), final=float(eq[-1]),
                    equity=self.equity, dates=self.dates, drawdown=dd_arr.tolist())

    def trade_df(self) -> pd.DataFrame:
        if not self.trades: return pd.DataFrame()
        return pd.DataFrame([{
            "Date":   t.date.strftime("%Y-%m-%d") if hasattr(t.date,"strftime") else str(t.date),
            "Ticker": t.ticker, "Action": t.action,
            "Qty":    int(t.quantity), "Price": f"${t.price:.2f}",
            "Value":  f"${abs(t.cost):.2f}",
        } for t in self.trades])

# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class HybridSignalGenerator:
    def __init__(self, w_fcm=0.5, w_dl=0.3, w_sent=0.2):
        self.w = {"fcm": w_fcm, "dl": w_dl, "sent": w_sent}; self._norm()

    def _norm(self):
        s = sum(self.w.values())
        if s > 0: self.w = {k: v/s for k, v in self.w.items()}

    def update(self, new: Dict): self.w.update(new); self._norm()

    def signals(self, fcm: Dict[str,float], dl: Optional[Dict]=None,
                sent: Optional[Dict]=None, buy_t=0.05, sell_t=-0.05):
        sigs, conf = {}, {}
        for t in fcm:
            sc  = self.w["fcm"] * fcm.get(t, 0.0)
            sc += self.w["dl"]   * (dl or {}).get(t, 0.0)
            sc += self.w["sent"] * (sent or {}).get(t, 0.0)
            sigs[t] = 1 if sc > buy_t else (-1 if sc < sell_t else 0)
            conf[t] = min(100.0, abs(sc) * 1000)
        return sigs, conf

# ============================================================================
# STREAMLIT CONFIG + CSS
# ============================================================================

st.set_page_config(page_title="FCM AI Terminal", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
.main { background: #0d1117; }
.block-container { padding: 1rem 2rem 1rem 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.9em; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #161b22 100%);
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 12px 16px;
}

/* ── Ticker cards (custom) ── */
.ticker-card {
    background: linear-gradient(160deg,#1c2230 0%,#161b22 100%);
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
    margin-bottom: 6px;
    position: relative;
}
.ticker-card:hover { border-color: #58a6ff; transition: border-color .2s; }
.tc-name  { font-size: 0.8em; color: #8b949e; letter-spacing: .06em; text-transform: uppercase; }
.tc-price { font-size: 1.6em; font-weight: 700; color: #e6edf3; line-height: 1.1; }
.tc-chg-pos { color: #00d68f; font-weight: 600; font-size: 0.95em; }
.tc-chg-neg { color: #ff4757; font-weight: 600; font-size: 0.95em; }

/* ── Signal badges ── */
.badge {
    display: inline-block; font-size: 0.72em; font-weight: 700;
    border-radius: 4px; padding: 2px 8px; letter-spacing: .05em;
}
.badge-buy  { background: rgba(0,214,143,.15); color:#00d68f; border:1px solid #00d68f; }
.badge-sell { background: rgba(255,71,87,.15);  color:#ff4757; border:1px solid #ff4757; }
.badge-hold { background: rgba(139,148,158,.12);color:#8b949e; border:1px solid #3d444d; }

/* ── Regime badges ── */
.regime-bull     { background:rgba(0,214,143,.12); color:#00d68f; border:1px solid #00d68f; border-radius:4px; padding:1px 7px; font-size:.75em; font-weight:700; }
.regime-bear     { background:rgba(255,71,87,.12);  color:#ff4757; border:1px solid #ff4757; border-radius:4px; padding:1px 7px; font-size:.75em; font-weight:700; }
.regime-volatile { background:rgba(227,179,65,.12); color:#e3b341; border:1px solid #e3b341; border-radius:4px; padding:1px 7px; font-size:.75em; font-weight:700; }
.regime-neutral  { background:rgba(88,166,255,.12); color:#58a6ff; border:1px solid #58a6ff; border-radius:4px; padding:1px 7px; font-size:.75em; font-weight:700; }

/* ── Section headers ── */
.section-header {
    font-size: 1.05em; font-weight: 600; color: #e6edf3;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin: 16px 0 12px 0;
    letter-spacing: .03em;
}

/* ── 52-week bar ── */
.bar52-wrap { background:#21262d; border-radius:3px; height:4px; margin:5px 0 2px 0; }
.bar52-fill { background: linear-gradient(90deg,#ff4757,#e3b341,#00d68f);
              border-radius:3px; height:4px; }
.bar52-labels { display:flex; justify-content:space-between;
                font-size:0.65em; color:#8b949e; }

/* ── Index strip ── */
.idx-strip { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:12px; }
.idx-item  { background:#161b22; border:1px solid #21262d; border-radius:7px;
             padding:8px 16px; min-width:110px; }
.idx-name  { font-size:0.7em; color:#8b949e; }
.idx-val   { font-size:1.15em; font-weight:700; color:#e6edf3; }

/* ── Sentiment inline ── */
.sent-pos { color:#00d68f; font-weight:600; }
.sent-neg { color:#ff4757; font-weight:600; }
.sent-neu { color:#8b949e; }

/* ── Health dot ── */
.h-dot-ok   { color:#00d68f; font-size:.9em; }
.h-dot-warn { color:#e3b341; font-size:.9em; }
.h-dot-err  { color:#ff4757; font-size:.9em; }

/* ── Dividers ── */
hr { border-color:#21262d !important; }

/* ── Expander tweaks ── */
.streamlit-expanderHeader { font-size:.88em !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION + UTILS
# ============================================================================

def init():
    defs = {
        "fcm_data": None, "fcm_err": None,
        "signal_gen": HybridSignalGenerator(),
        "health": _health,
    }
    for k, v in defs.items():
        if k not in st.session_state: st.session_state[k] = v

def _health_dot():
    h: HealthTracker = st.session_state["health"]
    if h.errors:
        st.sidebar.markdown(f'<span class="h-dot-err">● {len(h.errors)} error(s)</span>', unsafe_allow_html=True)
    elif h.warnings:
        st.sidebar.markdown(f'<span class="h-dot-warn">● {len(h.warnings)} warning(s)</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="h-dot-ok">● All systems OK</span>', unsafe_allow_html=True)

def _section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def _badge(sig: int) -> str:
    if sig == 1:  return '<span class="badge badge-buy">▲ BUY</span>'
    if sig == -1: return '<span class="badge badge-sell">▼ SELL</span>'
    return '<span class="badge badge-hold">● HOLD</span>'

def _sent_label(s: Dict) -> str:
    label = s.get("label", "NEUTRAL")
    pol   = s.get("polarity", 0.0)
    if label == "BULLISH": return f'<span class="sent-pos">{label} ({pol:+.3f})</span>'
    if label == "BEARISH": return f'<span class="sent-neg">{label} ({pol:+.3f})</span>'
    return f'<span class="sent-neu">{label} ({pol:+.3f})</span>'

def _regime_badge(label: str, cls: str) -> str:
    return f'<span class="{cls}">{label}</span>'

def _52w_bar(price, low, high) -> str:
    pct = (price - low) / max(high - low, 1e-8) * 100
    pct = min(max(pct, 0), 100)
    return f"""
<div class="bar52-wrap"><div class="bar52-fill" style="width:{pct:.0f}%"></div></div>
<div class="bar52-labels"><span>${low:.0f}</span><span>52w</span><span>${high:.0f}</span></div>
"""

def mini_sparkline(prices: pd.Series, color: str = C_BLUE, height: int = 55) -> go.Figure:
    """Tiny sparkline with fill — no axes, no margins."""
    fig = go.Figure(go.Scatter(
        y=prices.values, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba"),
    ))
    fig.update_layout(
        margin=dict(l=0,r=0,t=0,b=0), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

def _safe_len(v):
    """Return len(v) if v is array-like, else 0."""
    if v is None: return 0
    try: return len(v)
    except TypeError: return 0

def _model_metrics_row(pred, actual, name):
    """Compute model comparison metrics. Returns a dict."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        p = np.asarray(pred, dtype=float)
        a = np.asarray(actual, dtype=float)
        n = min(len(p), len(a))
        if n < 4:
            return {"Model": name, "Corr": 0.0, "MSE": 0.0, "MAE": 0.0, "Dir Acc %": 0.0, "R²": 0.0}
        p, a = p[-n:], a[-n:]
        mask = ~(np.isnan(p) | np.isnan(a))
        p, a = p[mask], a[mask]
        if len(p) < 4:
            return {"Model": name, "Corr": 0.0, "MSE": 0.0, "MAE": 0.0, "Dir Acc %": 0.0, "R²": 0.0}
        corr = float(np.corrcoef(p, a)[0, 1]) if np.std(p) > 0 and np.std(a) > 0 else 0.0
        if np.isnan(corr): corr = 0.0
        mse  = float(np.mean((p - a) ** 2))
        mae  = float(np.mean(np.abs(p - a)))
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
        p_dir = np.sign(np.diff(p)); a_dir = np.sign(np.diff(a))
        dir_acc = float(np.mean(p_dir == a_dir) * 100) if len(p_dir) > 0 else 0.0
        return {"Model": name, "Corr": round(corr, 4), "MSE": round(mse, 6),
                "MAE": round(mae, 6), "Dir Acc %": round(dir_acc, 1), "R²": round(r2, 4)}

def _blend_sentiment(fcm_preds, sentiment_series, dates, alpha=0.30):
    """Return FCM predictions blended with a sentiment overlay (alpha weight on sentiment)."""
    fcm = np.asarray(fcm_preds, dtype=float)
    try:
        idx = pd.DatetimeIndex(dates)
        sent = sentiment_series.reindex(idx).ffill().bfill().fillna(0.0)
        s = np.asarray(sent, dtype=float)[-len(fcm):]
        # normalise sentiment to [0,1] matching FCM output range
        s_norm = (s + 1.0) / 2.0  # sentiment is in [-1,1]
        blended = (1 - alpha) * fcm + alpha * s_norm[-len(fcm):]
        return np.clip(blended, 0, 1)
    except Exception:
        return fcm

def _fcm_weights_matrix(weights_dict, node_order):
    """Convert FCM weight dict to a DataFrame matrix."""
    nodes = [n for n in node_order if n in weights_dict or any(n in v for v in weights_dict.values())]
    # collect all unique src and tgt
    srcs = list(weights_dict.keys())
    tgts_set = set()
    for v in weights_dict.values():
        tgts_set.update(v.keys())
    tgts = sorted(tgts_set)
    mat = pd.DataFrame(0.0, index=srcs, columns=tgts)
    for src, tv in weights_dict.items():
        for tgt, w in tv.items():
            if tgt in mat.columns:
                mat.loc[src, tgt] = w
    return mat

# ── Shared layer/node constants used by all FCM visualisations ────────────────
_FCM_LAYERS = {
    "Input":    ["Large_Deficits","Foreign_Demand","Low_Unemployment",
                 "Supply_Increase","Energy_Price_Increase","Consumer_Demand",
                 "Business_Investment","AI_Technology"],
    "Sentiment":["News_Sentiment_Yahoo","News_Sentiment_Google",
                 "Sentiment_Divergence","Sentiment_Agreement"],
    "Concept":  ["Market_Volatility","Equity_Inflows",
                 "Monetary_Policy","Inflation",
                 "Corporate_Earnings","Investor_Sentiment"],
    "Output":   ["NASDAQ"],
}
_FCM_LAYER_X  = {"Input":0.0,"Sentiment":0.32,"Concept":0.64,"Output":1.0}
_FCM_LAYER_C  = {"Input":C_BLUE,"Sentiment":C_YELLOW,"Concept":C_ORANGE,"Output":C_GREEN}
_FCM_RING     = [n for layer in list(_FCM_LAYERS.values())[:-1] for n in layer]


def _layer_idx_map():
    m = {}
    for li, (_, nodes) in enumerate(_FCM_LAYERS.items()):
        for n in nodes:
            m[n] = li
    return m


def _build_full_edges(weights_dict, inputs_norm, layer_idx, forward_only=True):
    """
    Merge learned FCM weights with pairwise correlation edges so that every
    node connects forward to at least one other node.
    Returns {src: {tgt: weight}}.
    """
    edge: dict = {}

    # 1. Learned FCM weights
    for src, tgts in weights_dict.items():
        src_li = layer_idx.get(src, -1)
        if src_li < 0:
            continue
        for tgt, w in tgts.items():
            tgt_li = layer_idx.get(tgt, -1)
            if tgt_li < 0:
                continue
            if forward_only and tgt_li <= src_li:
                continue
            if abs(w) >= 0.01:
                edge.setdefault(src, {})[tgt] = float(w)

    # 2. Correlation-based edges from inputs_norm (outer nodes only)
    if inputs_norm is not None and len(inputs_norm) >= 10:
        try:
            corr = inputs_norm.corr()
            for src in corr.columns:
                src_li = layer_idx.get(src, -1)
                if src_li < 0:
                    continue
                for tgt in corr.columns:
                    if tgt == src:
                        continue
                    tgt_li = layer_idx.get(tgt, -1)
                    if tgt_li < 0:
                        continue
                    if forward_only and tgt_li <= src_li:
                        continue
                    if tgt in edge.get(src, {}):
                        continue   # learned weight takes precedence
                    c = float(corr.loc[src, tgt])
                    if pd.isna(c) or abs(c) < 0.10:
                        continue
                    edge.setdefault(src, {})[tgt] = c * 0.65
        except Exception:
            pass

    # 3. Guarantee every non-NASDAQ node has ≥1 forward / outgoing edge
    all_non_out = [n for nodes in list(_FCM_LAYERS.values())[:-1] for n in nodes]
    for src in all_non_out:
        src_li = layer_idx.get(src, -1)
        if src_li < 0:
            continue
        if edge.get(src):
            continue
        # fallback: connect to NASDAQ using learned weight or small default
        w = weights_dict.get(src, {}).get("NASDAQ")
        edge[src] = {"NASDAQ": float(w) if w is not None else 0.12}

    # 4. For circular view: ensure every ring node has an edge to NASDAQ
    if not forward_only:
        for src in _FCM_RING:
            if "NASDAQ" not in edge.get(src, {}):
                w = weights_dict.get(src, {}).get("NASDAQ")
                edge.setdefault(src, {})["NASDAQ"] = float(w) if w is not None else 0.12

    return edge


def _monthly_snapshots(weights_history, date_history, inputs_norm, n_months=13):
    """
    Generate (month_end_ts, weight_snapshot, rolling_corr_or_None) for the
    last n_months calendar months.  Used to drive the animated charts.
    """
    if not weights_history:
        return []

    wh_ts = []
    for d in date_history:
        try:
            wh_ts.append(pd.Timestamp(d))
        except Exception:
            wh_ts.append(pd.NaT)

    def _closest_w(ts):
        best, best_d = weights_history[-1], float("inf")
        for w, t in zip(weights_history, wh_ts):
            if pd.isna(t):
                continue
            diff = abs((ts - t).days)
            if diff < best_d:
                best, best_d = w, diff
        return best

    try:
        month_ends = pd.date_range(
            end=pd.Timestamp.now().normalize(), periods=n_months, freq="ME"
        )
    except Exception:
        month_ends = pd.date_range(
            end=pd.Timestamp.now().normalize(), periods=n_months, freq="ME"
        )

    snaps = []
    for me in month_ends:
        w_snap = _closest_w(me)
        corr_m = None
        if inputs_norm is not None and len(inputs_norm) >= 10:
            try:
                win = inputs_norm.loc[
                    (inputs_norm.index >= me - pd.Timedelta(days=60)) &
                    (inputs_norm.index <= me)
                ]
                if len(win) >= 5:
                    corr_m = win.corr()
            except Exception:
                pass
        snaps.append((me, w_snap, corr_m))
    return snaps


def _anim_play_menu():
    return [dict(
        type="buttons", showactive=False,
        y=1.06, x=0.5, xanchor="center", yanchor="bottom",
        buttons=[
            dict(label="▶  Play", method="animate",
                 args=[None, dict(frame=dict(duration=700, redraw=True),
                                  fromcurrent=True,
                                  transition=dict(duration=200))]),
            dict(label="⏸  Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0))]),
        ],
        direction="left", pad=dict(r=10, t=5),
        bgcolor="rgba(22,27,34,0.9)", bordercolor="#58a6ff", borderwidth=1,
        font=dict(color="#c9d1d9"),
    )]


def _anim_slider(steps):
    return [dict(
        active=0, steps=steps,
        x=0.05, len=0.90, y=-0.04, yanchor="top",
        currentvalue=dict(visible=True, prefix="Month: ",
                          font=dict(size=11, color="#c9d1d9")),
        transition=dict(duration=200),
        bgcolor="rgba(22,27,34,0.6)",
        bordercolor="#30363d",
        tickcolor="#c9d1d9",
        font=dict(color="#c9d1d9", size=9),
    )]


# ─────────────────────────────────────────────────────────────────────────────
# STATIC VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_fcm_layered_diagram(weights_dict, node_vals, inputs_norm=None,
                                title="FCM Layered Diagram", height=620) -> go.Figure:
    """4-column layered FCM diagram — all 19 nodes fully connected."""
    li_map  = _layer_idx_map()
    edge    = _build_full_edges(weights_dict, inputs_norm, li_map, forward_only=True)

    node_pos: dict = {}
    for layer, nodes in _FCM_LAYERS.items():
        x = _FCM_LAYER_X[layer]
        n = len(nodes)
        for i, node in enumerate(nodes):
            node_pos[node] = (x, 1.0 - i / max(1, n - 1) if n > 1 else 0.5)

    fig = go.Figure()
    pos_ex, pos_ey, neg_ex, neg_ey = [], [], [], []
    for src, tgts in edge.items():
        if src not in node_pos:
            continue
        sx, sy = node_pos[src]
        for tgt, w in tgts.items():
            if tgt not in node_pos:
                continue
            tx, ty = node_pos[tgt]
            if w > 0:
                pos_ex += [sx, tx, None]; pos_ey += [sy, ty, None]
            else:
                neg_ex += [sx, tx, None]; neg_ey += [sy, ty, None]

    if pos_ex:
        fig.add_trace(go.Scatter(x=pos_ex, y=pos_ey, mode="lines",
                                  line=dict(color="rgba(0,214,143,0.55)", width=1.4),
                                  name="Positive (+)", showlegend=True, hoverinfo="skip"))
    if neg_ex:
        fig.add_trace(go.Scatter(x=neg_ex, y=neg_ey, mode="lines",
                                  line=dict(color="rgba(255,71,87,0.55)", width=1.4),
                                  name="Negative (−)", showlegend=True, hoverinfo="skip"))

    for layer, nodes in _FCM_LAYERS.items():
        lx = [node_pos[n][0] for n in nodes if n in node_pos]
        ly = [node_pos[n][1] for n in nodes if n in node_pos]
        labels = [n.replace("_", " ") for n in nodes if n in node_pos]
        vals   = [float(node_vals.get(n, 0.5)) if node_vals else 0.5 for n in nodes if n in node_pos]
        sizes  = [max(14, min(30, 14 + abs(v - 0.5) * 32)) for v in vals]
        is_out = (layer == "Output")
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode="markers+text",
            marker=dict(size=sizes, color=_FCM_LAYER_C[layer],
                        symbol="square" if layer == "Input" else "circle",
                        line=dict(color="#0d1117", width=2), opacity=0.92),
            text=labels,
            textposition="middle right" if not is_out else "middle left",
            textfont=dict(size=9 if len(nodes) > 5 else 10, color="#c9d1d9"),
            name=layer, customdata=vals,
            hovertemplate="<b>%{text}</b><br>Activation: %{customdata:.3f}<extra></extra>",
        ))

    for layer, x in _FCM_LAYER_X.items():
        fig.add_annotation(x=x, y=1.06, text=f"<b>{layer}</b>",
                            showarrow=False, xanchor="center",
                            font=dict(size=11, color=_FCM_LAYER_C[layer]),
                            xref="paper", yref="paper")

    fig.update_layout(**CHART_THEME, height=height,
                       title=dict(text=title, font=dict(size=13)),
                       xaxis=dict(visible=False, range=[-0.12, 1.30]),
                       yaxis=dict(visible=False, range=[-0.08, 1.12]),
                       legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.04,
                                   bgcolor="rgba(22,27,34,0.88)", bordercolor="#30363d",
                                   borderwidth=1, font=dict(size=10)),
                       hovermode="closest")
    return fig


def _build_fcm_circular_network(weights_dict, node_vals, inputs_norm=None,
                                 title="FCM Circular Network", height=620) -> go.Figure:
    """Circular FCM network — NASDAQ at centre, all 18 ring nodes connected."""
    li_map  = _layer_idx_map()
    edge    = _build_full_edges(weights_dict, inputs_norm, li_map, forward_only=False)

    node_pos: dict = {}
    for i, node in enumerate(_FCM_RING):
        angle = 2 * math.pi * i / len(_FCM_RING) - math.pi / 2
        node_pos[node] = (math.cos(angle), math.sin(angle))
    node_pos["NASDAQ"] = (0.0, 0.0)

    fig = go.Figure()
    pos_ex, pos_ey, neg_ex, neg_ey = [], [], [], []
    for src, tgts in edge.items():
        if src not in node_pos:
            continue
        sx, sy = node_pos[src]
        for tgt, w in tgts.items():
            if tgt not in node_pos:
                continue
            tx, ty = node_pos[tgt]
            if w > 0:
                pos_ex += [sx, tx, None]; pos_ey += [sy, ty, None]
            else:
                neg_ex += [sx, tx, None]; neg_ey += [sy, ty, None]

    if pos_ex:
        fig.add_trace(go.Scatter(x=pos_ex, y=pos_ey, mode="lines",
                                  line=dict(color="rgba(0,214,143,0.55)", width=1.6),
                                  name="Positive (+)", showlegend=True, hoverinfo="skip"))
    if neg_ex:
        fig.add_trace(go.Scatter(x=neg_ex, y=neg_ey, mode="lines",
                                  line=dict(color="rgba(255,71,87,0.55)", width=1.6),
                                  name="Negative (−)", showlegend=True, hoverinfo="skip"))

    ring_vals  = [float(node_vals.get(n, 0.5)) if node_vals else 0.5 for n in _FCM_RING]
    ring_sizes = [max(12, min(26, 12 + abs(v - 0.5) * 28)) for v in ring_vals]
    fig.add_trace(go.Scatter(
        x=[node_pos[n][0] for n in _FCM_RING],
        y=[node_pos[n][1] for n in _FCM_RING],
        mode="markers+text",
        marker=dict(size=ring_sizes, color=C_BLUE, line=dict(color="#0d1117", width=2), opacity=0.9),
        text=[n.replace("_", " ") for n in _FCM_RING],
        textfont=dict(size=8, color="#c9d1d9"), textposition="top center",
        name="Ring Nodes", customdata=ring_vals,
        hovertemplate="<b>%{text}</b><br>Activation: %{customdata:.3f}<extra></extra>",
    ))

    cval = float(node_vals.get("NASDAQ", 0.5)) if node_vals else 0.5
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=32, color=C_GREEN, symbol="star",
                    line=dict(color="#ffffff", width=2), opacity=1.0),
        text=["NASDAQ"], textfont=dict(size=11, color="#ffffff"),
        textposition="middle center", name="NASDAQ",
        customdata=[cval],
        hovertemplate="<b>NASDAQ</b><br>Activation: %{customdata:.3f}<extra></extra>",
    ))

    fig.update_layout(**CHART_THEME, height=height,
                       title=dict(text=title, font=dict(size=13)),
                       xaxis=dict(visible=False, range=[-1.5, 1.5]),
                       yaxis=dict(visible=False, range=[-1.5, 1.5]),
                       legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.04,
                                   bgcolor="rgba(22,27,34,0.88)", bordercolor="#30363d",
                                   borderwidth=1, font=dict(size=10)),
                       hovermode="closest")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ANIMATED VISUALISATIONS  (month-over-month, last 12 months)
# ─────────────────────────────────────────────────────────────────────────────

def _build_fcm_circular_animated(
    weights_history, date_history, inputs_norm, node_order,
    title="FCM Circular — 1-Year Lookback", height=700,
) -> go.Figure:
    """
    Animated circular FCM: 12 monthly snapshots showing how edge polarity
    (positive / negative) and node activations evolve month by month.
    """
    li_map   = _layer_idx_map()
    snaps    = _monthly_snapshots(weights_history, date_history, inputs_norm)
    if not snaps:
        return go.Figure()

    node_pos: dict = {}
    for i, node in enumerate(_FCM_RING):
        angle = 2 * math.pi * i / len(_FCM_RING) - math.pi / 2
        node_pos[node] = (math.cos(angle), math.sin(angle))
    node_pos["NASDAQ"] = (0.0, 0.0)

    def _nv(ts):
        vals: dict = {}
        if inputs_norm is None:
            return vals
        try:
            idx = inputs_norm.index.get_indexer([ts], method="nearest")[0]
            if idx >= 0:
                row = inputs_norm.iloc[idx]
                for n in inputs_norm.columns:
                    vals[n] = float(row[n])
        except Exception:
            pass
        return vals

    def _edges_traces(w_snap, corr_m):
        # Build merged edge dict for this snapshot
        merged = _build_full_edges(w_snap, corr_m, li_map, forward_only=False)
        px, py, nx_, ny_ = [], [], [], []
        for src, tgts in merged.items():
            if src not in node_pos:
                continue
            sx, sy = node_pos[src]
            for tgt, wt in tgts.items():
                if tgt not in node_pos:
                    continue
                tx, ty = node_pos[tgt]
                if wt > 0:
                    px += [sx, tx, None]; py += [sy, ty, None]
                else:
                    nx_ += [sx, tx, None]; ny_ += [sy, ty, None]
        return px, py, nx_, ny_

    def _ring_data(nv):
        vals  = [float(nv.get(n, 0.5)) for n in _FCM_RING]
        sizes = [max(12, min(28, 12 + abs(v - 0.5) * 32)) for v in vals]
        return [node_pos[n][0] for n in _FCM_RING], \
               [node_pos[n][1] for n in _FCM_RING], \
               [n.replace("_", " ") for n in _FCM_RING], vals, sizes

    me0, w0, c0 = snaps[0]
    nv0 = _nv(me0)
    px0, py0, nx0, ny0 = _edges_traces(w0, c0)
    rx0, ry0, rl0, rv0, rs0 = _ring_data(nv0)
    cval0 = float(nv0.get("NASDAQ", 0.5))
    d0_str = me0.strftime("%Y-%m")

    fig = go.Figure(data=[
        go.Scatter(x=px0, y=py0, mode="lines",
                   line=dict(color="rgba(0,214,143,0.55)", width=1.8),
                   name="Positive (+)", showlegend=True, hoverinfo="skip"),
        go.Scatter(x=nx0, y=ny0, mode="lines",
                   line=dict(color="rgba(255,71,87,0.55)", width=1.8),
                   name="Negative (−)", showlegend=True, hoverinfo="skip"),
        go.Scatter(x=rx0, y=ry0, mode="markers+text",
                   marker=dict(size=rs0, color=C_BLUE,
                               line=dict(color="#0d1117", width=2), opacity=0.9),
                   text=rl0, textfont=dict(size=8, color="#c9d1d9"),
                   textposition="top center", name="Nodes",
                   customdata=rv0,
                   hovertemplate="<b>%{text}</b><br>Activation: %{customdata:.3f}<extra></extra>"),
        go.Scatter(x=[0], y=[0], mode="markers+text",
                   marker=dict(size=34, color=C_GREEN, symbol="star",
                               line=dict(color="#ffffff", width=2), opacity=1.0),
                   text=["NASDAQ"], textfont=dict(size=11, color="#ffffff"),
                   textposition="middle center", name="NASDAQ",
                   customdata=[cval0],
                   hovertemplate="<b>NASDAQ</b><br>Activation: %{customdata:.3f}<extra></extra>"),
    ])

    frames, steps = [], []
    for me, w_snap, corr_m in snaps:
        nv = _nv(me)
        px, py, nx_, ny_ = _edges_traces(w_snap, corr_m)
        rx, ry, rl, rv, rs = _ring_data(nv)
        cval = float(nv.get("NASDAQ", 0.5))
        d_str = me.strftime("%Y-%m")
        frames.append(go.Frame(
            data=[
                go.Scatter(x=px, y=py, mode="lines",
                           line=dict(color="rgba(0,214,143,0.55)", width=1.8), hoverinfo="skip"),
                go.Scatter(x=nx_, y=ny_, mode="lines",
                           line=dict(color="rgba(255,71,87,0.55)", width=1.8), hoverinfo="skip"),
                go.Scatter(x=rx, y=ry, mode="markers+text",
                           marker=dict(size=rs, color=C_BLUE,
                                       line=dict(color="#0d1117", width=2), opacity=0.9),
                           text=rl, textfont=dict(size=8, color="#c9d1d9"),
                           textposition="top center", customdata=rv,
                           hovertemplate="<b>%{text}</b><br>Activation: %{customdata:.3f}<extra></extra>"),
                go.Scatter(x=[0], y=[0], mode="markers+text",
                           marker=dict(size=34, color=C_GREEN, symbol="star",
                                       line=dict(color="#ffffff", width=2), opacity=1.0),
                           text=["NASDAQ"], textfont=dict(size=11, color="#ffffff"),
                           textposition="middle center", customdata=[cval],
                           hovertemplate="<b>NASDAQ</b><br>Activation: %{customdata:.3f}<extra></extra>"),
            ],
            name=d_str,
            layout=go.Layout(title=dict(text=f"{title} — {d_str}")),
        ))
        steps.append(dict(
            args=[[d_str], dict(frame=dict(duration=700, redraw=True),
                                mode="immediate", transition=dict(duration=200))],
            label=d_str, method="animate",
        ))

    fig.frames = frames
    _t = {k: v for k, v in CHART_THEME.items() if k != "margin"}
    fig.update_layout(
        **_t, height=height,
        margin=dict(l=10, r=10, t=70, b=120),
        title=dict(text=f"{title} — {d0_str}", font=dict(size=13)),
        xaxis=dict(visible=False, range=[-1.55, 1.55]),
        yaxis=dict(visible=False, range=[-1.55, 1.55]),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.10,
                    bgcolor="rgba(22,27,34,0.88)", bordercolor="#30363d",
                    borderwidth=1, font=dict(size=10)),
        hovermode="closest",
        updatemenus=_anim_play_menu(),
        sliders=_anim_slider(steps),
    )
    return fig


def _build_fcm_layered_animated(
    weights_history, date_history, inputs_norm, node_order,
    title="FCM Layered — 1-Year Lookback", height=680,
) -> go.Figure:
    """
    Animated 4-column layered FCM: 12 monthly snapshots showing how edge
    weights and node activations evolve month by month over the last year.
    """
    li_map = _layer_idx_map()
    snaps  = _monthly_snapshots(weights_history, date_history, inputs_norm)
    if not snaps:
        return go.Figure()

    node_pos: dict = {}
    for layer, nodes in _FCM_LAYERS.items():
        x = _FCM_LAYER_X[layer]
        n = len(nodes)
        for i, node in enumerate(nodes):
            node_pos[node] = (x, 1.0 - i / max(1, n - 1) if n > 1 else 0.5)

    def _nv(ts):
        vals: dict = {}
        if inputs_norm is None:
            return vals
        try:
            idx = inputs_norm.index.get_indexer([ts], method="nearest")[0]
            if idx >= 0:
                row = inputs_norm.iloc[idx]
                for n in inputs_norm.columns:
                    vals[n] = float(row[n])
        except Exception:
            pass
        return vals

    def _edge_traces(w_snap, corr_m):
        merged = _build_full_edges(w_snap, corr_m, li_map, forward_only=True)
        px, py, nx_, ny_ = [], [], [], []
        for src, tgts in merged.items():
            if src not in node_pos:
                continue
            sx, sy = node_pos[src]
            for tgt, wt in tgts.items():
                if tgt not in node_pos:
                    continue
                if li_map.get(tgt, -1) <= li_map.get(src, -1):
                    continue
                tx, ty = node_pos[tgt]
                if wt > 0:
                    px += [sx, tx, None]; py += [sy, ty, None]
                else:
                    nx_ += [sx, tx, None]; ny_ += [sy, ty, None]
        return px, py, nx_, ny_

    def _layer_traces(nv):
        traces = []
        for layer, nodes in _FCM_LAYERS.items():
            lx     = [node_pos[n][0] for n in nodes if n in node_pos]
            ly     = [node_pos[n][1] for n in nodes if n in node_pos]
            labels = [n.replace("_", " ") for n in nodes if n in node_pos]
            vals   = [float(nv.get(n, 0.5)) for n in nodes if n in node_pos]
            sizes  = [max(14, min(30, 14 + abs(v - 0.5) * 32)) for v in vals]
            is_out = (layer == "Output")
            traces.append(go.Scatter(
                x=lx, y=ly, mode="markers+text",
                marker=dict(size=sizes, color=_FCM_LAYER_C[layer],
                            symbol="square" if layer == "Input" else "circle",
                            line=dict(color="#0d1117", width=2), opacity=0.92),
                text=labels,
                textposition="middle right" if not is_out else "middle left",
                textfont=dict(size=9 if len(nodes) > 5 else 10, color="#c9d1d9"),
                name=layer, customdata=vals,
                hovertemplate="<b>%{text}</b><br>Activation: %{customdata:.3f}<extra></extra>",
            ))
        return traces

    me0, w0, c0 = snaps[0]
    nv0 = _nv(me0)
    px0, py0, nx0, ny0 = _edge_traces(w0, c0)
    lt0 = _layer_traces(nv0)
    d0_str = me0.strftime("%Y-%m")

    fig = go.Figure(data=[
        go.Scatter(x=px0, y=py0, mode="lines",
                   line=dict(color="rgba(0,214,143,0.55)", width=1.4),
                   name="Positive (+)", showlegend=True, hoverinfo="skip"),
        go.Scatter(x=nx0, y=ny0, mode="lines",
                   line=dict(color="rgba(255,71,87,0.55)", width=1.4),
                   name="Negative (−)", showlegend=True, hoverinfo="skip"),
        *lt0,
    ])

    frames, steps = [], []
    for me, w_snap, corr_m in snaps:
        nv = _nv(me)
        px, py, nx_, ny_ = _edge_traces(w_snap, corr_m)
        lt = _layer_traces(nv)
        d_str = me.strftime("%Y-%m")
        frames.append(go.Frame(
            data=[
                go.Scatter(x=px, y=py, mode="lines",
                           line=dict(color="rgba(0,214,143,0.55)", width=1.4), hoverinfo="skip"),
                go.Scatter(x=nx_, y=ny_, mode="lines",
                           line=dict(color="rgba(255,71,87,0.55)", width=1.4), hoverinfo="skip"),
                *lt,
            ],
            name=d_str,
            layout=go.Layout(title=dict(text=f"{title} — {d_str}")),
        ))
        steps.append(dict(
            args=[[d_str], dict(frame=dict(duration=700, redraw=True),
                                mode="immediate", transition=dict(duration=200))],
            label=d_str, method="animate",
        ))

    fig.frames = frames
    annotations = [dict(
        x=x, y=1.10, text=f"<b>{layer}</b>",
        showarrow=False, xanchor="center",
        font=dict(size=11, color=_FCM_LAYER_C[layer]),
        xref="paper", yref="paper",
    ) for layer, x in _FCM_LAYER_X.items()]

    _t = {k: v for k, v in CHART_THEME.items() if k != "margin"}
    fig.update_layout(
        **_t, height=height,
        margin=dict(l=10, r=10, t=70, b=120),
        title=dict(text=f"{title} — {d0_str}", font=dict(size=13)),
        annotations=annotations,
        xaxis=dict(visible=False, range=[-0.14, 1.30]),
        yaxis=dict(visible=False, range=[-0.10, 1.18]),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.10,
                    bgcolor="rgba(22,27,34,0.88)", bordercolor="#30363d",
                    borderwidth=1, font=dict(size=10)),
        hovermode="closest",
        updatemenus=_anim_play_menu(),
        sliders=_anim_slider(steps),
    )
    return fig


def conf_band_chart(dates, actual, pred, std_band=None,
                    extra_traces=None, title="", height=400) -> go.Figure:
    """Prediction chart with optional confidence band shading."""
    fig = go.Figure()
    if actual is not None and len(actual) > 0:
        fig.add_trace(go.Scatter(x=dates[-len(actual):], y=actual,
            name="Actual", line=dict(color=C_GREEN, width=1.5), opacity=0.8))
    if pred is not None and len(pred) > 0:
        n = len(pred)
        x = dates[-n:] if len(dates) >= n else list(range(n))
        fig.add_trace(go.Scatter(x=x, y=pred,
            name="FCM Pred", line=dict(color=C_BLUE, width=2, dash="dot")))
        if std_band is not None and len(std_band) == n:
            upper = np.array(pred) + np.array(std_band)
            lower = np.array(pred) - np.array(std_band)
            fig.add_trace(go.Scatter(x=list(x)+list(x)[::-1],
                y=list(upper)+list(lower)[::-1],
                fill="toself", fillcolor="rgba(88,166,255,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="FCM ±1σ", showlegend=True, opacity=0.6))
    if extra_traces:
        for tr in extra_traces: fig.add_trace(tr)
    fig.update_layout(**CHART_THEME, title=dict(text=title, font=dict(size=13)),
                       height=height, hovermode="x unified",
                       legend=dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top",
                                   bgcolor="rgba(22,27,34,0.88)", bordercolor="#30363d", borderwidth=1,
                                   font=dict(size=11), tracegroupgap=4))
    return fig

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

def page_dashboard():
    st.title("📈  FCM AI Trading Terminal")
    _health_dot()
    h: HealthTracker = st.session_state["health"]

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        selected = st.multiselect("Watchlist", EQUITY_TICKERS, default=["AAPL","MSFT","NVDA"])
        st.caption("Cache: market 60 s · news 600 s (see Settings)")
        st.subheader("Signal Weights")
        w_fcm  = st.slider("FCM",         0.0, 1.0, 0.5, 0.05)
        w_dl   = st.slider("Deep Learn",  0.0, 1.0, 0.3, 0.05)
        w_sent = st.slider("Sentiment",   0.0, 1.0, 0.2, 0.05)
        st.session_state.signal_gen.update({"fcm": w_fcm, "dl": w_dl, "sent": w_sent})

    # ── Load FCM ──────────────────────────────────────────────────────────
    if st.session_state.fcm_data is None and st.session_state.fcm_err is None:
        if not _FCM_OK:
            st.session_state.fcm_err = _FCM_ERR
        else:
            with st.spinner("Loading FCM + LSTM/GRU models…"):
                try:
                    st.session_state.fcm_data = prepare_data_and_run()
                    h.info_log("FCM", "Loaded OK")
                except Exception as e:
                    st.session_state.fcm_err = traceback.format_exc()
                    h.error("FCM", str(e))

    if st.session_state.fcm_err:
        st.warning("FCM unavailable — showing live data only.", icon="⚠️")

    # ── Market indices strip ──────────────────────────────────────────────
    _section("Market Indices")
    idx_cols = st.columns(len(INDEX_TICKERS))
    for col, (name, sym) in zip(idx_cols, INDEX_TICKERS.items()):
        df = fetch_ohlcv(sym, "5d")
        with col:
            if df.empty or "Close" not in df.columns:
                st.markdown(f'<div class="idx-item"><div class="idx-name">{name}</div><div class="idx-val">—</div></div>', unsafe_allow_html=True)
            else:
                p   = float(df["Close"].iloc[-1])
                p0  = float(df["Close"].iloc[-2]) if len(df) > 1 else p
                chg = (p - p0) / p0 * 100
                chg_col = C_GREEN if chg >= 0 else C_RED
                arrow   = "▲" if chg >= 0 else "▼"
                st.markdown(
                    f'<div class="idx-item">'
                    f'<div class="idx-name">{name}</div>'
                    f'<div class="idx-val">{p:,.2f}</div>'
                    f'<span style="color:{chg_col};font-size:.8em;font-weight:600">'
                    f'{arrow} {abs(chg):.2f}%</span></div>',
                    unsafe_allow_html=True,
                )

    # ── NASDAQ Regime ─────────────────────────────────────────────────────
    nasdaq_df = fetch_ohlcv("^IXIC", "1y")
    if not nasdaq_df.empty:
        regime_label, regime_cls = detect_regime(nasdaq_df["Close"])
        st.markdown(
            f'Market Regime: {_regime_badge(regime_label, regime_cls)}',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── FCM model summary metrics ─────────────────────────────────────────
    fcm  = st.session_state.fcm_data or {}
    mc   = fcm.get("metrics_comparison", {})
    m1, m2, m3, m4, m5 = st.columns(5)
    def _safe(d, k): return round(d.get(k) or 0, 4) if d else None
    fcm_c  = _safe(mc.get("Yahoo FCM"),  "corr")
    lstm_r = _safe(mc.get("LSTM"),       "r2")
    gru_r  = _safe(mc.get("GRU"),        "r2")
    ens_c  = _safe(mc.get("Ensemble"),   "corr")
    pred_h = fcm.get("pred_history_yahoo", [])
    last_s = float(pred_h[-1]) if pred_h else None
    m1.metric("FCM Corr",      f"{fcm_c}"  if fcm_c  is not None else "—")
    m2.metric("LSTM R²",       f"{lstm_r}" if lstm_r is not None else "—")
    m3.metric("GRU R²",        f"{gru_r}"  if gru_r  is not None else "—")
    m4.metric("Ensemble Corr", f"{ens_c}"  if ens_c  is not None else "—")
    m5.metric("FCM Last Score", f"{last_s:.4f}" if last_s is not None else "—")

    # ── Watchlist cards with sparklines ──────────────────────────────────
    _section("Watchlist")
    if not selected:
        st.info("Add tickers in the sidebar.")
        return

    n_cols  = min(4, len(selected))
    rows    = [selected[i:i+n_cols] for i in range(0, len(selected), n_cols)]
    fcm_last = float(pred_h[-1]) if pred_h else 0.0

    for row in rows:
        cols = st.columns(n_cols)
        for col, ticker in zip(cols, row):
            df   = fetch_ohlcv(ticker, "3mo")
            sent = fetch_sentiment_cached(ticker)
            with col:
                if df.empty or "Close" not in df.columns:
                    st.markdown(f'<div class="ticker-card"><b>{ticker}</b><br>No data</div>',
                                unsafe_allow_html=True)
                    continue

                close = df["Close"]
                price = float(close.iloc[-1])
                prev  = float(close.iloc[-2]) if len(close) > 1 else price
                chg   = (price - prev) / prev * 100
                h52   = float(close.tail(252).max())
                l52   = float(close.tail(252).min())
                vol   = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
                avol  = int(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 1
                vol_r = vol / max(avol, 1)

                sig_map, conf_map = st.session_state.signal_gen.signals(
                    {ticker: fcm_last}, sent={ticker: sent.get("polarity", 0)}
                )
                sig  = sig_map.get(ticker, 0)
                conf = conf_map.get(ticker, 0)
                chg_cls  = "tc-chg-pos" if chg >= 0 else "tc-chg-neg"
                arrow    = "▲" if chg >= 0 else "▼"
                spk_col  = C_GREEN if chg >= 0 else C_RED

                st.markdown(
                    f'<div class="ticker-card">'
                    f'<div class="tc-name">{ticker}</div>'
                    f'<div class="tc-price">${price:,.2f}</div>'
                    f'<span class="{chg_cls}">{arrow} {abs(chg):.2f}%</span> '
                    f'&nbsp;{_badge(sig)}<br>'
                    f'<span style="font-size:.7em;color:#8b949e">Conf {conf:.0f}% '
                    f'· Vol {vol_r:.1f}x avg</span><br>'
                    f'{_52w_bar(price,l52,h52)}'
                    f'<span style="font-size:.67em;color:#8b949e">'
                    f'Sent: </span>{_sent_label(sent)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Mini sparkline
                st.plotly_chart(
                    mini_sparkline(close.tail(60), spk_col),
                    width='stretch', config={"displayModeBar": False},
                )

    # ── Prediction chart ──────────────────────────────────────────────────
    if pred_h and len(pred_h) > 10:
        _section("Multi-Model NASDAQ Predictions")
        actual_norm = fcm.get("actual_norm", pd.Series())
        n           = min(len(pred_h), len(actual_norm))
        date_h      = fcm.get("date_history_yahoo", list(range(n)))

        # Compute rolling std of FCM pred as confidence band proxy
        ph_arr = np.array(pred_h)
        band   = pd.Series(ph_arr).rolling(10, min_periods=1).std().fillna(0).values

        extra = []
        lr    = fcm.get("lstm_results")
        if lr and lr[1] is not None and len(lr[1]) > 0:
            n_lr = len(lr[1])
            x_lr = date_h[-n_lr:] if len(date_h) >= n_lr else date_h
            extra.append(go.Scatter(x=x_lr, y=lr[1], name="LSTM", line=dict(color=C_ORANGE, width=1.5)))
        gr = fcm.get("gru_results")
        if gr and gr[1] is not None and len(gr[1]) > 0:
            n_gr = len(gr[1])
            x_gr = date_h[-n_gr:] if len(date_h) >= n_gr else date_h
            extra.append(go.Scatter(x=x_gr, y=gr[1], name="GRU",  line=dict(color=C_PURPLE, width=1.5)))
        en = fcm.get("ensemble_predictions")
        if en is not None and len(en) > 0:
            n_en = len(en)
            x_en = date_h[-n_en:] if len(date_h) >= n_en else date_h
            extra.append(go.Scatter(x=x_en, y=en, name="Ensemble", line=dict(color=C_RED, width=2.5)))

        fig = conf_band_chart(
            dates=date_h,
            actual=actual_norm.values[-n:] if n > 0 else None,
            pred=pred_h[-n:],
            std_band=band[-n:],
            extra_traces=extra,
            title="Normalised NASDAQ Returns — FCM + LSTM + GRU + Ensemble",
            height=380,
        )
        st.plotly_chart(fig, width='stretch')

    # ── Sector heatmap ────────────────────────────────────────────────────
    _section("Sector Performance (1-Month)")
    sector_syms = tuple(SECTOR_ETFS.values())
    sec_data = fetch_multi_close(sector_syms, "1mo")
    if not sec_data.empty:
        sec_rets = sec_data.pct_change().sum() * 100
        sym_to_name = {v: k for k, v in SECTOR_ETFS.items()}
        sec_rets.index = [sym_to_name.get(s, s) for s in sec_rets.index]
        sec_rets = sec_rets.sort_values()
        colors   = [C_GREEN if v >= 0 else C_RED for v in sec_rets.values]
        fig_sec  = go.Figure(go.Bar(
            x=sec_rets.index, y=sec_rets.values,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in sec_rets.values],
            textposition="outside",
        ))
        fig_sec.update_layout(**CHART_THEME, title="Sector 1-Month Returns (%)",
                               height=300, showlegend=False,
                               yaxis=dict(title="Return %"),
                               bargap=0.2)
        st.plotly_chart(fig_sec, width='stretch')


# ============================================================================
# PAGE: MARKET DATA
# ============================================================================

def page_market_data():
    st.title("📊  Market Data")
    _health_dot()

    ticker    = st.sidebar.selectbox("Ticker", EQUITY_TICKERS)
    period    = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","5y"], index=3)
    show_bb   = st.sidebar.checkbox("Bollinger Bands", True)
    show_macd = st.sidebar.checkbox("MACD", True)
    show_vol  = st.sidebar.checkbox("Volume Profile", True)

    data = fetch_download(ticker, period)
    if data.empty:
        st.error(f"No data for {ticker}")
        return

    regime_label, regime_cls = detect_regime(data["Close"])

    # ── Header stats ──────────────────────────────────────────────────────
    close  = data["Close"]
    price  = float(close.iloc[-1])
    prev   = float(close.iloc[-2]) if len(close) > 1 else price
    chg    = (price - prev) / prev * 100
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Price",       f"${price:,.2f}", f"{chg:+.2f}%")
    c2.metric("Period High", f"${close.max():,.2f}")
    c3.metric("Period Low",  f"${close.min():,.2f}")
    c4.metric("Avg Volume",  f"{data['Volume'].mean():,.0f}" if "Volume" in data.columns else "—")
    c5.markdown(f"**Regime:** {_regime_badge(regime_label, regime_cls)}", unsafe_allow_html=True)

    # ── Candlestick + Bollinger ────────────────────────────────────────────
    rows    = 2 if show_vol else 1
    row_h   = [0.75, 0.25] if show_vol else [1.0]
    fig     = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                             row_heights=row_h, vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name=ticker,
        increasing_line_color=C_GREEN, decreasing_line_color=C_RED,
    ), row=1, col=1)

    if show_bb:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2*std20; lower = sma20 - 2*std20
        for y, name, col, dash in [
            (upper, "BB Upper", "rgba(88,166,255,0.5)", "dot"),
            (sma20, "SMA 20",   "rgba(240,136,62,0.9)","solid"),
            (lower, "BB Lower", "rgba(88,166,255,0.5)", "dot"),
        ]:
            fig.add_trace(go.Scatter(x=data.index, y=y, name=name,
                line=dict(color=col, width=1, dash=dash), showlegend=(name=="SMA 20")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=list(data.index)+list(data.index)[::-1],
            y=list(upper)+list(lower)[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.05)",
            line=dict(color="rgba(0,0,0,0)"), name="BB Band", showlegend=False,
        ), row=1, col=1)

    if show_vol and "Volume" in data.columns:
        vol_colors = [C_GREEN if data["Close"].iloc[i] >= data["Open"].iloc[i] else C_RED
                      for i in range(len(data))]
        fig.add_trace(go.Bar(x=data.index, y=data["Volume"],
                              marker_color=vol_colors, name="Volume", opacity=0.6), row=2, col=1)

    fig.update_layout(**CHART_THEME, height=560,
                       xaxis_rangeslider_visible=False, hovermode="x unified",
                       title=dict(text=f"{ticker} — Candlestick / Bollinger / Volume", font=dict(size=13)))
    st.plotly_chart(fig, width='stretch')

    # ── RSI + MACD row ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = 100 - 100 / (1 + gain/(loss+1e-8))
        rsi_v = float(rsi.iloc[-1])
        rsi_c = C_RED if rsi_v > 70 else (C_GREEN if rsi_v < 30 else C_BLUE)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI",
                                    fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
                                    line=dict(color=C_BLUE, width=1.5)))
        fig_r.add_hline(y=70, line_dash="dash", line_color=C_RED,   annotation_text="OB 70")
        fig_r.add_hline(y=30, line_dash="dash", line_color=C_GREEN, annotation_text="OS 30")
        fig_r.add_annotation(x=rsi.index[-1], y=rsi_v, text=f"RSI {rsi_v:.1f}",
                               font=dict(color=rsi_c, size=11), showarrow=False, xshift=30)
        fig_r.update_layout(**CHART_THEME, height=280, title="RSI (14)",
                              yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_r, width='stretch')

    with col2:
        if show_macd:
            ema12  = close.ewm(span=12).mean()
            ema26  = close.ewm(span=26).mean()
            macd   = ema12 - ema26
            sig_l  = macd.ewm(span=9).mean()
            hist_v = macd - sig_l
            bar_c  = [C_GREEN if v >= 0 else C_RED for v in hist_v]
            fig_m  = go.Figure()
            fig_m.add_trace(go.Bar(x=data.index, y=hist_v, name="Hist", marker_color=bar_c, opacity=0.7))
            fig_m.add_trace(go.Scatter(x=data.index, y=macd,  name="MACD",   line=dict(color=C_BLUE)))
            fig_m.add_trace(go.Scatter(x=data.index, y=sig_l, name="Signal", line=dict(color=C_ORANGE)))
            fig_m.update_layout(**CHART_THEME, height=280, title="MACD (12/26/9)", hovermode="x unified")
            st.plotly_chart(fig_m, width='stretch')
        else:
            fig_v2 = go.Figure(go.Bar(x=data.index, y=data.get("Volume", pd.Series()), marker_color=C_BLUE))
            fig_v2.update_layout(**CHART_THEME, height=280, title="Volume")
            st.plotly_chart(fig_v2, width='stretch')

    # ── Return distribution ───────────────────────────────────────────────
    _section("Return Distribution")
    rets = close.pct_change().dropna() * 100
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=rets, nbinsx=60, name="Daily Returns",
                                  marker_color=C_BLUE, opacity=0.75))
    mu, sg = rets.mean(), rets.std()
    fig_d.add_vline(x=mu, line_dash="dash", line_color=C_YELLOW,
                     annotation_text=f"μ={mu:.2f}%")
    fig_d.add_vline(x=mu-2*sg, line_dash="dot", line_color=C_RED, annotation_text="−2σ")
    fig_d.add_vline(x=mu+2*sg, line_dash="dot", line_color=C_GREEN, annotation_text="+2σ")
    fig_d.update_layout(**CHART_THEME, height=260, title="Daily Return Distribution",
                         xaxis_title="Return (%)", bargap=0.04)
    st.plotly_chart(fig_d, width='stretch')


# ============================================================================
# PAGE: COMMODITIES
# ============================================================================

def page_commodities():
    st.title("🛢️  Commodities & Crypto")
    _health_dot()

    period  = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y"], index=2, key="cp")
    refresh = st.sidebar.button("Refresh All")
    if refresh:
        st.cache_data.clear()

    # ── Price grid with sparklines ────────────────────────────────────────
    _section("Current Prices")
    items  = list(COMMODITY_TICKERS.items())
    n_cols = 4
    for row_start in range(0, len(items), n_cols):
        row  = items[row_start: row_start + n_cols]
        cols = st.columns(n_cols)
        for col, (name, sym) in zip(cols, row):
            df = fetch_ohlcv(sym, "3mo")
            with col:
                if df.empty or "Close" not in df.columns:
                    st.markdown(f'<div class="ticker-card"><div class="tc-name">{name}</div>'
                                f'<div style="color:#8b949e">No data</div></div>', unsafe_allow_html=True)
                    continue
                close = df["Close"]
                p    = float(close.iloc[-1])
                p0   = float(close.iloc[-2]) if len(close) > 1 else p
                chg  = (p - p0) / p0 * 100
                chg_c = C_GREEN if chg >= 0 else C_RED
                arrow = "▲" if chg >= 0 else "▼"
                h52  = float(close.tail(252).max())
                l52  = float(close.tail(252).min())
                st.markdown(
                    f'<div class="ticker-card">'
                    f'<div class="tc-name">{name} <code style="font-size:.65em;color:#8b949e">{sym}</code></div>'
                    f'<div class="tc-price">${p:,.2f}</div>'
                    f'<span style="color:{chg_c};font-size:.9em;font-weight:600">{arrow} {abs(chg):.2f}%</span><br>'
                    f'{_52w_bar(p, l52, h52)}</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(mini_sparkline(close.tail(60), chg_c, height=50),
                                width='stretch', config={"displayModeBar": False})

    st.divider()

    # ── Comparative normalised performance ───────────────────────────────
    _section("Comparative Performance (Normalised to 100)")
    syms = tuple(COMMODITY_TICKERS.values())
    mc   = fetch_multi_close(syms, period)
    if not mc.empty:
        norm = mc / mc.iloc[0] * 100
        sym_to_name = {v: k for k, v in COMMODITY_TICKERS.items()}
        fig_comp = go.Figure()
        palette  = [C_YELLOW, C_RED, C_GREY, C_ORANGE, C_BLUE, C_GREEN, C_PURPLE, "#adbac7"]
        for i, col in enumerate(norm.columns):
            fig_comp.add_trace(go.Scatter(
                x=norm.index, y=norm[col],
                name=sym_to_name.get(col, col),
                line=dict(color=palette[i % len(palette)], width=1.8),
            ))
        fig_comp.add_hline(y=100, line_dash="dot", line_color="#3d444d")
        fig_comp.update_layout(**CHART_THEME, height=380,
                                title=f"Normalised Performance — {period}",
                                yaxis_title="Rebased to 100", hovermode="x unified")
        st.plotly_chart(fig_comp, width='stretch')

    # ── Detail chart ──────────────────────────────────────────────────────
    _section("Detail Chart")
    sel_n = st.selectbox("Select Asset", list(COMMODITY_TICKERS.keys()))
    sel_s = COMMODITY_TICKERS[sel_n]
    det   = fetch_download(sel_s, period)
    if det.empty:
        st.warning(f"No data for {sel_s}")
        return
    close  = det["Close"].squeeze()
    sma20  = close.rolling(20).mean()
    ema9   = close.ewm(span=9).mean()
    fig_d  = go.Figure()
    fig_d.add_trace(go.Scatter(x=close.index, y=close, name=sel_n,
                                fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
                                line=dict(color=C_BLUE, width=2)))
    fig_d.add_trace(go.Scatter(x=sma20.index, y=sma20, name="SMA 20",
                                line=dict(color=C_ORANGE, dash="dot", width=1.3)))
    fig_d.add_trace(go.Scatter(x=ema9.index, y=ema9, name="EMA 9",
                                line=dict(color=C_PURPLE, dash="dash", width=1.3)))
    fig_d.update_layout(**CHART_THEME, height=420,
                         title=f"{sel_n} ({sel_s}) — {period}",
                         yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig_d, width='stretch')

    # ── Correlation heatmap ────────────────────────────────────────────────
    if not mc.empty:
        _section("Cross-Asset Return Correlation")
        sym_to_name = {v: k for k, v in COMMODITY_TICKERS.items()}
        corr_df = mc.pct_change().dropna().corr()
        corr_df.index   = [sym_to_name.get(s,s) for s in corr_df.index]
        corr_df.columns = [sym_to_name.get(s,s) for s in corr_df.columns]
        fig_c = px.imshow(corr_df, text_auto=".2f",
                           color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_c.update_layout(**CHART_THEME, height=380, title="Commodity Return Correlation")
        st.plotly_chart(fig_c, width='stretch')

    # ── Cross-Asset Interaction Network ───────────────────────────────────
    st.divider()
    _section("Cross-Asset Interaction Network (Stocks × Commodities × Indices)")
    with st.spinner("Building cross-asset correlation matrix…"):
        try:
            cross_tickers = {
                # Indices
                "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "VIX": "^VIX", "DXY": "DX-Y.NYB",
                # Commodities
                "Gold": "GC=F", "Oil": "CL=F", "Silver": "SI=F",
                "Nat Gas": "NG=F", "Copper": "HG=F",
                "BTC": "BTC-USD", "ETH": "ETH-USD",
                # Equities
                "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA",
                "GOOG": "GOOG", "AMZN": "AMZN", "TSLA": "TSLA",
            }
            cross_syms  = tuple(cross_tickers.values())
            cross_names = list(cross_tickers.keys())
            cross_data  = fetch_multi_close(cross_syms, period)
            if not cross_data.empty:
                sym_to_name = {v: k for k, v in cross_tickers.items()}
                cross_data.columns = [sym_to_name.get(c, c) for c in cross_data.columns]
                cross_corr = cross_data.pct_change().dropna().corr()
                # group order: indices, commodities, equities
                grp_order = ["S&P 500","NASDAQ","VIX","DXY",
                             "Gold","Oil","Silver","Nat Gas","Copper","BTC","ETH",
                             "AAPL","MSFT","NVDA","GOOG","AMZN","TSLA"]
                available_cols = [c for c in grp_order if c in cross_corr.columns]
                cross_corr = cross_corr.loc[available_cols, available_cols]
                fig_cross = px.imshow(
                    cross_corr, text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"Cross-Asset Return Correlation — {period}",
                )
                # Add group divider lines
                n_idx = len([c for c in available_cols if c in ["S&P 500","NASDAQ","VIX","DXY"]])
                n_com = len([c for c in available_cols if c in ["Gold","Oil","Silver","Nat Gas","Copper","BTC","ETH"]])
                for boundary in [n_idx - 0.5, n_idx + n_com - 0.5]:
                    fig_cross.add_shape(type="line", x0=boundary, x1=boundary, y0=-0.5, y1=len(available_cols)-0.5,
                                         line=dict(color="#e3b341", width=2, dash="dot"))
                    fig_cross.add_shape(type="line", y0=boundary, y1=boundary, x0=-0.5, x1=len(available_cols)-0.5,
                                         line=dict(color="#e3b341", width=2, dash="dot"))
                fig_cross.update_layout(**CHART_THEME, height=560)
                st.plotly_chart(fig_cross, width='stretch')
                st.caption("Yellow dashed lines separate: Indices | Commodities | Equities")
        except Exception as e:
            st.warning(f"Cross-asset correlation unavailable: {e}")

    # ── FCM Node Weights × Market Factors ────────────────────────────────
    _section("FCM Node Weights — Stocks / Market / Commodity Factors")
    fcm_d = st.session_state.get("fcm_data")
    if not fcm_d:
        st.info("Load FCM models on the Dashboard to see node weight interactions.")
    else:
        wh         = fcm_d.get("weights_history_yahoo", [])
        node_order = fcm_d.get("node_order", [])
        if wh:
            # Show weight evolution: how key weights evolved over time
            # Extract NASDAQ-incoming weights over the weight history
            target_incoming = []
            dates_wh = fcm_d.get("date_history_yahoo", list(range(len(wh))))
            for snap in wh:
                row = {src: snap.get(src, {}).get("NASDAQ", 0.0) for src in node_order
                       if src not in ["Monetary_Policy","Inflation","Corporate_Earnings","Investor_Sentiment","NASDAQ"]}
                target_incoming.append(row)
            if target_incoming:
                tw_df = pd.DataFrame(target_incoming, index=dates_wh if len(dates_wh) == len(target_incoming) else range(len(target_incoming)))
                # Plot top nodes by final weight magnitude
                final_mag = tw_df.iloc[-1].abs().sort_values(ascending=False)
                top_nodes = final_mag.head(8).index.tolist()
                fig_wev = go.Figure()
                colors_w = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_YELLOW, C_GREY, "#adbac7"]
                for i, node in enumerate(top_nodes):
                    fig_wev.add_trace(go.Scatter(
                        x=list(tw_df.index), y=tw_df[node].values,
                        name=node.replace("_", " "),
                        line=dict(color=colors_w[i % len(colors_w)], width=1.8),
                    ))
                fig_wev.add_hline(y=0, line_dash="dot", line_color="#3d444d")
                fig_wev.update_layout(**CHART_THEME, height=360,
                                       title="FCM Edge Weight Evolution — NASDAQ-Incoming Weights",
                                       yaxis_title="Weight", hovermode="x unified",
                                       legend=dict(orientation="h", y=1.02, font=dict(size=10)))
                st.plotly_chart(fig_wev, width='stretch')

            # Final weight bar chart grouped by source
            final_w = wh[-1]
            w_mat   = _fcm_weights_matrix(final_w, node_order)
            if not w_mat.empty:
                fig_wbar = go.Figure()
                palette  = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_YELLOW]
                for i, tgt in enumerate(w_mat.columns):
                    fig_wbar.add_trace(go.Bar(
                        x=[n.replace("_"," ") for n in w_mat.index],
                        y=w_mat[tgt].values,
                        name=tgt.replace("_", " "),
                        marker_color=palette[i % len(palette)],
                    ))
                fig_wbar.update_layout(**CHART_THEME, height=380, barmode="group",
                                        title="Final FCM Weights by Source → Target",
                                        xaxis_title="Source Node", yaxis_title="Weight",
                                        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
                                        xaxis_tickangle=-30)
                st.plotly_chart(fig_wbar, width='stretch')


# ============================================================================
# PAGE: NEWS & SENTIMENT
# ============================================================================

def page_news_sentiment():
    st.title("📰  News & Sentiment Analysis")
    _health_dot()

    # ── Sidebar controls ─────────────────────────────────────────────────
    ticker  = st.sidebar.selectbox("Focus Ticker", EQUITY_TICKERS, key="ns_tick")
    _cmp_defaults = [t for t in ["AAPL","MSFT","NVDA","GOOG","AMZN","TSLA"] if t in EQUITY_TICKERS][:5]
    compare = st.sidebar.multiselect("Compare Tickers", EQUITY_TICKERS,
                                     default=_cmp_defaults, key="ns_cmp")
    if st.sidebar.button("🔄 Refresh News", key="ns_refresh"):
        st.cache_data.clear()
        st.rerun()

    articles = fetch_news_cached(ticker)
    sent     = fetch_sentiment_cached(ticker)
    pol      = sent.get("polarity", 0.0)
    pols_list = sent.get("polarities", [])

    # ── Top KPI strip ────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    sent_label = sent.get("label", "NEUTRAL")
    sent_color = C_GREEN if sent_label == "BULLISH" else (C_RED if sent_label == "BEARISH" else C_BLUE)
    k1.metric("Sentiment",    sent_label)
    k2.metric("Polarity",     f"{pol:+.3f}")
    k3.metric("Intensity",    f"{sent.get('intensity',0):.3f}")
    k4.metric("Articles",     sent.get("articles_count", 0))
    bullish_pct = sum(1 for p in pols_list if p > 0.05) / max(len(pols_list), 1) * 100
    k5.metric("Bullish Articles", f"{bullish_pct:.0f}%")

    st.markdown(
        f'<div style="height:3px;background:linear-gradient(90deg,{C_GREEN},{C_BLUE},{C_PURPLE});'
        f'border-radius:2px;margin:6px 0 18px 0"></div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_ov, tab_feed, tab_cmp, tab_mkt, tab_poly = st.tabs([
        "📊 Overview", f"📰 {ticker} Feed", "⚖️ Multi-Ticker", "🌐 Market Pulse", "📡 Live Wire"
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════
    with tab_ov:
        col_g, col_hist, col_src = st.columns([1, 2, 1])

        # Sentiment gauge
        with col_g:
            bar_c = C_GREEN if pol > 0 else C_RED
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pol,
                delta={"reference": 0, "valueformat": ".3f"},
                number={"valueformat": ".3f", "font": {"size": 26}},
                gauge={
                    "axis": {"range": [-1, 1], "tickcolor": C_GREY},
                    "bar":  {"color": bar_c, "thickness": 0.28},
                    "bgcolor": "#1c2230",
                    "steps": [
                        {"range": [-1, -0.05], "color": "rgba(255,71,87,.15)"},
                        {"range": [-0.05, 0.05], "color": "rgba(139,148,158,.1)"},
                        {"range": [0.05, 1],    "color": "rgba(0,214,143,.15)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": pol},
                },
                title={"text": f"<b>{ticker}</b> Sentiment", "font": {"size": 13}},
            ))
            fig_g.update_layout(**CHART_THEME, height=270)
            st.plotly_chart(fig_g, width='stretch')

        # Polarity distribution histogram
        with col_hist:
            if pols_list:
                bull_c = [C_GREEN if p > 0.05 else (C_RED if p < -0.05 else C_GREY) for p in pols_list]
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=pols_list, nbinsx=20, opacity=0.8,
                    marker=dict(color=C_BLUE, line=dict(color="#0d1117", width=0.5)),
                    name="Article Polarities",
                ))
                fig_h.add_vline(x=0, line_dash="solid", line_color=C_GREY, line_width=1)
                fig_h.add_vline(x=pol, line_dash="dash", line_color=C_YELLOW,
                                 annotation_text=f"μ={pol:+.3f}", annotation_position="top right")
                fig_h.update_layout(**CHART_THEME, height=270,
                                     title="Polarity Distribution Across Articles",
                                     xaxis_title="Sentiment Polarity", bargap=0.06)
                st.plotly_chart(fig_h, width='stretch')
            else:
                st.info("No polarity data available.")

        # Source breakdown
        with col_src:
            src_counts: Dict[str, int] = {}
            for a in articles:
                s = a.get("source", "Unknown") or "Unknown"
                src_counts[s] = src_counts.get(s, 0) + 1
            if src_counts:
                src_df = pd.DataFrame({"Source": list(src_counts.keys()),
                                        "Count":  list(src_counts.values())})
                fig_src = go.Figure(go.Pie(
                    labels=src_df["Source"], values=src_df["Count"], hole=0.45,
                    textinfo="percent", textfont=dict(size=9),
                    marker=dict(colors=[C_BLUE, C_GREEN, C_PURPLE, C_ORANGE, C_YELLOW, C_RED]),
                ))
                fig_src.update_layout(**CHART_THEME, height=270,
                                       title="News Source Breakdown",
                                       legend=dict(font=dict(size=9), x=0))
                st.plotly_chart(fig_src, width='stretch')

        # Sentiment over time (from article timestamps)
        _section("Sentiment Timeline — Article Polarity Over Time")
        if articles and pols_list:
            try:
                from textblob import TextBlob
                time_rows = []
                for a in articles:
                    ts    = a.get("ts")
                    title = (a.get("title") or "") + " " + (a.get("summary") or "")
                    try:
                        ap = TextBlob(title.strip()).sentiment.polarity
                    except Exception:
                        ap = 0.0
                    if ts:
                        time_rows.append({"dt": datetime.fromtimestamp(int(ts)), "polarity": ap,
                                          "title": a.get("title","")[:60], "source": a.get("source","")})
                if time_rows:
                    tdf = pd.DataFrame(time_rows).sort_values("dt")
                    tdf["roll"] = tdf["polarity"].rolling(window=3, min_periods=1).mean()
                    fig_tl = go.Figure()
                    colors_tl = [C_GREEN if p > 0.05 else (C_RED if p < -0.05 else C_GREY)
                                  for p in tdf["polarity"]]
                    fig_tl.add_trace(go.Bar(
                        x=tdf["dt"], y=tdf["polarity"], marker_color=colors_tl,
                        opacity=0.65, name="Per-Article", text=tdf["title"],
                        hovertemplate="<b>%{text}</b><br>Polarity: %{y:.3f}<extra></extra>",
                    ))
                    fig_tl.add_trace(go.Scatter(
                        x=tdf["dt"], y=tdf["roll"], line=dict(color=C_YELLOW, width=2),
                        name="3-article rolling avg",
                    ))
                    fig_tl.add_hline(y=0, line_color=C_GREY, line_width=1)
                    fig_tl.update_layout(**CHART_THEME, height=280,
                                          title=f"{ticker} — Article Sentiment Over Time",
                                          xaxis_title="Published", yaxis_title="Polarity",
                                          hovermode="x unified", barmode="relative")
                    st.plotly_chart(fig_tl, width='stretch')
            except ImportError:
                st.info("Install textblob for timeline chart.")

        # Sentiment vs Price overlay
        _section("Sentiment vs Price — 30-Day Overlay")
        try:
            price_df = fetch_ohlcv(ticker, "1mo")
            if not price_df.empty and "Close" in price_df.columns and pols_list:
                price_30 = price_df["Close"].tail(30)
                # normalise price to 0-1 for overlay
                pn = (price_30 - price_30.min()) / (price_30.max() - price_30.min() + 1e-9)
                fig_ov = go.Figure()
                fig_ov.add_trace(go.Scatter(
                    x=price_30.index, y=price_30.values,
                    name=f"{ticker} Price", line=dict(color=C_BLUE, width=2), yaxis="y1",
                ))
                # horizontal sentiment bar across the whole chart
                fig_ov.add_hrect(y0=price_30.min(), y1=price_30.max() if pol > 0 else price_30.min(),
                                   fillcolor=C_GREEN if pol > 0 else C_RED,
                                   opacity=0.04, line_width=0, yref="y")
                fig_ov.add_annotation(
                    x=price_30.index[-1], y=price_30.iloc[-1],
                    text=f"Sentiment: {pol:+.3f}",
                    font=dict(color=sent_color, size=11),
                    showarrow=True, arrowcolor=sent_color, ax=60, ay=0,
                )
                fig_ov.update_layout(**CHART_THEME, height=280,
                                      title=f"{ticker} — Price + Sentiment Background (30d)",
                                      yaxis_title="Price ($)", hovermode="x unified")
                st.plotly_chart(fig_ov, width='stretch')
        except Exception:
            pass

        # Top keywords from article titles
        _section("Top Keywords in Recent Headlines")
        try:
            STOPWORDS = {"the","a","an","in","of","for","and","or","to","is","on","at","its",
                          "with","by","as","from","it","be","has","are","was","that","this",
                          "will","have","not","new","after","over","says","could","about","into",
                          "up","he","she","they","we","you","how","what","when","where","who"}
            word_freq: Dict[str, int] = {}
            for a in articles:
                words = re.sub(r"[^a-zA-Z ]", " ", (a.get("title") or "").lower()).split()
                for w in words:
                    if len(w) > 3 and w not in STOPWORDS:
                        word_freq[w] = word_freq.get(w, 0) + 1
            if word_freq:
                top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:20]
                wdf = pd.DataFrame(top_words, columns=["Word","Count"])
                fig_kw = go.Figure(go.Bar(
                    x=wdf["Count"], y=wdf["Word"], orientation="h",
                    marker_color=C_PURPLE, opacity=0.85,
                    text=wdf["Count"], textposition="outside",
                ))
                fig_kw.update_layout(**CHART_THEME, height=420,
                                      title="Top Keywords in Recent Headlines",
                                      xaxis_title="Frequency", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_kw, width='stretch')
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — TICKER ARTICLE FEED
    # ════════════════════════════════════════════════════════════════════
    with tab_feed:
        _section(f"Recent Headlines — {ticker}")

        # Sort controls
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            sort_by = st.radio("Sort by", ["Newest First", "Most Bullish", "Most Bearish"],
                                horizontal=True, key="ns_sort")
        with sc2:
            show_n = st.slider("Articles to show", 5, 20, 15, key="ns_show")

        if not articles:
            st.info("No news available. Check back later or try another ticker.")
        else:
            try:
                from textblob import TextBlob
                enriched = []
                for a in articles:
                    title   = a.get("title", "Untitled")
                    summary = a.get("summary", "")
                    ap = TextBlob(title + " " + summary).sentiment.polarity
                    enriched.append({**a, "_pol": ap})
            except ImportError:
                enriched = [{**a, "_pol": 0.0} for a in articles]

            if sort_by == "Most Bullish":
                enriched = sorted(enriched, key=lambda x: -x["_pol"])
            elif sort_by == "Most Bearish":
                enriched = sorted(enriched, key=lambda x: x["_pol"])
            else:
                enriched = sorted(enriched, key=lambda x: -(x.get("ts") or 0))

            for article in enriched[:show_n]:
                title   = article.get("title", "Untitled")
                source  = article.get("source", "Unknown")
                link    = article.get("link", "#")
                summary = article.get("summary", "")
                pub_ts  = article.get("ts")
                pub_str = datetime.fromtimestamp(int(pub_ts)).strftime("%b %d %Y, %H:%M") if pub_ts else "—"
                ap      = article["_pol"]
                ap_col  = C_GREEN if ap > 0.05 else (C_RED if ap < -0.05 else C_GREY)
                ap_lbl  = "BULLISH" if ap > 0.05 else ("BEARISH" if ap < -0.05 else "NEUTRAL")
                ap_icon = "🟢" if ap > 0.05 else ("🔴" if ap < -0.05 else "⚪")

                with st.expander(f"{ap_icon} {title[:85]}{'…' if len(title) > 85 else ''}", expanded=False):
                    col_meta, col_score = st.columns([3, 1])
                    with col_meta:
                        st.markdown(f"**Source:** {source} &nbsp;·&nbsp; **Published:** {pub_str}")
                    with col_score:
                        st.markdown(
                            f'<div style="text-align:right;font-size:.9em;font-weight:700;color:{ap_col}">'
                            f'{ap_lbl} ({ap:+.3f})</div>',
                            unsafe_allow_html=True,
                        )
                    if summary:
                        st.write(summary)
                    else:
                        st.caption("No article summary available from this source.")
                    st.markdown(f"[Read Full Article ↗]({link})")

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — MULTI-TICKER COMPARISON
    # ════════════════════════════════════════════════════════════════════
    with tab_cmp:
        if not compare:
            st.info("Select tickers to compare in the sidebar.")
        else:
            _section("Cross-Ticker Sentiment Leaderboard")
            cmp_data = []
            with st.spinner("Fetching sentiment for all comparison tickers…"):
                all_tickers = list(dict.fromkeys([ticker] + compare))
                for t in all_tickers:
                    s = fetch_sentiment_cached(t)
                    pols_t = s.get("polarities", [])
                    bullish_t = sum(1 for p in pols_t if p > 0.05) / max(len(pols_t), 1) * 100
                    bearish_t = sum(1 for p in pols_t if p < -0.05) / max(len(pols_t), 1) * 100
                    cmp_data.append({
                        "Ticker":    t,
                        "Polarity":  round(s.get("polarity", 0.0), 4),
                        "Intensity": round(s.get("intensity", 0.0), 4),
                        "Articles":  s.get("articles_count", 0),
                        "Label":     s.get("label", "NEUTRAL"),
                        "Bullish%":  round(bullish_t, 1),
                        "Bearish%":  round(bearish_t, 1),
                    })

            cmp_df = pd.DataFrame(cmp_data).sort_values("Polarity", ascending=False)

            # Polarity bar
            bar_cs = [C_GREEN if v > 0.05 else (C_RED if v < -0.05 else C_GREY) for v in cmp_df["Polarity"]]
            fig_cmp = go.Figure(go.Bar(
                x=cmp_df["Ticker"], y=cmp_df["Polarity"],
                marker_color=bar_cs, opacity=0.85,
                text=[f"{v:+.3f}" for v in cmp_df["Polarity"]],
                textposition="outside",
            ))
            fig_cmp.add_hline(y=0, line_color=C_GREY, line_width=1)
            fig_cmp.update_layout(**CHART_THEME, height=300,
                                   title="Sentiment Polarity — Cross-Ticker Comparison",
                                   yaxis_title="Polarity", yaxis=dict(range=[-1, 1]), bargap=0.25)
            st.plotly_chart(fig_cmp, width='stretch')

            # Bullish vs Bearish stacked
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Bar(
                x=cmp_df["Ticker"], y=cmp_df["Bullish%"],
                name="Bullish %", marker_color=C_GREEN, opacity=0.8,
            ))
            fig_bb.add_trace(go.Bar(
                x=cmp_df["Ticker"], y=[-v for v in cmp_df["Bearish%"]],
                name="Bearish %", marker_color=C_RED, opacity=0.8,
            ))
            fig_bb.add_hline(y=0, line_color=C_GREY, line_width=1)
            fig_bb.update_layout(**CHART_THEME, height=280, barmode="relative",
                                   title="Bullish vs Bearish Article Split (%)",
                                   yaxis_title="% of Articles", bargap=0.25)
            st.plotly_chart(fig_bb, width='stretch')

            # Intensity scatter
            fig_sc = go.Figure(go.Scatter(
                x=cmp_df["Polarity"], y=cmp_df["Intensity"],
                mode="markers+text",
                text=cmp_df["Ticker"], textposition="top center",
                marker=dict(
                    size=cmp_df["Articles"].clip(5, 30),
                    color=cmp_df["Polarity"], colorscale="RdYlGn",
                    showscale=True, cmin=-0.5, cmax=0.5,
                    colorbar=dict(title="Polarity", thickness=12),
                ),
            ))
            fig_sc.add_vline(x=0, line_color=C_GREY, line_width=1, line_dash="dash")
            fig_sc.update_layout(**CHART_THEME, height=340,
                                   title="Polarity × Intensity Scatter (bubble = article count)",
                                   xaxis_title="Sentiment Polarity",
                                   yaxis_title="Sentiment Intensity (std dev)")
            st.plotly_chart(fig_sc, width='stretch')

            # Data table
            _section("Detailed Sentiment Table")
            st.dataframe(
                cmp_df.style
                    .background_gradient(subset=["Polarity"], cmap="RdYlGn", vmin=-0.5, vmax=0.5)
                    .format({"Polarity": "{:+.4f}", "Intensity": "{:.4f}",
                              "Bullish%": "{:.1f}%", "Bearish%": "{:.1f}%"}),
                width='stretch', hide_index=True,
            )

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — MARKET-WIDE SENTIMENT PULSE
    # ════════════════════════════════════════════════════════════════════
    with tab_mkt:
        _section("Market-Wide Sentiment Pulse — All Tracked Equities")
        st.caption("Aggregate sentiment across all tracked equity tickers. Size = article count.")

        all_eq = list(EQUITY_TICKERS)
        mkt_data = []
        with st.spinner("Building market sentiment map…"):
            for t in all_eq:
                s = fetch_sentiment_cached(t)
                pols_t = s.get("polarities", [])
                bull_pct = sum(1 for p in pols_t if p > 0.05) / max(len(pols_t), 1) * 100
                mkt_data.append({
                    "Ticker":    t,
                    "Polarity":  s.get("polarity", 0.0),
                    "Intensity": s.get("intensity", 0.0),
                    "Articles":  max(s.get("articles_count", 0), 1),
                    "Label":     s.get("label", "NEUTRAL"),
                    "Bull%":     bull_pct,
                })

        mkt_df = pd.DataFrame(mkt_data)
        avg_pol = mkt_df["Polarity"].mean()
        bull_n  = (mkt_df["Label"] == "BULLISH").sum()
        bear_n  = (mkt_df["Label"] == "BEARISH").sum()
        neu_n   = (mkt_df["Label"] == "NEUTRAL").sum()

        mk1, mk2, mk3, mk4 = st.columns(4)
        mc = C_GREEN if avg_pol > 0.02 else (C_RED if avg_pol < -0.02 else C_BLUE)
        mk1.metric("Market Avg Polarity", f"{avg_pol:+.3f}")
        mk2.metric("🟢 Bullish Tickers",  bull_n)
        mk3.metric("🔴 Bearish Tickers",  bear_n)
        mk4.metric("⚪ Neutral Tickers",  neu_n)

        # Heatmap style horizontal bar
        mkt_sorted = mkt_df.sort_values("Polarity")
        bar_colors = [C_GREEN if v > 0.05 else (C_RED if v < -0.05 else C_GREY)
                       for v in mkt_sorted["Polarity"]]
        fig_mkt = go.Figure(go.Bar(
            x=mkt_sorted["Polarity"], y=mkt_sorted["Ticker"],
            orientation="h", marker_color=bar_colors, opacity=0.85,
            text=[f"{v:+.3f}" for v in mkt_sorted["Polarity"]],
            textposition="outside",
        ))
        fig_mkt.add_vline(x=0, line_color=C_GREY, line_width=1)
        fig_mkt.update_layout(**CHART_THEME, height=max(340, len(all_eq) * 22 + 80),
                               title="All Equity Tickers — Sentiment Polarity Ranking",
                               xaxis_title="Sentiment Polarity",
                               xaxis=dict(range=[-0.8, 0.8]))
        st.plotly_chart(fig_mkt, width='stretch')

        # Scatter: polarity vs intensity with article size bubble
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_bubble = go.Figure(go.Scatter(
                x=mkt_df["Polarity"], y=mkt_df["Intensity"],
                mode="markers+text",
                text=mkt_df["Ticker"], textposition="top center",
                marker=dict(
                    size=(mkt_df["Articles"].clip(1, 20) * 2.5).tolist(),
                    color=mkt_df["Polarity"], colorscale="RdYlGn",
                    showscale=True, cmin=-0.4, cmax=0.4,
                    colorbar=dict(title="Polarity", thickness=10),
                ),
                hovertemplate="<b>%{text}</b><br>Polarity: %{x:.3f}<br>Intensity: %{y:.3f}<extra></extra>",
            ))
            fig_bubble.add_vline(x=0, line_dash="dash", line_color=C_GREY, line_width=1)
            fig_bubble.update_layout(**CHART_THEME, height=380,
                                      title="Sentiment Scatter — Polarity × Intensity",
                                      xaxis_title="Polarity", yaxis_title="Intensity")
            st.plotly_chart(fig_bubble, width='stretch')

        with col_s2:
            # Pie: bullish / neutral / bearish distribution
            fig_dist = go.Figure(go.Pie(
                labels=["Bullish", "Neutral", "Bearish"],
                values=[bull_n, neu_n, bear_n],
                marker=dict(colors=[C_GREEN, C_BLUE, C_RED]),
                hole=0.5, textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig_dist.update_layout(**CHART_THEME, height=380,
                                    title="Equity Universe Sentiment Distribution")
            st.plotly_chart(fig_dist, width='stretch')

        # FCM sentiment overlay if available
        fcm_data = st.session_state.get("fcm_data", {})
        poly_y = fcm_data.get("polarity_series_yahoo", pd.Series())
        poly_g = fcm_data.get("polarity_series_google", pd.Series())
        if len(poly_y) > 5 or len(poly_g) > 5:
            _section("FCM Sentiment Series — Yahoo vs Google")
            fig_fcm = go.Figure()
            if len(poly_y) > 5:
                fig_fcm.add_trace(go.Scatter(
                    x=poly_y.index, y=poly_y.values,
                    line=dict(color=C_BLUE, width=1.8), name="Yahoo FCM Sentiment",
                    fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
                ))
            if len(poly_g) > 5:
                fig_fcm.add_trace(go.Scatter(
                    x=poly_g.index, y=poly_g.values,
                    line=dict(color=C_PURPLE, width=1.8), name="Google FCM Sentiment",
                    fill="tozeroy", fillcolor="rgba(188,140,255,0.07)",
                ))
            fig_fcm.add_hline(y=0, line_color=C_GREY, line_width=1)
            fig_fcm.update_layout(**CHART_THEME, height=260,
                                   title="FCM Sentiment Series (Daily Polarity)",
                                   yaxis_title="Polarity", hovermode="x unified")
            st.plotly_chart(fig_fcm, width='stretch')

    # ════════════════════════════════════════════════════════════════════
    # TAB 5 — POLYGON LIVE NEWS WIRE
    # ════════════════════════════════════════════════════════════════════
    with tab_poly:
        _section("Live Market News Wire — Polygon.io")
        poly_news = fetch_polygon_news_feed(limit=25)

        if not poly_news:
            st.info("No Polygon news data. Check POLYGON_API_KEY in Settings.")
        else:
            # Sentiment-score each headline
            try:
                from textblob import TextBlob
                for item in poly_news:
                    text = (item.get("title") or "") + " " + (item.get("description") or "")
                    item["_pol"] = TextBlob(text).sentiment.polarity
            except ImportError:
                for item in poly_news:
                    item["_pol"] = 0.0

            # Aggregate summary
            all_pols = [i["_pol"] for i in poly_news]
            lw1, lw2, lw3 = st.columns(3)
            avg_lw = float(np.mean(all_pols)) if all_pols else 0.0
            lw1.metric("Live Feed Avg Sentiment", f"{avg_lw:+.3f}")
            lw2.metric("Bullish Headlines",
                        sum(1 for p in all_pols if p > 0.05))
            lw3.metric("Bearish Headlines",
                        sum(1 for p in all_pols if p < -0.05))

            # Timeline bar
            sorted_news = sorted(poly_news, key=lambda x: x.get("published_utc", ""))
            if sorted_news:
                pub_dts = []
                for item in sorted_news:
                    raw = item.get("published_utc", "")
                    try:
                        pub_dts.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
                    except Exception:
                        pub_dts.append(datetime.now(timezone.utc))

                lw_colors = [C_GREEN if p > 0.05 else (C_RED if p < -0.05 else C_GREY)
                              for p in [i["_pol"] for i in sorted_news]]
                fig_lw = go.Figure(go.Bar(
                    x=pub_dts,
                    y=[i["_pol"] for i in sorted_news],
                    marker_color=lw_colors, opacity=0.8,
                    text=[i.get("title", "")[:40] for i in sorted_news],
                    hovertemplate="<b>%{text}</b><br>Sentiment: %{y:.3f}<extra></extra>",
                ))
                fig_lw.add_hline(y=0, line_color=C_GREY, line_width=1)
                fig_lw.update_layout(**CHART_THEME, height=260,
                                      title="Polygon Live Feed — Headline Sentiment Timeline",
                                      xaxis_title="Published", yaxis_title="Polarity",
                                      hovermode="x unified")
                st.plotly_chart(fig_lw, width='stretch')

            # Article cards
            sort_lw = st.radio("Sort", ["Newest", "Most Bullish", "Most Bearish"],
                                horizontal=True, key="ns_lw_sort")
            if sort_lw == "Most Bullish":
                poly_news = sorted(poly_news, key=lambda x: -x["_pol"])
            elif sort_lw == "Most Bearish":
                poly_news = sorted(poly_news, key=lambda x: x["_pol"])

            for item in poly_news[:15]:
                title     = item.get("title", "Untitled")
                publisher = (item.get("publisher") or {}).get("name", "Unknown")
                link      = item.get("article_url", "#")
                pub_dt    = item.get("published_utc", "")
                desc      = item.get("description", "")
                tickers_n = item.get("tickers", [])
                ap        = item["_pol"]
                ap_icon   = "🟢" if ap > 0.05 else ("🔴" if ap < -0.05 else "⚪")
                ap_col    = C_GREEN if ap > 0.05 else (C_RED if ap < -0.05 else C_GREY)
                try:
                    dt_str = datetime.fromisoformat(pub_dt.replace("Z","+00:00")).strftime("%b %d, %H:%M UTC")
                except Exception:
                    dt_str = pub_dt[:16] if pub_dt else "—"

                with st.expander(f"{ap_icon} {title[:85]}{'…' if len(title)>85 else ''}", expanded=False):
                    col_m, col_p = st.columns([3, 1])
                    with col_m:
                        st.markdown(f"**Publisher:** {publisher} &nbsp;·&nbsp; **Published:** {dt_str}")
                        if tickers_n:
                            st.markdown(f"**Related Tickers:** {', '.join(tickers_n[:8])}")
                    with col_p:
                        st.markdown(
                            f'<div style="text-align:right;font-size:.92em;font-weight:700;color:{ap_col}">'
                            f'{ap:+.3f}</div>',
                            unsafe_allow_html=True,
                        )
                    if desc:
                        st.write(desc[:400] + ("…" if len(desc) > 400 else ""))
                    st.markdown(f"[Read Full Article ↗]({link})")


# ============================================================================
# PAGE: FCM INSIGHTS
# ============================================================================

def page_fcm_insights():
    st.title("🧠  FCM + ML Model Insights")
    _health_dot()

    if not st.session_state.fcm_data:
        if st.session_state.fcm_err:
            st.error("FCM failed to load — see Diagnostics.")
        else:
            st.info("Visit the Dashboard to load FCM + ML models first.")
        return

    fcm = st.session_state.fcm_data

    actual       = fcm.get("actual_norm", pd.Series())
    actual_arr   = np.asarray(actual.values if hasattr(actual, "values") else actual, dtype=float)
    actual_idx   = actual.index if hasattr(actual, "index") else pd.RangeIndex(len(actual_arr))

    # ── Build the four prediction series ─────────────────────────────────
    # 1. Yahoo Finance FCM (dense daily series)
    dyn_y    = fcm.get("dyn_norm_yahoo", pd.Series())
    # 2. Google Finance FCM (dense daily series)
    dyn_g    = fcm.get("dyn_norm_google", pd.Series())
    # 3. Yahoo FCM + Yahoo News Sentiment blend
    poly     = fcm.get("polarity_series_yahoo", pd.Series())
    polg     = fcm.get("polarity_series_google", pd.Series())
    pred_h_y = fcm.get("pred_history_yahoo", [])
    date_h_y = fcm.get("date_history_yahoo", [])
    pred_h_g = fcm.get("pred_history_google", [])
    date_h_g = fcm.get("date_history_google", [])

    # Use dense series if available, else fall back to sampled pred_history
    if len(dyn_y) > 10:
        y_dates  = dyn_y.index
        y_preds  = dyn_y.values
    else:
        y_dates  = pd.DatetimeIndex(date_h_y) if date_h_y else pd.RangeIndex(len(pred_h_y))
        y_preds  = np.asarray(pred_h_y, dtype=float)

    if len(dyn_g) > 10:
        g_dates  = dyn_g.index
        g_preds  = dyn_g.values
    else:
        g_dates  = pd.DatetimeIndex(date_h_g) if date_h_g else pd.RangeIndex(len(pred_h_g))
        g_preds  = np.asarray(pred_h_g, dtype=float)

    # Blended: Yahoo FCM + Yahoo News (30% sentiment overlay)
    y_sent_blend = _blend_sentiment(y_preds, poly, y_dates, alpha=0.30)
    # Blended: Google FCM + Google News (30% sentiment overlay)
    g_sent_blend = _blend_sentiment(g_preds, polg, g_dates, alpha=0.30)

    # Align actual to common dates
    def _align_actual(pred_dates, actual_s):
        if not hasattr(pred_dates, 'normalize'):
            return np.full(len(pred_dates), np.nan)
        try:
            return actual_s.reindex(pred_dates).ffill().bfill().values
        except Exception:
            n = min(len(pred_dates), len(actual_s))
            return np.asarray(actual_s.values[-n:], dtype=float)

    act_y = _align_actual(y_dates, actual)
    act_g = _align_actual(g_dates, actual)

    # ── Model Scoreboard ─────────────────────────────────────────────────
    _section("Model Performance Scoreboard")

    lstm_res = fcm.get("lstm_results",  (None, None))
    gru_res  = fcm.get("gru_results",   (None, None))
    ens_pred = fcm.get("ensemble_predictions")

    rows = [
        _model_metrics_row(y_preds,        act_y,                 "Yahoo Finance FCM"),
        _model_metrics_row(g_preds,        act_g,                 "Google Finance FCM"),
        _model_metrics_row(y_sent_blend,   act_y,                 "Yahoo FCM + Yahoo News"),
        _model_metrics_row(g_sent_blend,   act_g,                 "Google FCM + Google News"),
    ]
    if lstm_res and lstm_res[1] is not None and len(lstm_res[1]) > 2:
        rows.append(_model_metrics_row(lstm_res[1], actual_arr[-len(lstm_res[1]):], "LSTM"))
    if gru_res  and gru_res[1]  is not None and len(gru_res[1])  > 2:
        rows.append(_model_metrics_row(gru_res[1],  actual_arr[-len(gru_res[1]):],  "GRU"))
    if ens_pred is not None and _safe_len(ens_pred) > 2:
        rows.append(_model_metrics_row(ens_pred, actual_arr[-len(ens_pred):], "Ensemble"))

    score_df  = pd.DataFrame(rows)
    # rank by Corr desc
    score_df["Rank"] = score_df["Corr"].rank(ascending=False).astype(int)
    score_df = score_df.sort_values("Rank").reset_index(drop=True)
    best_model = score_df.iloc[0]["Model"] if not score_df.empty else "—"

    # colour winner row
    def _hl_winner(row):
        if row["Rank"] == 1:
            return [f"background-color: rgba(0,214,143,0.12); color: #00d68f; font-weight: 600"] * len(row)
        return [""] * len(row)

    st.dataframe(score_df.style.apply(_hl_winner, axis=1), width='stretch', hide_index=True)

    # Winner badge
    st.markdown(
        f'<div style="background:rgba(0,214,143,.1);border:1px solid #00d68f;border-radius:8px;'
        f'padding:12px 20px;margin:8px 0;font-size:1.05em;font-weight:700;color:#00d68f;">'
        f'Best Model: {best_model}</div>',
        unsafe_allow_html=True,
    )

    # ── 4-Model Overlay Chart ─────────────────────────────────────────────
    _section("All 4 Models vs Actual NASDAQ Returns")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "All Models", "Yahoo FCM", "Google FCM",
        "Yahoo + News", "Google + News",
    ])

    def _model_chart(dates, pred, actual_aligned, title, pred_color, show_sentiment=None, sent_series=None, height=420):
        fig = go.Figure()
        # Actual
        fig.add_trace(go.Scatter(
            x=dates, y=actual_aligned, name="Actual",
            line=dict(color=C_GREEN, width=1.5), opacity=0.85,
        ))
        # Prediction
        fig.add_trace(go.Scatter(
            x=dates, y=pred, name="Prediction",
            line=dict(color=pred_color, width=2, dash="dot"),
        ))
        # Optional sentiment overlay on secondary y
        if show_sentiment and sent_series is not None:
            try:
                sent_aligned = sent_series.reindex(pd.DatetimeIndex(dates)).ffill().bfill().fillna(0)
                fig.add_trace(go.Scatter(
                    x=dates, y=sent_aligned.values, name="Sentiment",
                    line=dict(color=C_YELLOW, width=1.2, dash="dash"),
                    yaxis="y2", opacity=0.7,
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="#3d444d", yref="y2")
                fig.update_layout(
                    yaxis2=dict(title="Sentiment Polarity", overlaying="y", side="right",
                                showgrid=False, range=[-1, 1], tickfont=dict(size=10)),
                )
            except Exception:
                pass
        fig.update_layout(**CHART_THEME, height=height, title=title, hovermode="x unified",
                           legend=dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top",
                                       bgcolor="rgba(22,27,34,0.88)", bordercolor="#30363d", borderwidth=1,
                                       font=dict(size=11), tracegroupgap=4))
        return fig

    with tab1:
        fig_all = go.Figure()
        n_act = min(len(y_preds), len(act_y))
        fig_all.add_trace(go.Scatter(x=list(y_dates[-n_act:]), y=act_y[-n_act:], name="Actual",
                                      line=dict(color=C_GREEN, width=2)))
        fig_all.add_trace(go.Scatter(x=list(y_dates[-n_act:]), y=y_preds[-n_act:], name="Yahoo Finance FCM",
                                      line=dict(color=C_BLUE, width=1.8, dash="dot")))
        n_act_g = min(len(g_preds), len(act_g))
        fig_all.add_trace(go.Scatter(x=list(g_dates[-n_act_g:]), y=g_preds[-n_act_g:], name="Google Finance FCM",
                                      line=dict(color=C_ORANGE, width=1.8, dash="dot")))
        fig_all.add_trace(go.Scatter(x=list(y_dates[-n_act:]), y=y_sent_blend[-n_act:], name="Yahoo FCM + Yahoo News",
                                      line=dict(color=C_PURPLE, width=1.8, dash="dash")))
        fig_all.add_trace(go.Scatter(x=list(g_dates[-n_act_g:]), y=g_sent_blend[-n_act_g:], name="Google FCM + Google News",
                                      line=dict(color=C_RED, width=1.8, dash="dash")))
        fig_all.update_layout(**CHART_THEME, height=460, title="All Models vs Actual NASDAQ Returns",
                               hovermode="x unified",
                               legend=dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top",
                                           bgcolor="rgba(22,27,34,0.88)", bordercolor="#30363d", borderwidth=1,
                                           font=dict(size=11), tracegroupgap=4))
        st.plotly_chart(fig_all, width='stretch')

    with tab2:
        n = min(len(y_preds), len(act_y))
        fig2 = _model_chart(list(y_dates[-n:]), y_preds[-n:], act_y[-n:],
                            "Yahoo Finance FCM vs Actual", C_BLUE)
        # residuals overlay
        resid = act_y[-n:] - y_preds[-n:]
        fig2.add_trace(go.Bar(x=list(y_dates[-n:]), y=resid,
                               name="Residual", marker_color=[C_GREEN if r >= 0 else C_RED for r in resid],
                               opacity=0.35, yaxis="y2"))
        fig2.update_layout(yaxis2=dict(title="Residual", overlaying="y", side="right",
                                        showgrid=False, tickfont=dict(size=10)))
        st.plotly_chart(fig2, width='stretch')
        m = _model_metrics_row(y_preds[-n:], act_y[-n:], "Yahoo Finance FCM")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Correlation", f"{m['Corr']:.4f}")
        c2.metric("MSE",         f"{m['MSE']:.6f}")
        c3.metric("MAE",         f"{m['MAE']:.6f}")
        c4.metric("Dir Acc",     f"{m['Dir Acc %']:.1f}%")

    with tab3:
        n = min(len(g_preds), len(act_g))
        fig3 = _model_chart(list(g_dates[-n:]), g_preds[-n:], act_g[-n:],
                            "Google Finance FCM vs Actual", C_ORANGE)
        resid = act_g[-n:] - g_preds[-n:]
        fig3.add_trace(go.Bar(x=list(g_dates[-n:]), y=resid,
                               name="Residual", marker_color=[C_GREEN if r >= 0 else C_RED for r in resid],
                               opacity=0.35, yaxis="y2"))
        fig3.update_layout(yaxis2=dict(title="Residual", overlaying="y", side="right",
                                        showgrid=False, tickfont=dict(size=10)))
        st.plotly_chart(fig3, width='stretch')
        m = _model_metrics_row(g_preds[-n:], act_g[-n:], "Google Finance FCM")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Correlation", f"{m['Corr']:.4f}")
        c2.metric("MSE",         f"{m['MSE']:.6f}")
        c3.metric("MAE",         f"{m['MAE']:.6f}")
        c4.metric("Dir Acc",     f"{m['Dir Acc %']:.1f}%")

    with tab4:
        n = min(len(y_sent_blend), len(act_y))
        fig4 = _model_chart(list(y_dates[-n:]), y_sent_blend[-n:], act_y[-n:],
                            "Yahoo FCM + Yahoo News Sentiment vs Actual", C_PURPLE,
                            show_sentiment=True, sent_series=poly)
        st.plotly_chart(fig4, width='stretch')
        m = _model_metrics_row(y_sent_blend[-n:], act_y[-n:], "Yahoo FCM + Yahoo News")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Correlation", f"{m['Corr']:.4f}")
        c2.metric("MSE",         f"{m['MSE']:.6f}")
        c3.metric("MAE",         f"{m['MAE']:.6f}")
        c4.metric("Dir Acc",     f"{m['Dir Acc %']:.1f}%")
        # Sentiment polarity standalone
        if len(poly) > 0:
            _section("Yahoo News Sentiment Polarity Over Time")
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=poly.index, y=poly.values, name="Yahoo Sentiment",
                                        fill="tozeroy", fillcolor="rgba(188,140,255,0.15)",
                                        line=dict(color=C_PURPLE, width=1.5)))
            fig_s.add_hline(y=0,     line_dash="solid", line_color="#3d444d", line_width=1)
            fig_s.add_hline(y=0.05,  line_dash="dot",   line_color=C_GREEN, annotation_text="Bullish")
            fig_s.add_hline(y=-0.05, line_dash="dot",   line_color=C_RED,   annotation_text="Bearish")
            fig_s.update_layout(**CHART_THEME, height=220, title="Yahoo News Sentiment (Daily)",
                                 yaxis_title="Polarity", hovermode="x unified")
            st.plotly_chart(fig_s, width='stretch')

    with tab5:
        n = min(len(g_sent_blend), len(act_g))
        fig5 = _model_chart(list(g_dates[-n:]), g_sent_blend[-n:], act_g[-n:],
                            "Google FCM + Google News Sentiment vs Actual", C_RED,
                            show_sentiment=True, sent_series=polg)
        st.plotly_chart(fig5, width='stretch')
        m = _model_metrics_row(g_sent_blend[-n:], act_g[-n:], "Google FCM + Google News")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Correlation", f"{m['Corr']:.4f}")
        c2.metric("MSE",         f"{m['MSE']:.6f}")
        c3.metric("MAE",         f"{m['MAE']:.6f}")
        c4.metric("Dir Acc",     f"{m['Dir Acc %']:.1f}%")
        if len(polg) > 0:
            _section("Google News Sentiment Polarity Over Time")
            fig_sg = go.Figure()
            fig_sg.add_trace(go.Scatter(x=polg.index, y=polg.values, name="Google Sentiment",
                                         fill="tozeroy", fillcolor="rgba(255,71,87,0.12)",
                                         line=dict(color=C_RED, width=1.5)))
            fig_sg.add_hline(y=0,     line_dash="solid", line_color="#3d444d", line_width=1)
            fig_sg.add_hline(y=0.05,  line_dash="dot",   line_color=C_GREEN, annotation_text="Bullish")
            fig_sg.add_hline(y=-0.05, line_dash="dot",   line_color=C_RED,   annotation_text="Bearish")
            fig_sg.update_layout(**CHART_THEME, height=220, title="Google News Sentiment (Daily)",
                                  yaxis_title="Polarity", hovermode="x unified")
            st.plotly_chart(fig_sg, width='stretch')

    # ── Residuals Comparison ─────────────────────────────────────────────
    _section("Residual Distribution Comparison")
    col1, col2 = st.columns(2)
    with col1:
        n = min(len(y_preds), len(act_y))
        resid_y = act_y[-n:] - y_preds[-n:]
        resid_ys = act_y[-n:] - y_sent_blend[-n:]
        fig_r = go.Figure()
        fig_r.add_trace(go.Histogram(x=resid_y,  nbinsx=50, name="Yahoo FCM",
                                      marker_color=C_BLUE, opacity=0.7))
        
        fig_r.add_trace(go.Histogram(x=resid_ys, nbinsx=50, name="Yahoo+News",
                                      marker_color=C_PURPLE, opacity=0.7))
        fig_r.update_layout(**CHART_THEME, height=280, barmode="overlay",
                             title="Yahoo Model Residuals", xaxis_title="Residual")
        st.plotly_chart(fig_r, width='stretch')
    with col2:
        n = min(len(g_preds), len(act_g))
        resid_g  = act_g[-n:] - g_preds[-n:]
        resid_gs = act_g[-n:] - g_sent_blend[-n:]
        fig_rg = go.Figure()
        fig_rg.add_trace(go.Histogram(x=resid_g,  nbinsx=50, name="Google FCM",
                                       marker_color=C_ORANGE, opacity=0.7))
        fig_rg.add_trace(go.Histogram(x=resid_gs, nbinsx=50, name="Google+News",
                                       marker_color=C_RED, opacity=0.7))
        fig_rg.update_layout(**CHART_THEME, height=280, barmode="overlay",
                              title="Google Model Residuals", xaxis_title="Residual")
        st.plotly_chart(fig_rg, width='stretch')

    # ── LSTM / GRU / Ensemble cards ──────────────────────────────────────
    _section("Deep Learning Models")
    c1, c2, c3 = st.columns(3)
    lm = fcm.get("lstm_metrics") or {}
    gm = fcm.get("gru_metrics")  or {}
    em = fcm.get("ensemble_metrics") or {}
    with c1:
        st.subheader("LSTM")
        if lm:
            st.metric("Train MAE", f"{lm.get('train_mae',0):.6f}")
            st.metric("Test  MAE", f"{lm.get('test_mae', 0):.6f}")
            st.metric("Test  R²",  f"{lm.get('test_r2',  0):.4f}")
        else: st.info("LSTM not available.")
    with c2:
        st.subheader("GRU")
        if gm:
            st.metric("Train MAE", f"{gm.get('train_mae',0):.6f}")
            st.metric("Test  MAE", f"{gm.get('test_mae', 0):.6f}")
            st.metric("Test  R²",  f"{gm.get('test_r2',  0):.4f}")
        else: st.info("GRU not available.")
    with c3:
        st.subheader("Ensemble")
        if em:
            st.metric("# Models",     em.get("num_models", 0))
            st.metric("Ensemble Avg", f"{em.get('ensemble_avg',0):.4f}")
            st.metric("Ensemble Std", f"{em.get('ensemble_std',0):.4f}")
        else: st.info("Ensemble info unavailable.")

    if lstm_res and lstm_res[1] is not None and len(lstm_res[1]) > 2:
        _section("LSTM + GRU + Ensemble vs Actual")
        fig_dl = go.Figure()
        n_dl = len(actual_arr)
        fig_dl.add_trace(go.Scatter(y=actual_arr, name="Actual",
                                     line=dict(color=C_GREEN, width=1.5)))
        fig_dl.add_trace(go.Scatter(y=lstm_res[1], name="LSTM",
                                     line=dict(color=C_ORANGE, width=1.8, dash="dot")))
        if gru_res and gru_res[1] is not None and len(gru_res[1]) > 2:
            fig_dl.add_trace(go.Scatter(y=gru_res[1], name="GRU",
                                         line=dict(color=C_PURPLE, width=1.8, dash="dot")))
        if ens_pred is not None and _safe_len(ens_pred) > 2:
            fig_dl.add_trace(go.Scatter(y=np.asarray(ens_pred), name="Ensemble",
                                         line=dict(color=C_RED, width=2.5)))
        fig_dl.update_layout(**CHART_THEME, height=380,
                              title="LSTM / GRU / Ensemble vs Actual (Normalised)",
                              hovermode="x unified")
        st.plotly_chart(fig_dl, width='stretch')

    # ── FCM Node Weights Heatmap ──────────────────────────────────────────
    _section("FCM Node Weights — Final Learned Weights")
    wh = fcm.get("weights_history_yahoo", [])
    node_order = fcm.get("node_order", [])
    if wh:
        final_w = wh[-1]
        w_mat = _fcm_weights_matrix(final_w, node_order)
        if not w_mat.empty:
            fig_w = px.imshow(
                w_mat, text_auto=".2f",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="FCM Edge Weights (rows=source, cols=target)",
                labels=dict(x="Target Node", y="Source Node", color="Weight"),
            )
            fig_w.update_layout(**CHART_THEME, height=max(320, len(w_mat) * 28 + 100))
            st.plotly_chart(fig_w, width='stretch')

    # ── FCM Node Visualizations ───────────────────────────────────────────
    wh_g        = fcm.get("weights_history_google", [])
    dh_y        = fcm.get("date_history_yahoo",     [])
    dh_g        = fcm.get("date_history_google",    [])
    inorm_y     = fcm.get("inputs_norm_yahoo",      None)
    inorm_g     = fcm.get("inputs_norm_google",     None)
    final_w_y   = wh[-1]   if wh   else {}
    final_w_g   = wh_g[-1] if wh_g else {}

    # Latest outer-node activation values from normalised inputs
    node_vals_y: dict = {}
    node_vals_g: dict = {}
    if inorm_y is not None and len(inorm_y) > 0:
        last_row = inorm_y.iloc[-1]
        node_vals_y = {c: float(last_row[c]) for c in inorm_y.columns}
    if inorm_g is not None and len(inorm_g) > 0:
        last_row = inorm_g.iloc[-1]
        node_vals_g = {c: float(last_row[c]) for c in inorm_g.columns}
    # Add NASDAQ prediction from pred_history
    ph_y = fcm.get("pred_history_yahoo", [])
    ph_g = fcm.get("pred_history_google", [])
    if ph_y:
        node_vals_y["NASDAQ"] = float(ph_y[-1])
    if ph_g:
        node_vals_g["NASDAQ"] = float(ph_g[-1])

    # ── Layered Diagrams (static snapshot) ───────────────────────────────
    _section("FCM Network Diagrams — Yahoo vs Google (Final Learned Weights)")
    if final_w_y or final_w_g:
        col_y, col_g = st.columns(2)
        with col_y:
            st.markdown("#### Yahoo Finance FCM")
            fig_ly = _build_fcm_layered_diagram(
                final_w_y, node_vals_y, inorm_y,
                title="Yahoo FCM — Layered Structure", height=600,
            )
            st.plotly_chart(fig_ly, width='stretch')
        with col_g:
            st.markdown("#### Google Finance FCM")
            fig_lg = _build_fcm_layered_diagram(
                final_w_g, node_vals_g, inorm_g,
                title="Google FCM — Layered Structure", height=600,
            )
            st.plotly_chart(fig_lg, width='stretch')

    # ── Layered Diagrams (1-year animated) ───────────────────────────────
    _section("FCM Network Diagrams — 1-Year Lookback Animation")
    st.caption(
        "Play button animates edge weights and node activations from ~1 year ago to today. "
        "Green edges = positive influence · Red edges = negative influence · "
        "Node size scales with activation strength."
    )
    if wh and dh_y:
        col_ay, col_ag = st.columns(2)
        with col_ay:
            st.markdown("#### Yahoo Finance FCM — Weight Evolution")
            fig_lay = _build_fcm_layered_animated(
                wh, dh_y, inorm_y, node_order,
                title="Yahoo FCM — Layered 1Y", height=660,
            )
            st.plotly_chart(fig_lay, width='stretch')
        with col_ag:
            st.markdown("#### Google Finance FCM — Weight Evolution")
            fig_lag = _build_fcm_layered_animated(
                wh_g if wh_g else wh, dh_g if dh_g else dh_y,
                inorm_g if inorm_g is not None else inorm_y, node_order,
                title="Google FCM — Layered 1Y", height=660,
            )
            st.plotly_chart(fig_lag, width='stretch')

    # ── Circular Networks (static snapshot) ──────────────────────────────
    _section("FCM Circular Network — Full Node Interactions")
    if final_w_y or final_w_g:
        col_cy, col_cg = st.columns(2)
        with col_cy:
            st.markdown("#### Yahoo Finance FCM")
            fig_cy = _build_fcm_circular_network(
                final_w_y, node_vals_y, inorm_y,
                title="Yahoo FCM — Circular Network", height=580,
            )
            st.plotly_chart(fig_cy, width='stretch')
        with col_cg:
            st.markdown("#### Google Finance FCM")
            fig_cg = _build_fcm_circular_network(
                final_w_g, node_vals_g, inorm_g,
                title="Google FCM — Circular Network", height=580,
            )
            st.plotly_chart(fig_cg, width='stretch')

    # ── Circular Networks (1-year animated) ──────────────────────────────
    _section("FCM Circular Network — 1-Year Lookback Animation")
    st.caption(
        "Watch how FCM edge weights and node activations evolve over the past year. "
        "Use ▶ Play or drag the slider. NASDAQ ★ sits at the centre; "
        "all 18 concept/input nodes surround it on the ring."
    )
    if wh and dh_y:
        col_cay, col_cag = st.columns(2)
        with col_cay:
            st.markdown("#### Yahoo Finance FCM — Network Evolution")
            fig_cay = _build_fcm_circular_animated(
                wh, dh_y, inorm_y, node_order,
                title="Yahoo FCM — Circular 1Y", height=680,
            )
            st.plotly_chart(fig_cay, width='stretch')
        with col_cag:
            st.markdown("#### Google Finance FCM — Network Evolution")
            fig_cag = _build_fcm_circular_animated(
                wh_g if wh_g else wh, dh_g if dh_g else dh_y,
                inorm_g if inorm_g is not None else inorm_y, node_order,
                title="Google FCM — Circular 1Y", height=680,
            )
            st.plotly_chart(fig_cag, width='stretch')

    # ── Edge Weight Heatmaps ──────────────────────────────────────────────
    _section("FCM Edge Weight Heatmaps — Yahoo vs Google")
    if final_w_y or final_w_g:
        col_hy, col_hg = st.columns(2)
        with col_hy:
            w_mat_y = _fcm_weights_matrix(final_w_y, node_order)
            if not w_mat_y.empty:
                fig_wy = px.imshow(
                    w_mat_y, text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Yahoo FCM Edge Weights",
                    labels=dict(x="Target", y="Source", color="Weight"),
                )
                fig_wy.update_layout(**CHART_THEME, height=max(320, len(w_mat_y) * 24 + 80))
                st.plotly_chart(fig_wy, width='stretch')
        with col_hg:
            w_mat_g = _fcm_weights_matrix(final_w_g, node_order)
            if not w_mat_g.empty:
                fig_wg = px.imshow(
                    w_mat_g, text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Google FCM Edge Weights",
                    labels=dict(x="Target", y="Source", color="Weight"),
                )
                fig_wg.update_layout(**CHART_THEME, height=max(320, len(w_mat_g) * 24 + 80))
                st.plotly_chart(fig_wg, width='stretch')

    # ── Feature Importance ─────────────────────────────────────────────
    _section("Feature Importance (|Correlation| with NASDAQ)")
    fi = fcm.get("feature_importance", pd.Series())
    if _safe_len(fi) > 0:
        fi_top = fi.head(14)
        colors = [C_GREEN if v > 0 else C_RED for v in fi_top.values]
        fig_fi = go.Figure(go.Bar(
            x=fi_top.values, y=fi_top.index, orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in fi_top.values],
            textposition="outside",
        ))
        fig_fi.update_layout(**CHART_THEME, height=420,
                              title="|Correlation| with NASDAQ Target",
                              xaxis_title="|Corr|", bargap=0.2)
        st.plotly_chart(fig_fi, width='stretch')

    # ── Yahoo vs Google FCM direct comparison ────────────────────────────
    _section("Yahoo Finance vs Google Finance FCM — Direct Comparison")
    n2 = min(len(y_preds), len(g_preds))
    if n2 > 4:
        fig2c = go.Figure()
        actual_common = act_y[-n2:] if len(act_y) >= n2 else np.full(n2, np.nan)
        fig2c.add_trace(go.Scatter(y=actual_common, name="Actual",
                                    line=dict(color=C_GREEN, width=2)))
        fig2c.add_trace(go.Scatter(y=y_preds[-n2:], name="Yahoo Finance FCM",
                                    line=dict(color=C_BLUE, width=1.8, dash="dot")))
        fig2c.add_trace(go.Scatter(y=g_preds[-n2:], name="Google Finance FCM",
                                    line=dict(color=C_ORANGE, width=1.8, dash="dot")))
        fig2c.add_trace(go.Scatter(y=y_sent_blend[-n2:], name="Yahoo + Yahoo News",
                                    line=dict(color=C_PURPLE, width=1.5, dash="dash")))
        fig2c.add_trace(go.Scatter(y=g_sent_blend[-n2:], name="Google + Google News",
                                    line=dict(color=C_RED, width=1.5, dash="dash")))
        fig2c.update_layout(**CHART_THEME, height=360,
                             title="Yahoo vs Google FCM — With and Without News Sentiment",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02, font=dict(size=10)))
        st.plotly_chart(fig2c, width='stretch')

    # ── Rolling Node Correlation Heatmap ─────────────────────────────────
    _section("Node-Target Rolling Correlation Over Time")
    ch = fcm.get("correlations_history_yahoo", {})
    if ch:
        cdf = pd.DataFrame(ch).T.fillna(0)
        if not cdf.empty:
            fig_ch = px.imshow(cdf.T, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                labels=dict(x="Time Snapshot", y="Node", color="Corr"))
            fig_ch.update_layout(**CHART_THEME, height=440,
                                  title="Rolling Node Correlations Over Time")
            st.plotly_chart(fig_ch, width='stretch')


# ============================================================================
# PAGE: BACKTESTING
# ============================================================================

def page_backtesting():
    st.title("🎯  Backtesting & Portfolio Simulation")
    _health_dot()

    cap      = st.sidebar.number_input("Initial Capital ($)", 10_000, 10_000_000, 100_000, 10_000)
    tc_pct   = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
    start_dt = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_dt   = st.sidebar.date_input("End Date",   datetime.now())
    tickers  = st.sidebar.multiselect("Tickers", EQUITY_TICKERS, default=["AAPL","MSFT","NVDA"])
    sig_mode = st.sidebar.radio("Signal Source", [
        "Technical (SMA+RSI)", "FCM + Sentiment", "Random (benchmark)"
    ])
    bench_sym = st.sidebar.text_input("Benchmark Symbol", "SPY")

    if not st.sidebar.button("Run Backtest"):
        st.info("Configure in the sidebar then press **Run Backtest**.")
        return
    if not tickers: st.warning("Select at least one ticker."); return

    with st.spinner("Running backtest…"):
        try:
            price_dfs: Dict[str, pd.DataFrame] = {}
            bench_df  = pd.DataFrame()
            for t in tickers:
                raw = yf.download(t, start=start_dt, end=end_dt, progress=False, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
                if not raw.empty: price_dfs[t] = raw
            try:
                bench_df = yf.download(bench_sym, start=start_dt, end=end_dt,
                                        progress=False, auto_adjust=True)
                if isinstance(bench_df.columns, pd.MultiIndex):
                    bench_df.columns = bench_df.columns.droplevel(1)
            except Exception: pass

            if not price_dfs: st.error("No price data."); return
            ml = min(len(df) for df in price_dfs.values())
            dates = list(list(price_dfs.values())[0].index[:ml])
            pc    = pd.DataFrame({t: df["Close"].values[:ml] for t, df in price_dfs.items()}, index=dates)

            fcm      = st.session_state.fcm_data or {}
            pred_h   = fcm.get("pred_history_yahoo", [])
            portfolio = Portfolio(cap, tc_pct)

            for i, date in enumerate(dates):
                prices = {t: float(pc[t].iloc[i]) for t in pc.columns}
                if sig_mode == "Technical (SMA+RSI)":
                    if i < 20: signals = {t: 0 for t in prices}
                    else:
                        tech    = compute_tech(pc.iloc[max(0,i-60):i+1])
                        signals = {t: (1 if v>0.3 else (-1 if v<-0.3 else 0)) for t,v in tech.items()}
                elif sig_mode == "FCM + Sentiment":
                    sc = float(pred_h[-1]) if pred_h else 0.0
                    sigs, _ = st.session_state.signal_gen.signals({t: sc for t in prices})
                    signals = sigs
                else:
                    rng     = np.random.default_rng(seed=i)
                    signals = {t: int(rng.choice([-1,0,1])) for t in prices}
                portfolio.step(date, prices, signals)

            metrics = portfolio.metrics()
            eq_arr  = np.array(metrics["equity"])
            eq_dates= metrics["dates"]
            dd_arr  = np.array(metrics["drawdown"])

            # Benchmark B&H
            bench_eq = None
            if not bench_df.empty and "Close" in bench_df.columns:
                bc = bench_df["Close"].reindex(pd.DatetimeIndex(eq_dates)).ffill().bfill().values
                bench_eq = bc / bc[0] * cap if bc[0] != 0 else None

            # ── Results metrics ──────────────────────────────────────────
            _section("Performance Metrics")
            m1,m2,m3,m4 = st.columns(4)
            m5,m6,m7,m8 = st.columns(4)
            m1.metric("Total Return",  f"{metrics['total_return']:.2f}%")
            m2.metric("Sharpe Ratio",  f"{metrics['sharpe']:.2f}")
            m3.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
            m4.metric("Calmar Ratio",  f"{metrics['calmar']:.2f}")
            m5.metric("Max Drawdown",  f"{metrics['max_dd']:.2f}%")
            m6.metric("Win Rate",      f"{metrics['win_rate']:.1f}%")
            m7.metric("Total Trades",  metrics["trades"])
            m8.metric("Final Value",   f"${metrics['final']:,.0f}")

            # ── Equity curve ────────────────────────────────────────────
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=eq_dates, y=eq_arr,
                name="Portfolio", fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
                line=dict(color=C_BLUE, width=2)))
            if bench_eq is not None:
                fig_e.add_trace(go.Scatter(x=eq_dates, y=bench_eq,
                    name=f"{bench_sym} B&H",
                    line=dict(color=C_YELLOW, dash="dot", width=1.5)))
            fig_e.add_hline(y=cap, line_dash="dash", line_color=C_GREY,
                              annotation_text="Initial Capital")
            fig_e.update_layout(**CHART_THEME, height=380, title="Equity Curve vs Benchmark",
                                  yaxis_title="Value ($)", hovermode="x unified")
            st.plotly_chart(fig_e, width='stretch')

            # ── Drawdown ────────────────────────────────────────────────
            fig_dd = go.Figure(go.Scatter(
                x=eq_dates, y=dd_arr, fill="tozeroy",
                fillcolor="rgba(255,71,87,0.1)",
                line=dict(color=C_RED, width=1.5), name="Drawdown %",
            ))
            fig_dd.update_layout(**CHART_THEME, height=220,
                                   title="Drawdown (%)", yaxis_title="%")
            st.plotly_chart(fig_dd, width='stretch')

            # ── Monthly returns heatmap ──────────────────────────────────
            _section("Monthly Returns Heatmap")
            eq_s    = pd.Series(eq_arr, index=pd.DatetimeIndex(eq_dates))
            monthly = eq_s.resample("ME").last().pct_change().dropna() * 100
            if len(monthly) > 0:
                monthly_df = pd.DataFrame({"Year": monthly.index.year,
                                            "Month": monthly.index.strftime("%b"),
                                            "Return": monthly.values})
                pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
                month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
                pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
                fig_mr = px.imshow(pivot, text_auto=".1f",
                                    color_continuous_scale="RdYlGn", zmin=-10, zmax=10,
                                    title="Monthly Returns (%)")
                fig_mr.update_layout(**CHART_THEME, height=max(160, len(pivot)*45+80))
                st.plotly_chart(fig_mr, width='stretch')

            # ── Trade P&L distribution ───────────────────────────────────
            _section("Trade Analysis")
            tdf = portfolio.trade_df()
            if not tdf.empty:
                col1, col2 = st.columns([2,1])
                with col1:
                    st.dataframe(tdf.tail(50), width='stretch', hide_index=True)
                with col2:
                    actions = tdf["Action"].value_counts()
                    fig_pie = go.Figure(go.Pie(
                        labels=actions.index, values=actions.values,
                        marker=dict(colors=[C_GREEN, C_RED]),
                        hole=0.45, textinfo="label+percent",
                    ))
                    fig_pie.update_layout(**CHART_THEME, height=260,
                                           title="Buy vs Sell Split", showlegend=False)
                    st.plotly_chart(fig_pie, width='stretch')
            else:
                st.info("No trades executed — try Technical or FCM signals.")

        except Exception as e:
            _health.error("Backtest", traceback.format_exc())
            st.error(f"Backtest error: {e}")


# ============================================================================
# PAGE: DIAGNOSTICS
# ============================================================================

def page_diagnostics():
    st.title("🩺  System Diagnostics")
    h: HealthTracker = st.session_state["health"]

    status_col = C_GREEN if h.ok and not h.warnings else (C_RED if h.errors else C_YELLOW)
    status_txt = "All Systems Operational" if h.ok and not h.warnings else h.summary()
    st.markdown(
        f'<div style="background:rgba(0,0,0,.3);border-left:4px solid {status_col};'
        f'padding:12px 16px;border-radius:6px;font-weight:600;color:{status_col}">'
        f'{status_txt}</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    c1,c2,c3 = st.columns(3)
    c1.metric("Errors",   len(h.errors),   delta=None)
    c2.metric("Warnings", len(h.warnings), delta=None)
    c3.metric("Info Logs",len(h.info),     delta=None)

    if h.errors:
        _section("Errors")
        for e in h.errors:
            with st.expander(f"[{e['ts']}] {e['source']}: {e['msg'][:80]}"):
                st.code(e["msg"])

    if h.warnings:
        _section("Warnings")
        for w in h.warnings:
            st.warning(f"[{w['ts']}] **{w['source']}**: {w['msg']}")

    # ── Dependency matrix ─────────────────────────────────────────────────
    _section("Dependency Status")
    deps = [("streamlit","streamlit"),("pandas","pandas"),("numpy","numpy"),
            ("yfinance","yfinance"),("plotly","plotly"),("textblob","textblob"),
            ("tensorflow","tensorflow"),("sklearn","sklearn"),("scipy","scipy"),
            ("nltk","nltk"),("protobuf","google.protobuf")]
    rows = []
    for name, mod in deps:
        try:
            m   = __import__(mod)
            ver = getattr(m,"__version__","?")
            rows.append({"Package":name,"Status":"✅ OK","Version":ver})
        except ImportError:
            rows.append({"Package":name,"Status":"❌ Missing","Version":"—"})
        except Exception as ex:
            rows.append({"Package":name,"Status":f"⚠️ {ex}","Version":"—"})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # ── FCM status ────────────────────────────────────────────────────────
    _section("FCM Model Status")
    fd = st.session_state.fcm_data
    if fd:
        info = [
            ("price_df_yahoo rows", len(fd.get("price_df_yahoo",[]))),
            ("pred_history length", len(fd.get("pred_history_yahoo",[]))),
            ("actual_norm length",  len(fd.get("actual_norm",[]))),
            ("feature_importance",  len(fd.get("feature_importance",[]))),
            ("LSTM available",      "Yes" if fd.get("lstm_model") else "No"),
            ("GRU available",       "Yes" if fd.get("gru_model")  else "No"),
            ("Ensemble predictions", len(fd.get("ensemble_predictions")) if fd.get("ensemble_predictions") is not None and hasattr(fd.get("ensemble_predictions"), '__len__') else 0),
            ("FCM corr (Yahoo)",    round(fd.get("dynamic_corr_yahoo",0) or 0, 5)),
        ]
        st.dataframe(pd.DataFrame(info, columns=["Key","Value"]),
                     width='stretch', hide_index=True)
        if st.button("Reload FCM"):
            st.session_state.fcm_data = None; st.session_state.fcm_err = None
            h.clear(); st.rerun()
    elif st.session_state.fcm_err:
        st.error("FCM load failed:")
        st.code(st.session_state.fcm_err)
        if st.button("Retry FCM"):
            st.session_state.fcm_data = None; st.session_state.fcm_err = None
            h.clear(); st.rerun()
    else:
        st.info("FCM not loaded — visit the Dashboard page.")

    # ── Env vars ──────────────────────────────────────────────────────────
    _section("Environment Variables")
    env_k = ["POLYGON_API_KEY","NEWSAPI_KEY","FRED_API_KEY",
             "FINNHUB_API_KEY","OPENAI_API_KEY","ALPHA_VANTAGE_API_KEY",
             "TF_ENABLE_ONEDNN_OPTS","TF_CPP_MIN_LOG_LEVEL"]
    st.dataframe(pd.DataFrame([
        {"Env Var": k, "Set": "✅ Yes" if os.getenv(k) else "— Not set"}
        for k in env_k
    ]), width='stretch', hide_index=True)

    # ── Live data health ──────────────────────────────────────────────────
    _section("Live Data Health Check")
    if st.button("Run Health Check"):
        results = []
        for sym in ["AAPL","GC=F","BTC-USD","^VIX","^IXIC","SPY"]:
            try:
                hist   = yf.Ticker(sym).history(period="5d")
                status = "✅ OK" if not hist.empty else "⚠️ Empty"
                latest = f"${hist['Close'].iloc[-1]:.2f}" if not hist.empty else "—"
            except Exception as ex:
                status = f"❌ {ex}"; latest = "—"
            results.append({"Ticker":sym,"Status":status,"Latest Close":latest})
        st.dataframe(pd.DataFrame(results), width='stretch', hide_index=True)

    if st.button("Clear Health Log"):
        h.clear(); st.success("Cleared."); st.rerun()


# ============================================================================
# PAGE: SETTINGS
# ============================================================================

def page_settings():
    st.title("⚙️  Settings & Configuration")
    _health_dot()

    st.subheader("Data Sources")
    st.info("Primary data: **yfinance** (no API key required). "
            "API keys are loaded from `.env` in the project root.")
    st.markdown("""
| Source | Env Var | Unlocks |
|--------|---------|---------|
| Yahoo Finance | — | Market data, news, always available |
| Polygon.io | `POLYGON_API_KEY` | Higher-quality OHLCV + news |
| NewsAPI | `NEWSAPI_KEY` | Full-text historical news |
| FRED | `FRED_API_KEY` | Unemployment & macro indicators |
| Finnhub | `FINNHUB_API_KEY` | Real-time quotes & fundamentals |
| OpenAI | `OPENAI_API_KEY` | LLM-powered analysis |
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | Technical indicators & fundamentals |
""")

    st.subheader("Suppressed Warnings")
    st.markdown("""
- `TF_ENABLE_ONEDNN_OPTS=0` — suppresses harmless floating-point note from TensorFlow
- `TF_CPP_MIN_LOG_LEVEL=3` — suppresses TF C++ info/warning logs
- Protobuf `UserWarning` — version mismatch warning, no functional impact
- Lexicon download — bails out immediately on timeout or HTML response; uses embedded fallback
""")

    st.subheader("Cache TTL Defaults")
    st.markdown("""
| Data Type | TTL | Override |
|-----------|-----|---------|
| OHLCV / Download | 60 s | Hard-coded (`@st.cache_data`) |
| News articles | 600 s | Hard-coded |
| Sentiment | 600 s | Hard-coded |
| Multi-close batch | 300 s | Hard-coded |
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Platform Info")
        st.write(f"**Python**: {sys.version.split()[0]}")
        st.write(f"**Platform**: {sys.platform}")
        st.write("**Dashboard**: FCM AI Terminal v3.0")
        st.write("**FCM Module**: " + ("Available ✅" if _FCM_OK else f"Error ❌"))
    with col2:
        st.subheader("Clear Cache")
        if st.button("Clear All Data Cache"):
            st.cache_data.clear()
            st.success("All cached data cleared — next page load will re-fetch from APIs.")
        if st.button("Clear Health Log"):
            st.session_state.health.clear()
            st.success("Health log cleared.")


# ============================================================================
# PAGE: AI MARKET ORACLE
# ============================================================================

def _compute_pulse_score(fcm_data: dict, macro_raw: float = 0.0) -> dict:
    """
    Compute a 0-100 Market Pulse Score from all available signals.
    Returns dict with total, label, color, and per-component breakdown.
    """
    comps = {}

    # 1. FCM Signal (0–25)
    ph_y = fcm_data.get("pred_history_yahoo", [])
    ph_g = fcm_data.get("pred_history_google", [])
    fcm_vals = ([float(ph_y[-1])] if ph_y else []) + ([float(ph_g[-1])] if ph_g else [])
    avg_pred  = float(np.mean(fcm_vals)) if fcm_vals else 0.5
    fcm_raw   = (avg_pred - 0.5) * 2           # map [0,1] → [-1,1]
    fcm_score = int(round(np.clip(12.5 + fcm_raw * 12.5, 0, 25)))
    comps["FCM Models"] = {"score": fcm_score, "max": 25, "raw": fcm_raw,
                            "label": f"Avg prediction {avg_pred:.4f}"}

    # 2. News Sentiment (0–25)
    poly_y = fcm_data.get("polarity_series_yahoo",  pd.Series())
    poly_g = fcm_data.get("polarity_series_google", pd.Series())
    sent_vals = []
    if len(poly_y) > 0: sent_vals.append(float(poly_y.iloc[-min(10, len(poly_y)):].mean()))
    if len(poly_g) > 0: sent_vals.append(float(poly_g.iloc[-min(10, len(poly_g)):].mean()))
    sent_raw   = float(np.clip(np.mean(sent_vals) if sent_vals else 0.0, -1, 1))
    sent_score = int(round(np.clip(12.5 + sent_raw * 12.5, 0, 25)))
    comps["News Sentiment"] = {"score": sent_score, "max": 25, "raw": sent_raw,
                                "label": f"10-day avg polarity {sent_raw:+.4f}"}

    # 3. ML Consensus (0–25)
    lstm_r = fcm_data.get("lstm_results",  (None, None))
    gru_r  = fcm_data.get("gru_results",   (None, None))
    ens_p  = fcm_data.get("ensemble_predictions")
    ml_dirs = []
    if lstm_r and lstm_r[1] is not None and len(lstm_r[1]) > 1:
        ml_dirs.append(float(np.sign(lstm_r[1][-1] - lstm_r[1][-2])))
    if gru_r  and gru_r[1]  is not None and len(gru_r[1])  > 1:
        ml_dirs.append(float(np.sign(gru_r[1][-1] - gru_r[1][-2])))
    if ens_p is not None and _safe_len(ens_p) > 1:
        ep = np.asarray(ens_p, dtype=float)
        ml_dirs.append(float(np.sign(ep[-1] - ep[-2])))
    ml_raw   = float(np.mean(ml_dirs)) if ml_dirs else 0.0
    ml_score = int(round(np.clip(12.5 + ml_raw * 12.5, 0, 25)))
    comps["ML Models"] = {"score": ml_score, "max": 25, "raw": ml_raw,
                           "label": f"{len(ml_dirs)} model(s) direction consensus"}

    # 4. Macro & Technical (0–25)
    macro_clipped = float(np.clip(macro_raw, -1, 1))
    macro_score   = int(round(np.clip(12.5 + macro_clipped * 12.5, 0, 25)))
    comps["Macro & Technical"] = {"score": macro_score, "max": 25, "raw": macro_clipped,
                                   "label": "FRED yield curve · Fed Funds · AV RSI"}

    total = int(np.clip(sum(c["score"] for c in comps.values()), 0, 100))
    if total >= 68:   label, col = "STRONGLY BULLISH", C_GREEN
    elif total >= 56: label, col = "BULLISH",          C_GREEN
    elif total >= 44: label, col = "NEUTRAL",          C_BLUE
    elif total >= 32: label, col = "BEARISH",          C_RED
    else:             label, col = "STRONGLY BEARISH", C_RED

    return {"total": total, "label": label, "color": col, "components": comps}


def _scenario_cone(fcm_preds: list, nasdaq_close: pd.Series) -> go.Figure:
    """Fan/cone chart: Bull / Base / Bear NASDAQ scenarios guided by FCM trend."""
    if len(fcm_preds) < 10 or len(nasdaq_close) < 30:
        return go.Figure()
    last_price = float(nasdaq_close.iloc[-1])
    rv_daily   = float(nasdaq_close.pct_change().tail(60).std())
    ph = np.asarray(fcm_preds[-20:], dtype=float)
    trend = float(np.polyfit(np.arange(len(ph)), ph, 1)[0])
    drift = float(np.clip(trend * 8, -0.0015, 0.0015))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(nasdaq_close.index[-60:]), y=nasdaq_close.tail(60).values,
        name="NASDAQ (60d)", line=dict(color=C_GREEN, width=2),
    ))

    last_date = pd.Timestamp(nasdaq_close.index[-1])
    scenarios = [
        ("Bull",  C_GREEN,  drift + rv_daily * 1.28, "dot"),
        ("Base",  C_BLUE,   drift,                   "solid"),
        ("Bear",  C_RED,    drift - rv_daily * 1.28, "dot"),
    ]
    for horizon in (30, 60, 90):
        fd = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
        for label, color, d, dash in scenarios:
            prices = [last_price * (1 + d) ** i for i in range(1, len(fd) + 1)]
            fig.add_trace(go.Scatter(
                x=list(fd), y=prices,
                name=f"{label} ({horizon}d)",
                line=dict(color=color, width=1.5, dash=dash),
                opacity=0.6 if label != "Base" else 1.0,
                legendgroup=label,
                showlegend=(horizon == 30),
            ))

    # Probability cone fill (outermost horizon)
    fd90 = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=90)
    bull90 = [last_price * (1 + drift + rv_daily * 1.28) ** i for i in range(1, 91)]
    bear90 = [last_price * (1 + drift - rv_daily * 1.28) ** i for i in range(1, 91)]
    fig.add_trace(go.Scatter(
        x=list(fd90) + list(fd90)[::-1],
        y=bull90 + bear90[::-1],
        fill="toself", fillcolor="rgba(88,166,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90-Day Probability Cone", showlegend=True,
    ))

    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=last_date, x1=last_date, y0=0, y1=1,
        line=dict(dash="dash", color=C_GREY, width=1.5),
    )
    fig.add_annotation(
        x=last_date, y=1, xref="x", yref="paper",
        text="Today", showarrow=False, yanchor="bottom",
        font=dict(color=C_GREY, size=10),
    )
    fig.update_layout(**CHART_THEME, height=440,
                       title="NASDAQ Scenario Probability Cone — FCM-Guided Bull / Base / Bear",
                       yaxis_title="NASDAQ Index Level", hovermode="x unified",
                       legend=dict(orientation="v", x=1.01, xanchor="left", y=1.0,
                                   bgcolor="rgba(22,27,34,0.88)", bordercolor="#30363d",
                                   borderwidth=1, font=dict(size=10)))
    return fig


def _signal_row_html(name: str, value_str: str, direction: str, strength: int, color: str) -> str:
    """Build one HTML <tr> for the Signal Convergence Matrix."""
    arrows = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→", "CAUTION": "⚠"}
    arrow  = arrows.get(direction, "→")
    dots   = "●" * max(0, min(5, strength)) + "○" * (5 - max(0, min(5, strength)))
    return (
        f'<tr style="border-top:1px solid #21262d">'
        f'<td style="padding:7px 12px;color:#c9d1d9;font-size:.88em">{name}</td>'
        f'<td style="padding:7px 12px;color:#8b949e;font-family:monospace;font-size:.85em">{value_str}</td>'
        f'<td style="padding:7px 12px"><span style="color:{color};font-weight:700;font-size:.85em">'
        f'{arrow} {direction}</span></td>'
        f'<td style="padding:7px 12px;color:{color};letter-spacing:3px;font-size:.9em">{dots}</td>'
        f'</tr>'
    )


def page_ai_oracle():
    st.title("🔮  AI Market Oracle")
    _health_dot()
    st.caption(
        "Every available signal — FCM · LSTM/GRU/Ensemble · Yahoo & Google News Sentiment · "
        "FRED Macro (Yield Curve, Fed Funds, CPI, Unemployment) · Alpha Vantage Technicals · "
        "Finnhub Analyst Consensus · Polygon Live News · OpenAI GPT-4o-mini Synthesis — "
        "unified into one intelligence view."
    )

    fcm_data = st.session_state.fcm_data or {}

    # ── Fetch all external data (all cached) ─────────────────────────────
    with st.spinner("Fetching macro, technical, and institutional data…"):
        fred_fedfunds = fetch_fred_series("FEDFUNDS", limit=60)
        fred_unrate   = fetch_fred_series("UNRATE",   limit=60)
        fred_t10y2y   = fetch_fred_series("T10Y2Y",   limit=252)
        fred_cpi      = fetch_fred_series("CPIAUCSL", limit=40)
        nasdaq_df     = fetch_ohlcv("^IXIC", "1y")
        rsi_av        = fetch_alpha_vantage_indicator("QQQ", "RSI",  time_period="14", series_type="close")
        macd_av       = fetch_alpha_vantage_indicator("QQQ", "MACD", fastperiod="12",
                                                      slowperiod="26", signalperiod="9")
        recs          = fetch_finnhub_recommendation("QQQ")
        poly_news     = fetch_polygon_news_feed(limit=15)

    nasdaq_close = nasdaq_df["Close"] if not nasdaq_df.empty and "Close" in nasdaq_df.columns else None

    # ── Compute macro/technical composite signal ─────────────────────────
    macro_sigs = []
    if len(fred_t10y2y) >= 1:
        macro_sigs.append(float(np.clip(fred_t10y2y.iloc[-1] / 2.0, -1, 1)))
    if len(fred_unrate) >= 3:
        macro_sigs.append(float(np.clip(-(fred_unrate.iloc[-1] - fred_unrate.iloc[-3]) * 5, -1, 1)))
    if len(fred_fedfunds) >= 1:
        macro_sigs.append(float(np.clip((4.0 - fred_fedfunds.iloc[-1]) / 4.0, -1, 1)))
    if len(rsi_av) >= 1:
        r = float(rsi_av.iloc[-1])
        macro_sigs.append(float(np.clip(0.5 if r < 30 else (-0.5 if r > 70 else (r - 50) / 50), -1, 1)))
    macro_raw = float(np.mean(macro_sigs)) if macro_sigs else 0.0

    pulse = _compute_pulse_score(fcm_data, macro_raw)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 1 — MARKET PULSE SCORE
    # ═════════════════════════════════════════════════════════════════════
    _section("Market Pulse Score — Unified Signal Intelligence")
    col_gauge, col_bars = st.columns([1, 2])

    with col_gauge:
        pc = pulse["color"]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pulse["total"],
            number={"font": {"size": 52, "color": pc}, "suffix": ""},
            gauge={
                "axis":   {"range": [0, 100], "tickwidth": 1, "tickcolor": C_GREY,
                            "tickvals": [0, 25, 50, 75, 100]},
                "bar":    {"color": pc, "thickness": 0.30},
                "bgcolor": "#1c2230",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  32], "color": "rgba(255,71,87,.18)"},
                    {"range": [32, 44], "color": "rgba(255,71,87,.07)"},
                    {"range": [44, 56], "color": "rgba(139,148,158,.10)"},
                    {"range": [56, 68], "color": "rgba(0,214,143,.07)"},
                    {"range": [68,100], "color": "rgba(0,214,143,.18)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                               "value": pulse["total"]},
            },
            title={"text": f"<b>{pulse['label']}</b>",
                   "font": {"size": 13, "color": pc}},
        ))
        _ct = {k: v for k, v in CHART_THEME.items() if k != "margin"}
        fig_g.update_layout(**_ct, height=300,
                              margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_g, width='stretch')

    with col_bars:
        st.markdown("<br>", unsafe_allow_html=True)
        for cname, cd in pulse["components"].items():
            pct   = cd["score"] / cd["max"] * 100
            bc    = C_GREEN if pct >= 60 else (C_RED if pct <= 40 else C_YELLOW)
            st.markdown(
                f'<div style="margin-bottom:12px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                f'<span style="font-size:.9em;color:#c9d1d9;font-weight:600">{cname}</span>'
                f'<span style="font-size:.82em;color:{bc}">'
                f'{cd["score"]}/{cd["max"]} &nbsp;·&nbsp; {cd["label"]}</span></div>'
                f'<div style="background:#21262d;border-radius:4px;height:9px">'
                f'<div style="background:{bc};width:{pct:.0f}%;height:9px;border-radius:4px"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 2 — FRED MACRO INTELLIGENCE
    # ═════════════════════════════════════════════════════════════════════
    _section("Macro Intelligence — FRED Economic Indicators")

    m1, m2, m3, m4 = st.columns(4)
    def _fm(col, label, series, suffix="", invert=False):
        with col:
            if len(series) >= 2:
                cur = float(series.iloc[-1]); prv = float(series.iloc[-2])
                d   = cur - prv
                dc  = (C_RED if d > 0 else C_GREEN) if invert else (C_GREEN if d > 0 else C_RED)
                col.metric(label, f"{cur:.2f}{suffix}",
                            delta=f"{'▲' if d>0 else '▼'} {abs(d):.2f}{suffix}",
                            delta_color="off")
                col.markdown(f'<span style="color:{dc};font-size:.78em">{"▲" if d>0 else "▼"} '
                              f'{abs(d):.2f}{suffix} vs prev</span>', unsafe_allow_html=True)
            elif len(series) == 1:
                col.metric(label, f"{float(series.iloc[-1]):.2f}{suffix}")
            else:
                col.metric(label, "—  (no FRED key)")

    _fm(m1, "Fed Funds Rate", fred_fedfunds, suffix="%", invert=True)
    _fm(m2, "Unemployment",   fred_unrate,   suffix="%", invert=True)
    _fm(m3, "10Y−2Y Spread",  fred_t10y2y,   suffix="%")
    with m4:
        if len(fred_cpi) >= 13:
            cpi_yoy  = (float(fred_cpi.iloc[-1]) / float(fred_cpi.iloc[-13]) - 1) * 100
            cpi_prev = (float(fred_cpi.iloc[-2]) / float(fred_cpi.iloc[-14]) - 1) * 100 \
                       if len(fred_cpi) >= 14 else cpi_yoy
            d = cpi_yoy - cpi_prev
            st.metric("CPI YoY Inflation", f"{cpi_yoy:.2f}%",
                       delta=f"{'▲' if d>0 else '▼'} {abs(d):.2f}%", delta_color="off")
            dc = C_RED if d > 0 else C_GREEN
            st.markdown(f'<span style="color:{dc};font-size:.78em">{"▲" if d>0 else "▼"} '
                         f'{abs(d):.2f}% vs prev month</span>', unsafe_allow_html=True)
        else:
            st.metric("CPI YoY Inflation", "—  (no FRED key)")

    col_yc, col_ff = st.columns(2)
    with col_yc:
        if len(fred_t10y2y) > 10:
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(
                x=fred_t10y2y.index, y=fred_t10y2y.values,
                fill="tozeroy", fillcolor="rgba(88,166,255,0.10)",
                line=dict(color=C_BLUE, width=1.8), name="10Y−2Y Spread",
            ))
            fig_yc.add_hline(y=0, line_dash="dash", line_color=C_RED,
                               annotation_text="Inversion line")
            fig_yc.update_layout(**CHART_THEME, height=240,
                                   title="Yield Curve — 10Y minus 2Y Treasury Spread",
                                   yaxis_title="Spread (%)", hovermode="x unified")
            st.plotly_chart(fig_yc, width='stretch')
    with col_ff:
        if len(fred_fedfunds) > 10:
            fig_ff = go.Figure()
            fig_ff.add_trace(go.Scatter(
                x=fred_fedfunds.index, y=fred_fedfunds.values,
                fill="tozeroy", fillcolor="rgba(240,136,62,0.10)",
                line=dict(color=C_ORANGE, width=1.8), name="Fed Funds Rate",
            ))
            fig_ff.update_layout(**CHART_THEME, height=240,
                                   title="Federal Funds Rate (FRED)",
                                   yaxis_title="Rate (%)", hovermode="x unified")
            st.plotly_chart(fig_ff, width='stretch')

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 3 — SIGNAL CONVERGENCE MATRIX
    # ═════════════════════════════════════════════════════════════════════
    _section("Signal Convergence Matrix — All Sources at a Glance")

    def _dir(v, is_norm=True):
        """Return (direction_label, color, strength 1-5) for a signal value."""
        if is_norm:
            if v >= 0.55: return "BULLISH", C_GREEN,  min(5, int((v - 0.5) * 20) + 2)
            if v <= 0.45: return "BEARISH", C_RED,    min(5, int((0.5 - v) * 20) + 2)
            return "NEUTRAL", C_BLUE, 2
        else:
            if v >  0.05: return "BULLISH", C_GREEN,  min(5, int(abs(v) * 5) + 1)
            if v < -0.05: return "BEARISH", C_RED,    min(5, int(abs(v) * 5) + 1)
            return "NEUTRAL", C_BLUE, 2

    rows = []
    ph_y = fcm_data.get("pred_history_yahoo", [])
    ph_g = fcm_data.get("pred_history_google", [])
    ens_p = fcm_data.get("ensemble_predictions")
    poly_y = fcm_data.get("polarity_series_yahoo",  pd.Series())
    poly_g = fcm_data.get("polarity_series_google", pd.Series())

    if ph_y:
        v = float(ph_y[-1]); d, c, s = _dir(v)
        rows.append(_signal_row_html("Yahoo Finance FCM", f"{v:.4f}", d, s, c))
    if ph_g:
        v = float(ph_g[-1]); d, c, s = _dir(v)
        rows.append(_signal_row_html("Google Finance FCM", f"{v:.4f}", d, s, c))
    if lstm_r := fcm_data.get("lstm_results"):
        if lstm_r[1] is not None and len(lstm_r[1]) > 0:
            v = float(lstm_r[1][-1]); d, c, s = _dir(v)
            rows.append(_signal_row_html("LSTM Deep Learning", f"{v:.4f}", d, s, c))
    if gru_r := fcm_data.get("gru_results"):
        if gru_r[1] is not None and len(gru_r[1]) > 0:
            v = float(gru_r[1][-1]); d, c, s = _dir(v)
            rows.append(_signal_row_html("GRU Deep Learning", f"{v:.4f}", d, s, c))
    if ens_p is not None and _safe_len(ens_p) > 0:
        v = float(np.asarray(ens_p, dtype=float)[-1]); d, c, s = _dir(v)
        rows.append(_signal_row_html("Ensemble (FCM + LSTM + GRU)", f"{v:.4f}", d, s, c))
    if len(poly_y) > 0:
        v = float(poly_y.iloc[-min(10, len(poly_y)):].mean()); d, c, s = _dir(v, False)
        rows.append(_signal_row_html("Yahoo News Sentiment (10d)", f"{v:+.4f}", d, s, c))
    if len(poly_g) > 0:
        v = float(poly_g.iloc[-min(10, len(poly_g)):].mean()); d, c, s = _dir(v, False)
        rows.append(_signal_row_html("Google News Sentiment (10d)", f"{v:+.4f}", d, s, c))
    if len(rsi_av) > 0:
        rv = float(rsi_av.iloc[-1])
        rd = "BEARISH" if rv > 70 else ("BULLISH" if rv < 30 else "NEUTRAL")
        rc = C_RED if rv > 70 else (C_GREEN if rv < 30 else C_BLUE)
        rs = 4 if rv > 70 or rv < 30 else 2
        rows.append(_signal_row_html("RSI-14 QQQ (Alpha Vantage)", f"{rv:.1f}", rd, rs, rc))
    if len(fred_t10y2y) >= 1:
        sp = float(fred_t10y2y.iloc[-1])
        yd = "BULLISH" if sp > 0.5 else ("BEARISH" if sp < 0 else "CAUTION")
        yc = C_GREEN if sp > 0.5 else (C_RED if sp < 0 else C_YELLOW)
        rows.append(_signal_row_html("Yield Curve 10Y−2Y (FRED)", f"{sp:+.2f}%", yd, min(5, int(abs(sp)*2+1)), yc))
    if len(fred_fedfunds) >= 1:
        ff = float(fred_fedfunds.iloc[-1])
        fd = "CAUTION" if ff > 4.5 else ("BULLISH" if ff < 2.0 else "NEUTRAL")
        fc = C_YELLOW if ff > 4.5 else (C_GREEN if ff < 2.0 else C_BLUE)
        rows.append(_signal_row_html("Federal Funds Rate (FRED)", f"{ff:.2f}%", fd, 4 if ff > 4.5 else 2, fc))
    if len(fred_unrate) >= 3:
        ur = float(fred_unrate.iloc[-1])
        ut = float(fred_unrate.iloc[-1] - fred_unrate.iloc[-3])
        ud = "BEARISH" if ut > 0.2 else ("BULLISH" if ut < -0.1 else "NEUTRAL")
        uc = C_RED if ut > 0.2 else (C_GREEN if ut < -0.1 else C_BLUE)
        rows.append(_signal_row_html("Unemployment Rate (FRED)", f"{ur:.1f}% (trend {ut:+.1f}%)", ud, 3, uc))

    bullish_n = sum(1 for r in rows if "BULLISH" in r)
    bearish_n = sum(1 for r in rows if "BEARISH" in r)
    neutral_n = len(rows) - bullish_n - bearish_n

    st.markdown(
        f'<div style="display:flex;gap:16px;margin-bottom:12px">'
        f'<div style="background:rgba(0,214,143,.12);border:1px solid {C_GREEN};border-radius:6px;'
        f'padding:8px 18px;font-weight:700;color:{C_GREEN}">▲ {bullish_n} Bullish</div>'
        f'<div style="background:rgba(255,71,87,.12);border:1px solid {C_RED};border-radius:6px;'
        f'padding:8px 18px;font-weight:700;color:{C_RED}">▼ {bearish_n} Bearish</div>'
        f'<div style="background:rgba(88,166,255,.12);border:1px solid {C_BLUE};border-radius:6px;'
        f'padding:8px 18px;font-weight:700;color:{C_BLUE}">→ {neutral_n} Neutral</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<table style="width:100%;border-collapse:collapse;background:rgba(22,27,34,0.55);'
        'border-radius:10px;overflow:hidden">'
        '<thead><tr style="background:rgba(33,38,45,0.95)">'
        '<th style="padding:9px 12px;text-align:left;color:#8b949e;font-size:.78em;'
        'font-weight:700;letter-spacing:.06em">SIGNAL SOURCE</th>'
        '<th style="padding:9px 12px;text-align:left;color:#8b949e;font-size:.78em;'
        'font-weight:700;letter-spacing:.06em">VALUE</th>'
        '<th style="padding:9px 12px;text-align:left;color:#8b949e;font-size:.78em;'
        'font-weight:700;letter-spacing:.06em">DIRECTION</th>'
        '<th style="padding:9px 12px;text-align:left;color:#8b949e;font-size:.78em;'
        'font-weight:700;letter-spacing:.06em">STRENGTH</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>',
        unsafe_allow_html=True,
    )

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 4 — SCENARIO PROBABILITY CONE
    # ═════════════════════════════════════════════════════════════════════
    _section("NASDAQ Scenario Probability Cone — FCM-Guided Projections")
    st.caption(
        "Bull / Base / Bear trajectories derived from the FCM prediction slope and "
        "60-day NASDAQ realized volatility. Shaded region = 1.28σ probability envelope (~90%)."
    )
    if nasdaq_close is not None and len(ph_y) > 10:
        st.plotly_chart(_scenario_cone(ph_y, nasdaq_close), width='stretch')
    else:
        st.info("Load FCM models on the Dashboard to enable scenario projections.")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 5 — ALPHA VANTAGE TECHNICAL DEEP DIVE
    # ═════════════════════════════════════════════════════════════════════
    _section("Advanced Technical Analysis — QQQ (Alpha Vantage)")
    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        if len(rsi_av) > 20:
            rp = rsi_av.tail(120); rv_last = float(rp.iloc[-1])
            rc = C_RED if rv_last > 70 else (C_GREEN if rv_last < 30 else C_BLUE)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=rp.index, y=rp.values, fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
                line=dict(color=C_BLUE, width=1.8), name="RSI-14",
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color=C_RED,   annotation_text="Overbought 70")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color=C_GREEN, annotation_text="Oversold 30")
            fig_rsi.add_hline(y=50, line_dash="dot",  line_color=C_GREY,  line_width=1)
            fig_rsi.add_annotation(x=rp.index[-1], y=rv_last, text=f"RSI {rv_last:.1f}",
                                    font=dict(color=rc, size=11), showarrow=False, xshift=45)
            fig_rsi.update_layout(**CHART_THEME, height=290,
                                   title="RSI-14 — QQQ (Alpha Vantage)",
                                   yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_rsi, width='stretch')
        else:
            st.info("RSI unavailable — check ALPHA_VANTAGE_API_KEY or API rate limit.")

    with col_macd:
        if len(macd_av) > 20:
            mp = macd_av.tail(120)
            bc = [C_GREEN if v >= 0 else C_RED for v in mp.values]
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=mp.index, y=mp.values,
                                       marker_color=bc, name="MACD Histogram", opacity=0.8))
            fig_macd.add_hline(y=0, line_color=C_GREY, line_width=1)
            fig_macd.update_layout(**CHART_THEME, height=290,
                                    title="MACD Histogram — QQQ (Alpha Vantage)",
                                    hovermode="x unified")
            st.plotly_chart(fig_macd, width='stretch')
        else:
            st.info("MACD unavailable — check ALPHA_VANTAGE_API_KEY or API rate limit.")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 6 — FINNHUB ANALYST CONSENSUS
    # ═════════════════════════════════════════════════════════════════════
    _section("Institutional Analyst Consensus — QQQ (Finnhub)")
    if recs:
        rec_df  = pd.DataFrame(recs)
        latest  = rec_df.iloc[0]
        keys    = ["strongBuy", "buy", "hold", "sell", "strongSell"]
        labels  = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
        colors  = [C_GREEN, "#5dd48a", C_YELLOW, C_ORANGE, C_RED]
        values  = [int(latest.get(k, 0)) for k in keys]
        total_r = sum(values)
        if total_r > 0:
            col_pie, col_rec = st.columns([1, 2])
            with col_pie:
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker=dict(colors=colors), hole=0.50,
                    textinfo="label+percent", textfont=dict(size=10),
                ))
                fig_pie.update_layout(**CHART_THEME, height=290,
                                       title=f"Analyst Consensus — {latest.get('period','Latest')}")
                st.plotly_chart(fig_pie, width='stretch')
            with col_rec:
                st.markdown("<br>", unsafe_allow_html=True)
                for lbl, key, col in zip(labels, keys, colors):
                    cnt = int(latest.get(key, 0))
                    pct = cnt / total_r * 100
                    st.markdown(
                        f'<div style="display:flex;align-items:center;margin-bottom:8px">'
                        f'<span style="width:120px;font-size:.86em;color:#c9d1d9">{lbl}</span>'
                        f'<div style="flex:1;background:#21262d;border-radius:3px;height:7px;margin:0 12px">'
                        f'<div style="background:{col};width:{pct:.0f}%;height:7px;border-radius:3px"></div></div>'
                        f'<span style="font-size:.84em;color:{col};font-weight:700;width:36px">{cnt}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        if len(rec_df) > 1 and "strongBuy" in rec_df.columns:
            fig_rt = go.Figure()
            for key, color, label in zip(keys, colors, labels):
                if key in rec_df.columns:
                    fig_rt.add_trace(go.Bar(x=rec_df["period"], y=rec_df[key],
                                             name=label, marker_color=color))
            fig_rt.update_layout(**CHART_THEME, height=280, barmode="stack",
                                   title="Recommendation Trend — QQQ (Finnhub)",
                                   xaxis_title="Period", yaxis_title="# Analysts",
                                   legend=dict(orientation="h", y=1.02, font=dict(size=10)))
            st.plotly_chart(fig_rt, width='stretch')
    else:
        st.info("Analyst data unavailable — check FINNHUB_API_KEY in Settings.")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 7 — POLYGON LIVE NEWS INTELLIGENCE
    # ═════════════════════════════════════════════════════════════════════
    _section("Live Market Intelligence — Polygon News Feed")
    if poly_news:
        for article in poly_news[:10]:
            title     = article.get("title", "Untitled")
            publisher = (article.get("publisher") or {}).get("name", "Unknown")
            link      = article.get("article_url", "#")
            pub_dt    = article.get("published_utc", "")
            tickers_m = ", ".join(article.get("tickers", [])[:5])
            try:
                dt_str = pd.Timestamp(pub_dt).strftime("%b %d, %H:%M UTC") if pub_dt else "—"
            except Exception:
                dt_str = pub_dt[:16] if pub_dt else "—"
            art_pol = 0.0
            try:
                from textblob import TextBlob as _TB
                art_pol = _TB(title).sentiment.polarity
            except Exception:
                pass
            pol_c = C_GREEN if art_pol > 0.05 else (C_RED if art_pol < -0.05 else C_GREY)
            with st.expander(f"📡  {title[:100]}{'…' if len(title)>100 else ''}", expanded=False):
                st.markdown(
                    f"**Source:** {publisher} &nbsp;|&nbsp; **Published:** {dt_str}"
                    f" &nbsp;|&nbsp; **Tickers:** `{tickers_m or '—'}`"
                    f" &nbsp;|&nbsp; **Headline Sentiment:** "
                    f"<span style='color:{pol_c}'>{art_pol:+.2f}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"[Read Full Article ↗]({link})")
    else:
        st.info("Polygon news unavailable — check POLYGON_API_KEY in Settings.")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 8 — AI STRATEGIC NARRATIVE (OpenAI GPT-4o-mini)
    # ═════════════════════════════════════════════════════════════════════
    _section("AI Strategic Narrative — GPT-4o-mini Market Synthesis")
    st.caption(
        "All signals above are packaged into a structured prompt and sent to GPT-4o-mini "
        "to produce a hedge-fund-grade investment thesis. Click Generate to run."
    )

    def _build_context() -> str:
        lines = [
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            f"Market Pulse Score: {pulse['total']}/100 — {pulse['label']}",
        ]
        if ph_y: lines.append(f"Yahoo FCM (norm pred): {float(ph_y[-1]):.4f}")
        if ph_g: lines.append(f"Google FCM (norm pred): {float(ph_g[-1]):.4f}")
        lr = fcm_data.get("lstm_results")
        gr = fcm_data.get("gru_results")
        if lr and lr[1] is not None and len(lr[1]) > 0:
            lines.append(f"LSTM last output: {float(lr[1][-1]):.4f}")
        if gr and gr[1] is not None and len(gr[1]) > 0:
            lines.append(f"GRU last output: {float(gr[1][-1]):.4f}")
        if ens_p is not None and _safe_len(ens_p) > 0:
            lines.append(f"Ensemble prediction: {float(np.asarray(ens_p)[-1]):.4f}")
        if len(poly_y) > 0:
            lines.append(f"Yahoo sentiment (10d avg): {float(poly_y.iloc[-min(10,len(poly_y)):].mean()):+.4f}")
        if len(poly_g) > 0:
            lines.append(f"Google sentiment (10d avg): {float(poly_g.iloc[-min(10,len(poly_g)):].mean()):+.4f}")
        if len(fred_t10y2y) >= 1:  lines.append(f"10Y-2Y yield spread: {float(fred_t10y2y.iloc[-1]):+.2f}%")
        if len(fred_fedfunds) >= 1: lines.append(f"Federal funds rate: {float(fred_fedfunds.iloc[-1]):.2f}%")
        if len(fred_unrate) >= 1:   lines.append(f"US unemployment: {float(fred_unrate.iloc[-1]):.1f}%")
        if len(fred_cpi) >= 13:
            yoy = (float(fred_cpi.iloc[-1]) / float(fred_cpi.iloc[-13]) - 1) * 100
            lines.append(f"CPI YoY inflation: {yoy:.2f}%")
        if len(rsi_av) >= 1: lines.append(f"QQQ RSI-14: {float(rsi_av.iloc[-1]):.1f}")
        if nasdaq_close is not None and len(nasdaq_close) >= 2:
            p = float(nasdaq_close.iloc[-1]); p0 = float(nasdaq_close.iloc[-2])
            lines.append(f"NASDAQ last close: {p:,.0f} ({(p-p0)/p0*100:+.2f}% 1-day)")
        if recs:
            r0 = recs[0]
            sb = int(r0.get("strongBuy", 0)) + int(r0.get("buy", 0))
            h  = int(r0.get("hold", 0))
            ss = int(r0.get("sell", 0)) + int(r0.get("strongSell", 0))
            lines.append(f"Finnhub QQQ consensus: {sb} Buy / {h} Hold / {ss} Sell")
        if poly_news:
            lines.append(f"Latest headline: {poly_news[0].get('title', '')}")
        return "\n".join(lines)

    if "oracle_narrative" not in st.session_state:
        st.session_state.oracle_narrative = None
    if "oracle_ts" not in st.session_state:
        st.session_state.oracle_ts = None

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        gen_clicked = st.button("🤖  Generate AI Analysis", type="primary")
    with col_info:
        if st.session_state.oracle_ts:
            st.caption(f"Last generated: {st.session_state.oracle_ts}")
        else:
            st.caption("No analysis generated yet this session.")

    if gen_clicked:
        oai_key = os.getenv("OPENAI_API_KEY", "")
        if not oai_key:
            st.error("OPENAI_API_KEY not set — add it to your .env file.")
        else:
            with st.spinner("Synthesising all signals with GPT-4o-mini…"):
                try:
                    from openai import OpenAI as _OAI
                    client = _OAI(api_key=oai_key)
                    ctx = _build_context()
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": (
                                "You are a senior quantitative strategist at a top-tier hedge fund. "
                                "Synthesise ML model outputs, macro indicators, and news sentiment "
                                "into precise, data-driven investment theses. Use exact numbers. "
                                "Be concise and structured."
                            )},
                            {"role": "user", "content": (
                                f"Complete market signal picture for NASDAQ as of today:\n\n{ctx}\n\n"
                                "Write a structured strategic intelligence report with these sections:\n"
                                "## Executive Summary\n"
                                "## Key Market Drivers (bullet points with data)\n"
                                "## Signal Convergence Analysis (where models agree or conflict)\n"
                                "## Primary Risk Factors\n"
                                "## Strategic Trade Thesis\n"
                                "## 30-Day Probability Scenarios\n"
                                "- Bull Case (X%): target, catalyst\n"
                                "- Base Case (X%): target, assumption\n"
                                "- Bear Case (X%): target, risk\n"
                                "## Conviction Score: X/10 with one-line rationale"
                            )},
                        ],
                        max_tokens=1500,
                        temperature=0.3,
                    )
                    st.session_state.oracle_narrative = resp.choices[0].message.content
                    st.session_state.oracle_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    st.error(f"OpenAI error: {e}")

    if st.session_state.oracle_narrative:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1f2e 0%,#161b22 100%);'
            f'border:1px solid #30363d;border-left:4px solid {C_PURPLE};'
            f'border-radius:10px;padding:22px 26px;margin-top:14px">',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:.78em;color:#8b949e;margin-bottom:14px;font-family:monospace">'
            f'GPT-4o-mini  ·  Generated {st.session_state.oracle_ts}  ·  '
            f'Market Pulse {pulse["total"]}/100 — {pulse["label"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state.oracle_narrative)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇  Download Report (.md)",
            data=(
                f"# FCM AI Market Oracle — Strategic Intelligence Report\n"
                f"*Generated: {st.session_state.oracle_ts}*\n\n"
                f"**Market Pulse Score: {pulse['total']}/100 — {pulse['label']}**\n\n"
                f"---\n\n{st.session_state.oracle_narrative}"
            ),
            file_name=f"oracle_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )
    else:
        st.markdown(
            f'<div style="background:rgba(22,27,34,0.6);border:1px dashed #30363d;'
            f'border-radius:10px;padding:28px;text-align:center;color:#8b949e;margin-top:8px">'
            f'Click <b style="color:#c9d1d9">Generate AI Analysis</b> above to synthesise every '
            f'available market signal into a GPT-4o-mini investment thesis with probability scenarios.'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    init()
    with st.sidebar:
        st.title("📈 FCM AI Terminal")
        page = st.radio("Navigate", [
            "Dashboard", "Market Data", "Commodities",
            "News & Sentiment", "FCM Insights",
            "AI Market Oracle",
            "Backtesting", "Diagnostics", "Settings",
        ], label_visibility="collapsed")
        st.markdown("---")
        st.caption("FCM AI Terminal v3.0")
        st.caption("FCM · LSTM · GRU · Ensemble")

    dispatch = {
        "Dashboard":        page_dashboard,
        "Market Data":      page_market_data,
        "Commodities":      page_commodities,
        "News & Sentiment": page_news_sentiment,
        "FCM Insights":     page_fcm_insights,
        "Backtesting":      page_backtesting,
        "Diagnostics":      page_diagnostics,
        "Settings":         page_settings,
        "AI Market Oracle": page_ai_oracle,
    }
    dispatch[page]()

if __name__ == "__main__":
    main()
