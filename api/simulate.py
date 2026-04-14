"""
Vercel Python serverless function.
Fetches Yahoo Finance data, runs 4 FCM simulations, returns JSON.
Cached for 1 hour so it stays fast for repeated visitors.
"""

from http.server import BaseHTTPRequestHandler
import json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

SIM_DAYS   = 30
END_DATE   = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=100)

TICKERS = {
    "nasdaq": "^IXIC",
    "vix":    "^VIX",
    "tnx":    "^TNX",
    "nvda":   "NVDA",
    "aapl":   "AAPL",
    "tsla":   "TSLA",
    "msft":   "MSFT",
    "meta":   "META",
    "wmt":    "WMT",
    "jpm":    "JPM",
    "googl":  "GOOGL",
    "v":      "V",
    "amzn":   "AMZN",
    "gspc":   "^GSPC",
}

# ─── FCM Definitions ──────────────────────────────────────────────────────────

FCMS = [
    {
        "id": "fcm1", "name": "AI-Backed + Research",
        "subtitle": "Macro FCM — corrected edges per annotation",
        "nodes": [
            {"id":"inflation",    "label":"Inflation Rates",    "proxy":"tnx",    "invert":False, "color":"#FF6B6B", "x":-2.5,"y": 1.5,"z": 0},
            {"id":"unemployment", "label":"Unemployment",       "proxy":None,     "invert":False, "color":"#FFA07A", "x":-2.5,"y":-1.5,"z": 0},
            {"id":"sentiment",    "label":"Investor Sentiment", "proxy":"vix",    "invert":True,  "color":"#4ECDC4", "x": 0,  "y": 2.5,"z": 1},
            {"id":"earnings",     "label":"Corporate Earnings", "proxy":"gspc",   "invert":False, "color":"#45B7D1", "x": 0,  "y":-2.5,"z": 1},
            {"id":"aitech",       "label":"AI Technology",      "proxy":"nvda",   "invert":False, "color":"#96CEB4", "x": 2.5,"y": 0,  "z": 0},
            {"id":"nasdaq",       "label":"Nasdaq",             "proxy":"nasdaq", "invert":False, "color":"#FFD700", "x": 0,  "y": 0,  "z": 2.5, "target":True},
        ],
        "edges": [
            {"from":"inflation",    "to":"nasdaq",    "weight":-0.70},
            {"from":"inflation",    "to":"sentiment", "weight":-0.50},
            {"from":"inflation",    "to":"earnings",  "weight":-0.40},
            {"from":"unemployment", "to":"nasdaq",    "weight":-0.60},
            {"from":"unemployment", "to":"earnings",  "weight":-0.40},
            {"from":"sentiment",    "to":"nasdaq",    "weight": 0.80},
            {"from":"earnings",     "to":"nasdaq",    "weight": 0.70},
            {"from":"aitech",       "to":"nasdaq",    "weight": 0.60},
            {"from":"aitech",       "to":"sentiment", "weight": 0.50},
        ],
    },
    {
        "id": "fcm2", "name": "Research-Based",
        "subtitle": "Macro FCM derived from academic literature",
        "nodes": [
            {"id":"inflation",    "label":"Inflation Rates",    "proxy":"tnx",    "invert":False, "color":"#FF6B6B", "x":-2.5,"y": 1.5,"z": 0},
            {"id":"unemployment", "label":"Unemployment",       "proxy":None,     "invert":False, "color":"#FFA07A", "x":-2.5,"y":-1.5,"z": 0},
            {"id":"sentiment",    "label":"Investor Sentiment", "proxy":"vix",    "invert":True,  "color":"#4ECDC4", "x": 0,  "y": 2.5,"z": 1},
            {"id":"earnings",     "label":"Corporate Earnings", "proxy":"gspc",   "invert":False, "color":"#45B7D1", "x": 0,  "y":-2.5,"z": 1},
            {"id":"aitech",       "label":"AI Technology",      "proxy":"nvda",   "invert":False, "color":"#96CEB4", "x": 2.5,"y": 0,  "z": 0},
            {"id":"nasdaq",       "label":"Nasdaq",             "proxy":"nasdaq", "invert":False, "color":"#FFD700", "x": 0,  "y": 0,  "z": 2.5, "target":True},
        ],
        "edges": [
            {"from":"inflation",    "to":"nasdaq",    "weight":-0.80},
            {"from":"inflation",    "to":"sentiment", "weight":-0.40},
            {"from":"unemployment", "to":"nasdaq",    "weight":-0.50},
            {"from":"sentiment",    "to":"nasdaq",    "weight": 0.70},
            {"from":"earnings",     "to":"nasdaq",    "weight": 0.80},
            {"from":"aitech",       "to":"nasdaq",    "weight": 0.50},
            {"from":"aitech",       "to":"sentiment", "weight": 0.40},
            {"from":"earnings",     "to":"sentiment", "weight": 0.30},
        ],
    },
    {
        "id": "fcm3", "name": "3-Layer FCM",
        "subtitle": "Hierarchical FCM: inputs → hidden conditions → Nasdaq",
        "nodes": [
            {"id":"inflation",  "label":"Inflation",      "proxy":"tnx",    "invert":False, "color":"#FF6B6B", "x":-3,  "y": 2,  "z":-1, "layer":1},
            {"id":"rates",      "label":"Interest Rates", "proxy":"tnx",    "invert":False, "color":"#FF8C00", "x":-1,  "y": 2.5,"z":-1, "layer":1},
            {"id":"global",     "label":"Global Markets", "proxy":"gspc",   "invert":False, "color":"#FFD700", "x": 1,  "y": 2.5,"z":-1, "layer":1},
            {"id":"employment", "label":"Employment",     "proxy":None,     "invert":False, "color":"#90EE90", "x": 3,  "y": 2,  "z":-1, "layer":1},
            {"id":"mktcond",    "label":"Mkt Conditions", "proxy":None,     "invert":False, "color":"#00CED1", "x":-1.5,"y": 0,  "z": 1, "layer":2, "hidden":True},
            {"id":"riskapp",    "label":"Risk Appetite",  "proxy":None,     "invert":False, "color":"#1E90FF", "x": 0,  "y":-1,  "z": 1, "layer":2, "hidden":True},
            {"id":"techhealth", "label":"Tech Health",    "proxy":"nvda",   "invert":False, "color":"#9370DB", "x": 1.5,"y": 0,  "z": 1, "layer":2, "hidden":True},
            {"id":"nasdaq",     "label":"Nasdaq",         "proxy":"nasdaq", "invert":False, "color":"#FFD700", "x": 0,  "y": 0,  "z": 3, "target":True, "layer":3},
        ],
        "edges": [
            {"from":"inflation",  "to":"mktcond",    "weight":-0.60},
            {"from":"inflation",  "to":"riskapp",    "weight":-0.40},
            {"from":"rates",      "to":"mktcond",    "weight":-0.70},
            {"from":"rates",      "to":"techhealth", "weight":-0.50},
            {"from":"global",     "to":"mktcond",    "weight": 0.60},
            {"from":"global",     "to":"riskapp",    "weight": 0.50},
            {"from":"employment", "to":"mktcond",    "weight": 0.40},
            {"from":"employment", "to":"riskapp",    "weight": 0.30},
            {"from":"mktcond",    "to":"nasdaq",     "weight": 0.70},
            {"from":"riskapp",    "to":"nasdaq",     "weight": 0.60},
            {"from":"techhealth", "to":"nasdaq",     "weight": 0.80},
        ],
    },
    {
        "id": "fcm4", "name": "Stock-Based FCM",
        "subtitle": "Daily FCM using Nasdaq-100 stock correlations",
        "nodes": [
            {"id":"aapl",   "label":"AAPL",   "proxy":"aapl",   "invert":False, "color":"#A8D8EA", "x":-1, "y": 3, "z": 0},
            {"id":"tsla",   "label":"TSLA",   "proxy":"tsla",   "invert":False, "color":"#FF9999", "x": 1, "y": 3, "z": 0},
            {"id":"nvda",   "label":"NVDA",   "proxy":"nvda",   "invert":False, "color":"#76FF76", "x": 3, "y": 1, "z": 0},
            {"id":"msft",   "label":"MSFT",   "proxy":"msft",   "invert":False, "color":"#87CEEB", "x": 3, "y":-1, "z": 0},
            {"id":"meta",   "label":"META",   "proxy":"meta",   "invert":False, "color":"#DDA0DD", "x": 1, "y":-3, "z": 0},
            {"id":"wmt",    "label":"WMT",    "proxy":"wmt",    "invert":False, "color":"#F0E68C", "x":-1, "y":-3, "z": 0},
            {"id":"jpm",    "label":"JPM",    "proxy":"jpm",    "invert":False, "color":"#FFA500", "x":-3, "y":-1, "z": 0},
            {"id":"googl",  "label":"GOOGL",  "proxy":"googl",  "invert":False, "color":"#98FB98", "x":-3, "y": 1, "z": 0},
            {"id":"v",      "label":"V",      "proxy":"v",      "invert":False, "color":"#87CEFA", "x":-2, "y": 0, "z": 2},
            {"id":"amzn",   "label":"AMZN",   "proxy":"amzn",   "invert":False, "color":"#FFD700", "x": 2, "y": 0, "z": 2},
            {"id":"nasdaq", "label":"Nasdaq", "proxy":"nasdaq", "invert":False, "color":"#FF6347", "x": 0, "y": 0, "z": 0, "target":True},
        ],
        "edges": [
            {"from":"aapl",  "to":"nasdaq", "weight": 0.85},
            {"from":"msft",  "to":"nasdaq", "weight": 0.82},
            {"from":"nvda",  "to":"nasdaq", "weight": 0.80},
            {"from":"googl", "to":"nasdaq", "weight": 0.78},
            {"from":"amzn",  "to":"nasdaq", "weight": 0.75},
            {"from":"meta",  "to":"nasdaq", "weight": 0.72},
            {"from":"tsla",  "to":"nasdaq", "weight": 0.65},
            {"from":"v",     "to":"nasdaq", "weight": 0.55},
            {"from":"jpm",   "to":"nasdaq", "weight": 0.50},
            {"from":"wmt",   "to":"nasdaq", "weight": 0.35},
            {"from":"nvda",  "to":"aapl",   "weight": 0.65},
            {"from":"nvda",  "to":"msft",   "weight": 0.70},
            {"from":"nvda",  "to":"tsla",   "weight": 0.50},
            {"from":"aapl",  "to":"msft",   "weight": 0.75},
            {"from":"googl", "to":"meta",   "weight": 0.70},
            {"from":"googl", "to":"aapl",   "weight": 0.65},
            {"from":"amzn",  "to":"googl",  "weight": 0.60},
            {"from":"jpm",   "to":"v",      "weight": 0.60},
            {"from":"wmt",   "to":"amzn",   "weight": 0.45},
            {"from":"tsla",  "to":"nvda",   "weight":-0.15},
        ],
    },
]

# ─── Data & Simulation ────────────────────────────────────────────────────────

def fetch_data():
    raw = yf.download(
        list(TICKERS.values()),
        start=START_DATE.strftime("%Y-%m-%d"),
        end=END_DATE.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        threads=True,
    )["Close"]
    # Map Yahoo ticker columns back to our keys
    inv = {v: k for k, v in TICKERS.items()}
    raw.columns = [inv.get(str(c), str(c)) for c in raw.columns]
    return raw.ffill().dropna()


def norm_return(series):
    return np.tanh(series.pct_change().fillna(0) * 20)


def simulate(fcm_def, df):
    nodes   = {n["id"]: n for n in fcm_def["nodes"]}
    target  = next(nid for nid, nd in nodes.items() if nd.get("target"))

    norm = {}
    for nid, nd in nodes.items():
        proxy = nd.get("proxy")
        if proxy and proxy in df.columns:
            s = norm_return(df[proxy])
            norm[nid] = -s if nd.get("invert") else s
        else:
            norm[nid] = pd.Series(0.0, index=df.index)

    incoming = {nid: [] for nid in nodes}
    for e in fcm_def["edges"]:
        incoming[e["to"]].append((e["from"], e["weight"]))

    sim_start = len(df) - SIM_DAYS
    results = []

    for i in range(sim_start, len(df) - 1):
        # No leakage: use day-i observed returns to predict day-(i+1) direction
        vals = {nid: float(norm[nid].iloc[i]) for nid in nodes}
        vals[target] = 0.0

        for _ in range(3):
            new = {}
            for nid in nodes:
                inc = incoming[nid]
                if not inc:
                    new[nid] = vals[nid]
                else:
                    ws = sum(w * vals.get(f, 0.0) for f, w in inc)
                    new[nid] = float(np.tanh(ws))
            vals = new

        pred_dir   = "UP" if vals[target] > 0 else "DOWN"
        nasdaq_i   = float(df["nasdaq"].iloc[i])
        nasdaq_i1  = float(df["nasdaq"].iloc[i + 1])
        actual_dir = "UP" if nasdaq_i1 > nasdaq_i else "DOWN"
        actual_pct = (nasdaq_i1 - nasdaq_i) / nasdaq_i * 100.0

        results.append({
            "date":         df.index[i + 1].strftime("%Y-%m-%d"),
            "nodeValues":   {k: round(v, 4) for k, v in vals.items()},
            "predictedVal": round(vals[target], 4),
            "predictedDir": pred_dir,
            "actualDir":    actual_dir,
            "actualPct":    round(actual_pct, 3),
            "correct":      pred_dir == actual_dir,
            "nasdaqClose":  round(nasdaq_i1, 2),
        })

    accuracy = sum(r["correct"] for r in results) / len(results) * 100 if results else 0.0
    return {"results": results, "accuracy": round(accuracy, 1)}


def run_all():
    df = fetch_data()
    sim_results = [simulate(f, df) for f in FCMS]
    dates = df.index[-SIM_DAYS:].strftime("%Y-%m-%d").tolist()
    return {
        "fcms":    FCMS,
        "simdata": sim_results,
        "meta": {
            "startDate": dates[0],
            "endDate":   dates[-1],
            "simDays":   SIM_DAYS,
            "fetchedAt": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        },
    }

# ─── Vercel Handler ───────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            data = run_all()
            body = json.dumps(data, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            # Cache 1 hour on CDN, 10 min stale-while-revalidate
            self.send_header("Cache-Control", "public, s-maxage=3600, stale-while-revalidate=600")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(err)

    def log_message(self, *args):
        pass  # suppress default access log noise
