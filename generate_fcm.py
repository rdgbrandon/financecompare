#!/usr/bin/env python3
"""
FCM Nasdaq Predictor
Fetches Yahoo Finance data, simulates 4 FCMs over the last 30 trading days,
then generates a self-contained 3D interactive HTML visualization.
No data leakage: each day's prediction uses only prior-day observed data.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import json
import warnings
import sys
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

END_DATE   = datetime.now()
START_DATE = END_DATE - timedelta(days=100)   # extra history for normalization
SIM_DAYS   = 30

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

# ─────────────────────────────────────────────────────────────────────────────
# FCM DEFINITIONS  (edges extracted from the 4 images)
# ─────────────────────────────────────────────────────────────────────────────

FCMS = [
    # ── FCM 1: AI-Backed + Research (Corrected) ───────────────────────────
    {
        "id":       "fcm1",
        "name":     "AI-Backed + Research",
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
            {"from":"inflation",    "to":"nasdaq",      "weight":-0.70},
            {"from":"inflation",    "to":"sentiment",   "weight":-0.50},
            {"from":"inflation",    "to":"earnings",    "weight":-0.40},
            {"from":"unemployment", "to":"nasdaq",      "weight":-0.60},
            {"from":"unemployment", "to":"earnings",    "weight":-0.40},
            {"from":"sentiment",    "to":"nasdaq",      "weight": 0.80},
            {"from":"earnings",     "to":"nasdaq",      "weight": 0.70},
            {"from":"aitech",       "to":"nasdaq",      "weight": 0.60},
            {"from":"aitech",       "to":"sentiment",   "weight": 0.50},
        ],
    },

    # ── FCM 2: Research-Based ─────────────────────────────────────────────
    {
        "id":       "fcm2",
        "name":     "Research-Based",
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
            {"from":"inflation",    "to":"nasdaq",      "weight":-0.80},
            {"from":"inflation",    "to":"sentiment",   "weight":-0.40},
            {"from":"unemployment", "to":"nasdaq",      "weight":-0.50},
            {"from":"sentiment",    "to":"nasdaq",      "weight": 0.70},
            {"from":"earnings",     "to":"nasdaq",      "weight": 0.80},
            {"from":"aitech",       "to":"nasdaq",      "weight": 0.50},
            {"from":"aitech",       "to":"sentiment",   "weight": 0.40},
            {"from":"earnings",     "to":"sentiment",   "weight": 0.30},
        ],
    },

    # ── FCM 3: 3-Layer ────────────────────────────────────────────────────
    {
        "id":       "fcm3",
        "name":     "3-Layer FCM",
        "subtitle": "Hierarchical FCM: inputs → hidden conditions → Nasdaq",
        "nodes": [
            # Layer 1 – observable inputs
            {"id":"inflation",   "label":"Inflation",       "proxy":"tnx",    "invert":False, "color":"#FF6B6B", "x":-3,  "y": 2,"z":-1, "layer":1},
            {"id":"rates",       "label":"Interest Rates",  "proxy":"tnx",    "invert":False, "color":"#FF8C00", "x":-1,  "y": 2.5,"z":-1,"layer":1},
            {"id":"global",      "label":"Global Markets",  "proxy":"gspc",   "invert":False, "color":"#FFD700", "x": 1,  "y": 2.5,"z":-1,"layer":1},
            {"id":"employment",  "label":"Employment",      "proxy":None,     "invert":False, "color":"#90EE90", "x": 3,  "y": 2,"z":-1, "layer":1},
            # Layer 2 – hidden
            {"id":"mktcond",     "label":"Mkt Conditions",  "proxy":None,     "invert":False, "color":"#00CED1", "x":-1.5,"y": 0,"z": 1, "layer":2, "hidden":True},
            {"id":"riskapp",     "label":"Risk Appetite",   "proxy":None,     "invert":False, "color":"#1E90FF", "x": 0,  "y":-1,"z": 1, "layer":2, "hidden":True},
            {"id":"techhealth",  "label":"Tech Health",     "proxy":"nvda",   "invert":False, "color":"#9370DB", "x": 1.5,"y": 0,"z": 1, "layer":2, "hidden":True},
            # Layer 3 – output
            {"id":"nasdaq",      "label":"Nasdaq",          "proxy":"nasdaq", "invert":False, "color":"#FFD700", "x": 0,  "y": 0,"z": 3, "target":True, "layer":3},
        ],
        "edges": [
            # L1 → L2
            {"from":"inflation",  "to":"mktcond",   "weight":-0.60},
            {"from":"inflation",  "to":"riskapp",   "weight":-0.40},
            {"from":"rates",      "to":"mktcond",   "weight":-0.70},
            {"from":"rates",      "to":"techhealth","weight":-0.50},
            {"from":"global",     "to":"mktcond",   "weight": 0.60},
            {"from":"global",     "to":"riskapp",   "weight": 0.50},
            {"from":"employment", "to":"mktcond",   "weight": 0.40},
            {"from":"employment", "to":"riskapp",   "weight": 0.30},
            # L2 → L3
            {"from":"mktcond",    "to":"nasdaq",    "weight": 0.70},
            {"from":"riskapp",    "to":"nasdaq",    "weight": 0.60},
            {"from":"techhealth", "to":"nasdaq",    "weight": 0.80},
        ],
    },

    # ── FCM 4: Stock-Based Correlations ───────────────────────────────────
    {
        "id":       "fcm4",
        "name":     "Stock-Based FCM",
        "subtitle": "Daily FCM using Nasdaq-100 stock correlations",
        "nodes": [
            {"id":"aapl",   "label":"AAPL",   "proxy":"aapl",   "invert":False, "color":"#A8D8EA", "x":-1,  "y": 3,  "z": 0},
            {"id":"tsla",   "label":"TSLA",   "proxy":"tsla",   "invert":False, "color":"#FF9999", "x": 1,  "y": 3,  "z": 0},
            {"id":"nvda",   "label":"NVDA",   "proxy":"nvda",   "invert":False, "color":"#76FF76", "x": 3,  "y": 1,  "z": 0},
            {"id":"msft",   "label":"MSFT",   "proxy":"msft",   "invert":False, "color":"#87CEEB", "x": 3,  "y":-1,  "z": 0},
            {"id":"meta",   "label":"META",   "proxy":"meta",   "invert":False, "color":"#DDA0DD", "x": 1,  "y":-3,  "z": 0},
            {"id":"wmt",    "label":"WMT",    "proxy":"wmt",    "invert":False, "color":"#F0E68C", "x":-1,  "y":-3,  "z": 0},
            {"id":"jpm",    "label":"JPM",    "proxy":"jpm",    "invert":False, "color":"#FFA500", "x":-3,  "y":-1,  "z": 0},
            {"id":"googl",  "label":"GOOGL",  "proxy":"googl",  "invert":False, "color":"#98FB98", "x":-3,  "y": 1,  "z": 0},
            {"id":"v",      "label":"V",      "proxy":"v",      "invert":False, "color":"#87CEFA", "x":-2,  "y": 0,  "z": 2},
            {"id":"amzn",   "label":"AMZN",   "proxy":"amzn",   "invert":False, "color":"#FFD700", "x": 2,  "y": 0,  "z": 2},
            {"id":"nasdaq", "label":"Nasdaq", "proxy":"nasdaq", "invert":False, "color":"#FF6347", "x": 0,  "y": 0,  "z": 0, "target":True},
        ],
        "edges": [
            # Stocks → Nasdaq (weights ≈ index weightings × correlation)
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
            # Inter-stock correlations (from image color scale)
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

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data():
    print("\nFetching market data from Yahoo Finance …")
    frames = {}
    for key, ticker in TICKERS.items():
        try:
            df = yf.download(
                ticker,
                start=START_DATE.strftime("%Y-%m-%d"),
                end=END_DATE.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if not df.empty:
                frames[key] = df["Close"].squeeze()
                print(f"  ✓  {ticker:6s}  {len(frames[key])} days")
            else:
                print(f"  ✗  {ticker}  (no data)")
        except Exception as exc:
            print(f"  ✗  {ticker}  ({exc})")

    combined = pd.DataFrame(frames).ffill().dropna()
    print(f"\nClean dataset: {len(combined)} trading days\n")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# FCM SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def norm_return(series: pd.Series) -> pd.Series:
    """Daily pct-change mapped to [-1, 1] via tanh.  Scale=20 → ±5 % → ±0.93."""
    return np.tanh(series.pct_change().fillna(0) * 20)


def simulate(fcm_def: dict, df: pd.DataFrame) -> dict:
    nodes   = {n["id"]: n for n in fcm_def["nodes"]}
    target  = next(nid for nid, nd in nodes.items() if nd.get("target"))

    # Pre-compute normalised series for each node
    norm = {}
    for nid, nd in nodes.items():
        proxy = nd.get("proxy")
        if proxy and proxy in df.columns:
            s = norm_return(df[proxy])
            norm[nid] = -s if nd.get("invert") else s
        else:
            norm[nid] = pd.Series(0.0, index=df.index)

    # incoming edges per node
    incoming = {nid: [] for nid in nodes}
    for e in fcm_def["edges"]:
        incoming[e["to"]].append((e["from"], e["weight"]))

    sim_start = len(df) - SIM_DAYS
    results = []

    for i in range(sim_start, len(df) - 1):
        # ── NO LEAKAGE ──────────────────────────────────────────────────────
        # At close of day i we observe all input-node values (returns day i).
        # We predict whether Nasdaq closes HIGHER on day i+1 than day i.
        # ────────────────────────────────────────────────────────────────────

        # Initialize node values from day-i observed returns
        vals = {nid: float(norm[nid].iloc[i]) for nid in nodes}
        vals[target] = 0.0           # unknown – we are predicting this

        # FCM propagation (3 iterations → convergence for acyclic/weakly cyclic)
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

        pred_val  = vals[target]
        pred_dir  = "UP" if pred_val > 0 else "DOWN"

        nasdaq_i   = float(df["nasdaq"].iloc[i])
        nasdaq_i1  = float(df["nasdaq"].iloc[i + 1])
        actual_dir = "UP" if nasdaq_i1 > nasdaq_i else "DOWN"
        actual_pct = (nasdaq_i1 - nasdaq_i) / nasdaq_i * 100.0

        results.append({
            "date":         df.index[i + 1].strftime("%Y-%m-%d"),
            "nodeValues":   {k: round(v, 4) for k, v in vals.items()},
            "predictedVal": round(pred_val, 4),
            "predictedDir": pred_dir,
            "actualDir":    actual_dir,
            "actualPct":    round(actual_pct, 3),
            "correct":      pred_dir == actual_dir,
            "nasdaqClose":  round(nasdaq_i1, 2),
        })

    accuracy = sum(r["correct"] for r in results) / len(results) * 100 if results else 0.0
    return {"results": results, "accuracy": round(accuracy, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FCM Nasdaq Simulator</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#020b18;color:#c8e6ff;font-family:'Segoe UI',monospace;overflow:hidden;height:100vh}

/* ── Neon / glow utilities ── */
.glow{text-shadow:0 0 8px #00d4ff,0 0 20px #00d4ff}
.panel{background:rgba(0,20,50,.75);border:1px solid rgba(0,180,255,.25);border-radius:10px;backdrop-filter:blur(6px)}

/* ── Layout ── */
#app{display:grid;grid-template-rows:54px 1fr 110px;height:100vh}

/* header */
#header{display:flex;align-items:center;padding:0 20px;gap:16px;
        border-bottom:1px solid rgba(0,180,255,.3);background:rgba(0,10,30,.9)}
#header h1{font-size:1.25rem;letter-spacing:3px;color:#00d4ff;text-transform:uppercase}
#header .badge{font-size:.7rem;padding:3px 10px;border:1px solid #00d4ff44;border-radius:20px;color:#7dd3fc}
#tabs{display:flex;gap:6px;margin-left:auto}
.tab{padding:6px 14px;border:1px solid rgba(0,180,255,.3);border-radius:6px;cursor:pointer;
     font-size:.75rem;letter-spacing:1px;transition:all .2s;color:#7dd3fc}
.tab:hover,.tab.active{background:rgba(0,180,255,.2);border-color:#00d4ff;color:#fff;
                        box-shadow:0 0 12px #00d4ff55}

/* main */
#main{display:grid;grid-template-columns:1fr 340px;gap:0;overflow:hidden}
#canvas-wrap{position:relative;overflow:hidden}
canvas{display:block}

/* right panel */
#side{display:flex;flex-direction:column;gap:10px;padding:12px;overflow-y:auto;
      border-left:1px solid rgba(0,180,255,.2)}

/* accuracy ring */
#acc-ring-wrap{text-align:center;padding:8px 0}
#acc-title{font-size:.65rem;letter-spacing:2px;color:#7dd3fc;margin-bottom:4px}
#acc-ring{position:relative;display:inline-block}
#acc-ring canvas{display:block}
#acc-num{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
         font-size:1.6rem;font-weight:700;color:#00d4ff}
#acc-label{position:absolute;top:62%;left:50%;transform:translateX(-50%);
           font-size:.6rem;color:#7dd3fc;white-space:nowrap}

/* day card */
#day-card{padding:10px 12px}
.dc-date{font-size:.65rem;color:#7dd3fc;letter-spacing:2px;margin-bottom:6px}
.dc-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.dc-label{font-size:.7rem;color:#7dd3fc}
.dc-val{font-size:.85rem;font-weight:600}
.up{color:#4ade80}.down{color:#f87171}.neutral{color:#7dd3fc}
.correct-badge{text-align:center;padding:5px;border-radius:6px;font-size:.75rem;font-weight:700;letter-spacing:1px}
.correct-badge.ok{background:rgba(74,222,128,.15);border:1px solid #4ade80;color:#4ade80}
.correct-badge.bad{background:rgba(248,113,113,.15);border:1px solid #f87171;color:#f87171}

/* nodes list */
#nodes-list{padding:8px 12px;max-height:160px;overflow-y:auto}
#nodes-list h3{font-size:.6rem;letter-spacing:2px;color:#7dd3fc;margin-bottom:6px}
.node-row{display:flex;justify-content:space-between;align-items:center;
          padding:3px 0;border-bottom:1px solid rgba(255,255,255,.05)}
.node-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;flex-shrink:0}
.node-name{font-size:.68rem;flex:1}
.node-bar-wrap{width:60px;background:rgba(255,255,255,.08);border-radius:3px;height:5px;margin:0 6px}
.node-bar{height:5px;border-radius:3px;transition:width .4s}
.node-val{font-size:.65rem;width:38px;text-align:right}

/* timeline chart */
#chart-wrap{padding:8px 12px}
#chart-wrap h3{font-size:.6rem;letter-spacing:2px;color:#7dd3fc;margin-bottom:4px}

/* ── Footer controls ── */
#footer{display:flex;align-items:center;gap:12px;padding:0 16px;
        border-top:1px solid rgba(0,180,255,.2);background:rgba(0,10,30,.9)}
#day-info{font-size:.75rem;color:#7dd3fc;min-width:180px}
#day-info b{color:#fff}
#slider-wrap{flex:1;display:flex;align-items:center;gap:8px}
#day-slider{flex:1;-webkit-appearance:none;height:4px;border-radius:2px;
            background:rgba(0,180,255,.3);outline:none;cursor:pointer}
#day-slider::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:#00d4ff;box-shadow:0 0 8px #00d4ff}
.ctrl-btn{padding:5px 14px;border:1px solid rgba(0,180,255,.4);border-radius:6px;
          background:transparent;color:#7dd3fc;cursor:pointer;font-size:.75rem;letter-spacing:1px;
          transition:all .2s}
.ctrl-btn:hover{background:rgba(0,180,255,.15);color:#fff}
.ctrl-btn.play{border-color:#4ade80;color:#4ade80}
.ctrl-btn.play:hover{background:rgba(74,222,128,.15)}
#streak{font-size:.7rem;color:#7dd3fc}

/* scrollbar */
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#00d4ff44;border-radius:2px}

/* tooltip */
#tooltip{position:fixed;background:rgba(0,20,50,.95);border:1px solid #00d4ff55;
         border-radius:8px;padding:8px 12px;font-size:.72rem;pointer-events:none;
         opacity:0;transition:opacity .2s;z-index:999;color:#c8e6ff;max-width:200px}
</style>
</head>
<body>
<div id="app">

<!-- HEADER -->
<div id="header">
  <h1 class="glow">FCM NASDAQ SIMULATOR</h1>
  <span class="badge" id="date-range-badge"></span>
  <div id="tabs">
    <div class="tab active" data-fcm="0">AI-Backed</div>
    <div class="tab" data-fcm="1">Research</div>
    <div class="tab" data-fcm="2">3-Layer</div>
    <div class="tab" data-fcm="3">Stock-Based</div>
  </div>
</div>

<!-- MAIN -->
<div id="main">
  <div id="canvas-wrap">
    <canvas id="three-canvas"></canvas>
  </div>

  <div id="side">
    <!-- accuracy ring -->
    <div class="panel" id="acc-ring-wrap">
      <div id="acc-title">TREND ACCURACY</div>
      <div id="acc-ring">
        <canvas id="ring-canvas" width="130" height="130"></canvas>
        <div id="acc-num">--%</div>
        <div id="acc-label">correct predictions</div>
      </div>
    </div>

    <!-- current day -->
    <div class="panel" id="day-card">
      <div class="dc-date" id="dc-date">–</div>
      <div class="dc-row">
        <span class="dc-label">FCM PREDICTION</span>
        <span class="dc-val" id="dc-pred">–</span>
      </div>
      <div class="dc-row">
        <span class="dc-label">ACTUAL NASDAQ</span>
        <span class="dc-val" id="dc-actual">–</span>
      </div>
      <div class="dc-row">
        <span class="dc-label">CLOSE</span>
        <span class="dc-val" id="dc-close">–</span>
      </div>
      <div class="correct-badge" id="dc-badge">–</div>
    </div>

    <!-- nodes -->
    <div class="panel" id="nodes-list">
      <h3>NODE ACTIVATIONS</h3>
      <div id="node-rows"></div>
    </div>

    <!-- mini accuracy chart -->
    <div class="panel" id="chart-wrap">
      <h3>DAILY RESULTS (LAST 30 DAYS)</h3>
      <canvas id="mini-chart" height="80"></canvas>
    </div>
  </div>
</div>

<!-- FOOTER -->
<div id="footer">
  <div id="day-info">Day <b id="fi-day">1</b> / <b id="fi-total">30</b></div>
  <div id="slider-wrap">
    <button class="ctrl-btn" id="btn-prev">◀</button>
    <input type="range" id="day-slider" min="0" value="0"/>
    <button class="ctrl-btn" id="btn-next">▶</button>
  </div>
  <button class="ctrl-btn play" id="btn-play">▶  PLAY</button>
  <div id="streak" id="streak-info">–</div>
</div>

</div><!-- /app -->
<div id="tooltip"></div>

<script>
// ═══════════════════════════════════════════════════════════════════
// EMBEDDED DATA (generated by generate_fcm.py)
// ═══════════════════════════════════════════════════════════════════
const FCMS   = __FCMS_JSON__;
const SIMDATA= __SIMDATA_JSON__;
const META   = __META_JSON__;

// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════
let activeFCM = 0;
let dayIdx    = 0;
let playing   = false;
let playTimer = null;

// ═══════════════════════════════════════════════════════════════════
// THREE.JS SETUP
// ═══════════════════════════════════════════════════════════════════
const wrap   = document.getElementById('canvas-wrap');
const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
renderer.setPixelRatio(window.devicePixelRatio);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x020b18);
scene.fog = new THREE.FogExp2(0x020b18, 0.04);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
camera.position.set(0, 2, 10);

// Orbit controls (manual)
let isDragging=false, prevMouse={x:0,y:0}, spherical={theta:0,phi:Math.PI/3,r:10};
function applyCamera(){
  camera.position.set(
    spherical.r*Math.sin(spherical.phi)*Math.sin(spherical.theta),
    spherical.r*Math.cos(spherical.phi),
    spherical.r*Math.sin(spherical.phi)*Math.cos(spherical.theta)
  );
  camera.lookAt(0,0,0);
}
applyCamera();
canvas.addEventListener('mousedown', e=>{isDragging=true;prevMouse={x:e.clientX,y:e.clientY}});
window.addEventListener('mouseup',   ()=>isDragging=false);
window.addEventListener('mousemove', e=>{
  if(!isDragging)return;
  spherical.theta -= (e.clientX-prevMouse.x)*0.005;
  spherical.phi    = Math.max(0.2,Math.min(Math.PI-0.2,spherical.phi-(e.clientY-prevMouse.y)*0.005));
  prevMouse={x:e.clientX,y:e.clientY};
  applyCamera();
});
canvas.addEventListener('wheel', e=>{
  spherical.r = Math.max(4, Math.min(25, spherical.r + e.deltaY*0.01));
  applyCamera();
});

// Lights
scene.add(new THREE.AmbientLight(0x112244, 1.5));
const pLight = new THREE.PointLight(0x00d4ff, 2, 30);
pLight.position.set(0,5,0);
scene.add(pLight);

// ── Grid floor ──────────────────────────────────────────────────────
const grid = new THREE.GridHelper(20, 20, 0x0a2040, 0x0a2040);
grid.position.y = -3;
scene.add(grid);

// ── Particles ───────────────────────────────────────────────────────
(function buildParticles(){
  const geo = new THREE.BufferGeometry();
  const pts = [];
  for(let i=0;i<600;i++){
    pts.push((Math.random()-0.5)*30,(Math.random()-0.5)*20,(Math.random()-0.5)*30);
  }
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({color:0x1a4a7a,size:0.06})));
})();

// ═══════════════════════════════════════════════════════════════════
// FCM GRAPH OBJECTS
// ═══════════════════════════════════════════════════════════════════
let graphGroup = null;
let nodeObjects = {};   // nodeId → {mesh, ring, label, baseColor}
let edgeObjects = [];   // [{line, fromId, toId, weight}]

function hexToRGB(hex){
  const r=parseInt(hex.slice(1,3),16)/255;
  const g=parseInt(hex.slice(3,5),16)/255;
  const b=parseInt(hex.slice(5,7),16)/255;
  return [r,g,b];
}

function buildGraph(fcmIdx){
  if(graphGroup){scene.remove(graphGroup);}
  graphGroup = new THREE.Group();
  nodeObjects = {};
  edgeObjects = [];

  const fcm = FCMS[fcmIdx];
  const nodeMap = {};
  fcm.nodes.forEach(n=>nodeMap[n.id]=n);

  // ── Nodes ──────────────────────────────────────────────────────
  fcm.nodes.forEach(nd=>{
    const pos = new THREE.Vector3(nd.x, nd.y, nd.z);
    const rgb = hexToRGB(nd.color);
    const col = new THREE.Color(nd.color);

    const isTarget = !!nd.target;
    const r = isTarget ? 0.38 : 0.28;

    // Sphere
    const geo  = new THREE.SphereGeometry(r, 24, 24);
    const mat  = new THREE.MeshPhongMaterial({
      color: col, emissive: col, emissiveIntensity: 0.4,
      transparent: true, opacity: 0.9,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    graphGroup.add(mesh);

    // Glow ring
    const rGeo = new THREE.RingGeometry(r+0.05, r+0.15, 32);
    const rMat = new THREE.MeshBasicMaterial({color:col, side:THREE.DoubleSide, transparent:true, opacity:0.3});
    const ring = new THREE.Mesh(rGeo, rMat);
    ring.position.copy(pos);
    ring.lookAt(camera.position);
    graphGroup.add(ring);

    // Label sprite
    const label = makeLabel(nd.label, nd.color);
    label.position.set(pos.x, pos.y + r + 0.4, pos.z);
    graphGroup.add(label);

    nodeObjects[nd.id] = {mesh, ring, label, baseColor: col, pos, isTarget};
  });

  // ── Edges ──────────────────────────────────────────────────────
  fcm.edges.forEach(edge=>{
    const from = nodeMap[edge.from];
    const to   = nodeMap[edge.to];
    if(!from||!to) return;

    const posA = new THREE.Vector3(from.x,from.y,from.z);
    const posB = new THREE.Vector3(to.x,to.y,to.z);

    // Slightly curve the edge
    const mid = posA.clone().lerp(posB, 0.5).addScalar(0.2);
    const curve = new THREE.QuadraticBezierCurve3(posA, mid, posB);
    const pts   = curve.getPoints(24);
    const geo   = new THREE.BufferGeometry().setFromPoints(pts);

    const positive = edge.weight >= 0;
    const col = positive ? new THREE.Color(0x4ade80) : new THREE.Color(0xf87171);
    const mat = new THREE.LineBasicMaterial({
      color: col, transparent: true,
      opacity: Math.min(1, Math.abs(edge.weight) * 0.9 + 0.15),
    });
    const line = new THREE.Line(geo, mat);
    graphGroup.add(line);

    // Arrow head
    const dir = posB.clone().sub(posA).normalize();
    const arrowGeo = new THREE.ConeGeometry(0.06, 0.2, 8);
    const arrowMat = new THREE.MeshBasicMaterial({color:col, transparent:true, opacity:0.7});
    const arrow = new THREE.Mesh(arrowGeo, arrowMat);
    const arrowPos = posA.clone().lerp(posB, 0.88);
    arrow.position.copy(arrowPos);
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), dir);
    graphGroup.add(arrow);

    // Weight label
    const wLabel = makeLabel(edge.weight.toFixed(2), positive ? '#4ade80' : '#f87171', true);
    wLabel.position.copy(posA.clone().lerp(posB, 0.5));
    wLabel.position.y += 0.18;
    graphGroup.add(wLabel);

    edgeObjects.push({line, fromId:edge.from, toId:edge.to, weight:edge.weight, mat});
  });

  scene.add(graphGroup);
}

function makeLabel(text, color='#ffffff', small=false){
  const sz = small ? 200 : 280;
  const cvs = document.createElement('canvas');
  cvs.width = sz; cvs.height = small ? 60 : 80;
  const ctx = cvs.getContext('2d');

  if(!small){
    ctx.fillStyle = 'rgba(2,11,24,0.7)';
    ctx.beginPath();
    ctx.roundRect(4,4,cvs.width-8,cvs.height-8,8);
    ctx.fill();
  }

  ctx.fillStyle = color;
  ctx.font = `${small?18:20}px monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, cvs.width/2, cvs.height/2);

  const tex = new THREE.CanvasTexture(cvs);
  const mat = new THREE.SpriteMaterial({map:tex, transparent:true, depthWrite:false});
  const spr = new THREE.Sprite(mat);
  const scale = small ? 0.6 : 1.0;
  spr.scale.set(cvs.width/cvs.height*scale, scale, 1);
  return spr;
}

// ═══════════════════════════════════════════════════════════════════
// UPDATE NODE VISUALS FOR CURRENT DAY
// ═══════════════════════════════════════════════════════════════════
function updateNodeVisuals(simRow){
  if(!simRow) return;
  const vals = simRow.nodeValues;
  Object.entries(nodeObjects).forEach(([nid, obj])=>{
    const v = vals[nid] ?? 0;
    const intensity = Math.abs(v);
    obj.mesh.material.emissiveIntensity = 0.3 + intensity * 0.9;

    // Pulse scale
    const scale = 1 + intensity * 0.3;
    obj.mesh.scale.setScalar(scale);
    obj.ring.material.opacity = 0.15 + intensity * 0.5;

    // Color shift for negative values
    if(v < -0.1){
      obj.mesh.material.emissive.setHex(0xff3355);
    } else if(v > 0.1){
      obj.mesh.material.emissive.copy(obj.baseColor);
    } else {
      obj.mesh.material.emissive.setHex(0x334455);
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// ACCURACY RING (canvas 2D)
// ═══════════════════════════════════════════════════════════════════
function drawRing(pct){
  const c  = document.getElementById('ring-canvas');
  const cx = c.getContext('2d');
  const W  = c.width, H = c.height;
  const cx2= W/2, cy2= H/2, R= W/2 - 12;
  cx.clearRect(0,0,W,H);

  // track
  cx.beginPath();cx.arc(cx2,cy2,R,0,Math.PI*2);
  cx.strokeStyle='rgba(0,100,180,0.25)';cx.lineWidth=8;cx.stroke();

  // value arc
  const start = -Math.PI/2;
  const end   = start + (pct/100)*Math.PI*2;
  const grad  = cx.createLinearGradient(0,0,W,H);
  grad.addColorStop(0,'#00d4ff');grad.addColorStop(1,'#4ade80');
  cx.beginPath();cx.arc(cx2,cy2,R,start,end);
  cx.strokeStyle=grad;cx.lineWidth=8;cx.lineCap='round';cx.stroke();

  // glow
  cx.beginPath();cx.arc(cx2,cy2,R,start,end);
  cx.strokeStyle='rgba(0,212,255,0.18)';cx.lineWidth=18;cx.stroke();

  document.getElementById('acc-num').textContent = pct.toFixed(1)+'%';
}

// ═══════════════════════════════════════════════════════════════════
// MINI ACCURACY CHART
// ═══════════════════════════════════════════════════════════════════
function drawMiniChart(simData, currentDay){
  const mc   = document.getElementById('mini-chart');
  mc.width   = mc.parentElement.clientWidth - 24;
  const ctx  = mc.getContext('2d');
  const W    = mc.width, H = mc.height;
  ctx.clearRect(0,0,W,H);

  const n   = simData.length;
  const bW  = W / n - 1;

  simData.forEach((r,i)=>{
    const x = i*(bW+1);
    const col = r.correct ? (i===currentDay?'#22ff88':'#4ade8066') : (i===currentDay?'#ff4455':'#f8717166');
    ctx.fillStyle = col;
    ctx.fillRect(x, 0, bW, H);
    if(i===currentDay){
      ctx.strokeStyle='#fff';ctx.lineWidth=1.5;
      ctx.strokeRect(x,0,bW,H);
    }
  });

  // rolling accuracy line
  ctx.beginPath();
  simData.forEach((r,i)=>{
    const window_data = simData.slice(0, i+1);
    const acc = window_data.filter(d=>d.correct).length/window_data.length;
    const x = i*(bW+1)+bW/2;
    const y = H - acc*H;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.strokeStyle='#00d4ff';ctx.lineWidth=1.5;ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════════
// NODE ROWS IN SIDE PANEL
// ═══════════════════════════════════════════════════════════════════
function buildNodeRows(fcmIdx){
  const fcm = FCMS[fcmIdx];
  const container = document.getElementById('node-rows');
  container.innerHTML = '';
  fcm.nodes.forEach(nd=>{
    const row = document.createElement('div');
    row.className = 'node-row';
    row.dataset.id = nd.id;
    row.innerHTML = `
      <span class="node-dot" style="background:${nd.color}"></span>
      <span class="node-name">${nd.label}</span>
      <div class="node-bar-wrap"><div class="node-bar" id="bar-${nd.id}" style="background:${nd.color};width:50%"></div></div>
      <span class="node-val" id="nval-${nd.id}">0.00</span>`;
    container.appendChild(row);
  });
}

function updateNodeRows(simRow){
  if(!simRow) return;
  Object.entries(simRow.nodeValues).forEach(([nid, v])=>{
    const bar = document.getElementById('bar-'+nid);
    const val = document.getElementById('nval-'+nid);
    if(!bar||!val) return;
    const pct = ((v + 1) / 2 * 100).toFixed(0);
    bar.style.width = pct+'%';
    bar.style.background = v >= 0 ? '#4ade80' : '#f87171';
    val.textContent = v.toFixed(3);
    val.style.color = v >= 0 ? '#4ade80' : '#f87171';
  });
}

// ═══════════════════════════════════════════════════════════════════
// DAY CARD UPDATE
// ═══════════════════════════════════════════════════════════════════
function updateDayCard(simRow){
  if(!simRow) return;
  document.getElementById('dc-date').textContent    = simRow.date;
  const pd = document.getElementById('dc-pred');
  pd.textContent = simRow.predictedDir;
  pd.className   = 'dc-val ' + (simRow.predictedDir==='UP'?'up':'down');

  const ad = document.getElementById('dc-actual');
  const sign = simRow.actualPct >= 0 ? '+' : '';
  ad.textContent = simRow.actualDir + ' ('+sign+simRow.actualPct.toFixed(2)+'%)';
  ad.className   = 'dc-val ' + (simRow.actualDir==='UP'?'up':'down');

  document.getElementById('dc-close').textContent = '$'+simRow.nasdaqClose.toLocaleString();

  const badge = document.getElementById('dc-badge');
  badge.textContent = simRow.correct ? '✓  CORRECT PREDICTION' : '✗  INCORRECT PREDICTION';
  badge.className   = 'correct-badge ' + (simRow.correct?'ok':'bad');
}

// ═══════════════════════════════════════════════════════════════════
// SWITCH FCM
// ═══════════════════════════════════════════════════════════════════
function switchFCM(idx){
  activeFCM = idx;
  dayIdx    = 0;

  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i===idx));
  buildGraph(idx);
  buildNodeRows(idx);

  const sim = SIMDATA[idx];
  document.getElementById('day-slider').max   = sim.results.length - 1;
  document.getElementById('day-slider').value = 0;
  document.getElementById('fi-total').textContent = sim.results.length;
  drawRing(sim.accuracy);
  renderDay(0);
}

// ═══════════════════════════════════════════════════════════════════
// RENDER A SPECIFIC DAY
// ═══════════════════════════════════════════════════════════════════
function renderDay(d){
  dayIdx = d;
  const sim    = SIMDATA[activeFCM];
  const simRow = sim.results[d];

  document.getElementById('day-slider').value = d;
  document.getElementById('fi-day').textContent = d + 1;

  updateDayCard(simRow);
  updateNodeRows(simRow);
  updateNodeVisuals(simRow);

  // Rolling accuracy up to this day
  const subset = sim.results.slice(0, d+1);
  const rolling = subset.filter(r=>r.correct).length / subset.length * 100;
  drawRing(rolling);
  drawMiniChart(sim.results, d);

  // Streak
  let streak=0;
  for(let i=d;i>=0;i--){
    if(sim.results[i].correct===sim.results[d].correct) streak++;
    else break;
  }
  document.getElementById('streak').textContent =
    (sim.results[d].correct?'✓':'✗') + ' Streak: '+streak;
}

// ═══════════════════════════════════════════════════════════════════
// CONTROLS
// ═══════════════════════════════════════════════════════════════════
document.getElementById('day-slider').addEventListener('input', e=>{
  renderDay(parseInt(e.target.value));
});
document.getElementById('btn-prev').addEventListener('click',()=>{
  if(dayIdx>0) renderDay(dayIdx-1);
});
document.getElementById('btn-next').addEventListener('click',()=>{
  const max = SIMDATA[activeFCM].results.length - 1;
  if(dayIdx<max) renderDay(dayIdx+1);
});
document.getElementById('btn-play').addEventListener('click',()=>{
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? '⏸  PAUSE' : '▶  PLAY';
  if(playing) autoPlay();
});
function autoPlay(){
  if(!playing) return;
  const max = SIMDATA[activeFCM].results.length - 1;
  if(dayIdx >= max){ playing=false; document.getElementById('btn-play').textContent='▶  PLAY'; return; }
  renderDay(dayIdx+1);
  playTimer = setTimeout(autoPlay, 900);
}

document.querySelectorAll('.tab').forEach((t,i)=>{
  t.addEventListener('click',()=>{ playing=false; document.getElementById('btn-play').textContent='▶  PLAY'; switchFCM(i); });
});

// ═══════════════════════════════════════════════════════════════════
// RESIZE
// ═══════════════════════════════════════════════════════════════════
function onResize(){
  const W = wrap.clientWidth, H = wrap.clientHeight;
  renderer.setSize(W, H);
  camera.aspect = W/H;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);
onResize();

// ═══════════════════════════════════════════════════════════════════
// ANIMATION LOOP
// ═══════════════════════════════════════════════════════════════════
let t=0;
function animate(){
  requestAnimationFrame(animate);
  t += 0.01;

  // Bill-board rings toward camera
  Object.values(nodeObjects).forEach(obj=>{
    obj.ring.lookAt(camera.position);
    obj.mesh.material.emissiveIntensity = obj.mesh.material.emissiveIntensity * 0.98 + (0.3 + Math.sin(t*1.2)*0.1) * 0.02;
  });

  // Gentle auto-rotate when idle
  if(!isDragging){
    spherical.theta += 0.0008;
    applyCamera();
  }

  renderer.render(scene, camera);
}
animate();

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════
document.getElementById('date-range-badge').textContent =
  META.startDate + ' → ' + META.endDate + '  |  ' + META.simDays + ' sim days';

switchFCM(0);
</script>
</body>
</html>
"""


def build_html(fcms_json: str, simdata_json: str, meta_json: str) -> str:
    html = HTML_TEMPLATE
    html = html.replace("__FCMS_JSON__",    fcms_json)
    html = html.replace("__SIMDATA_JSON__", simdata_json)
    html = html.replace("__META_JSON__",    meta_json)
    return html


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_data()

    sim_results = []
    for fcm_def in FCMS:
        print(f"Simulating {fcm_def['name']} …")
        res = simulate(fcm_def, df)
        sim_results.append(res)
        print(f"  Accuracy: {res['accuracy']:.1f}%  ({len(res['results'])} days)")

    sim_dates = df.index[-(SIM_DAYS):].strftime("%Y-%m-%d").tolist()
    meta = {
        "startDate": sim_dates[0],
        "endDate":   sim_dates[-1],
        "simDays":   SIM_DAYS,
    }

    html = build_html(
        fcms_json    = json.dumps(FCMS,        separators=(",",":")),
        simdata_json = json.dumps(sim_results,  separators=(",",":")),
        meta_json    = json.dumps(meta,          separators=(",",":")),
    )

    out_path = "fcm_visualization.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓  Visualization saved → {out_path}")
    print("  Open it in a browser (internet required for Three.js CDN).")
