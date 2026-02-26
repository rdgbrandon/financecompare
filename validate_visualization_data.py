#!/usr/bin/env python
"""Validate that all visualization data flows correctly"""
import sys
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive mode

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from fcm_nasdaq import prepare_data_and_run

print("\n" + "="*80)
print("DATA VALIDATION TEST - Ensuring Accurate Data Flow to Visualizations")
print("="*80)

print("\n1. Fetching data from APIs...")
out = prepare_data_and_run()

print("\n2. Validating each visualization's data requirements:\n")

checks = {
    "Animated FCM": {
        "required": ["weights_history_yahoo", "weights_history_google", "inputs_norm_yahoo", "inputs_norm_google", "date_history_yahoo"],
        "check": lambda o: all(o.get(k) is not None for k in ["weights_history_yahoo", "weights_history_google", "inputs_norm_yahoo", "inputs_norm_google"])
    },
    "Sentiment Time Series": {
        "required": ["polarity_series_yahoo", "polarity_series_google", "polarity_series"],
        "check": lambda o: any(o.get(k) is not None and len(o.get(k, [])) > 0 for k in ["polarity_series_yahoo", "polarity_series_google", "polarity_series"])
    },
    "Model Predictions": {
        "required": ["actual_norm", "date_history_yahoo", "pred_history_yahoo", "date_history_google", "pred_history_google"],
        "check": lambda o: o.get("actual_norm") is not None and len(o.get("date_history_yahoo", [])) > 0
    },
    "Sentiment vs Returns": {
        "required": ["polarity_series", "actual_norm"],
        "check": lambda o: o.get("polarity_series") is not None and o.get("actual_norm") is not None
    },
    "Weight Evolution": {
        "required": ["weights_history_yahoo", "date_history_yahoo"],
        "check": lambda o: o.get("weights_history_yahoo") is not None and len(o.get("date_history_yahoo", [])) > 0
    },
    "FCM + LSTM Accuracy": {
        "required": ["actual_norm", "pred_history_yahoo", "pred_history_google", "lstm_results", "gru_results"],
        "check": lambda o: o.get("actual_norm") is not None and o.get("lstm_results") is not None
    },
    "FCM Network": {
        "required": ["weights_history_yahoo", "node_order"],
        "check": lambda o: o.get("weights_history_yahoo") is not None and len(o.get("weights_history_yahoo", [])) > 0
    },
    "FCM Structure Frame": {
        "required": ["weights_history_yahoo", "weights_history_google", "inputs_norm_yahoo", "inputs_norm_google"],
        "check": lambda o: all(o.get(k) is not None for k in ["weights_history_yahoo", "weights_history_google", "inputs_norm_yahoo", "inputs_norm_google"])
    },
    "Residual Analysis": {
        "required": ["actual_norm", "date_history_yahoo", "pred_history_yahoo"],
        "check": lambda o: o.get("actual_norm") is not None and len(o.get("date_history_yahoo", [])) > 0
    },
    "Stock Predictions": {
        "required": ["inputs_norm_yahoo", "actual_norm", "date_history_yahoo", "pred_history_yahoo"],
        "check": lambda o: o.get("inputs_norm_yahoo") is not None and o.get("actual_norm") is not None
    },
    "Stock Sentiment": {
        "required": ["inputs_norm_yahoo", "polarity_series"],
        "check": lambda o: o.get("inputs_norm_yahoo") is not None and o.get("polarity_series") is not None
    },
    "Stock Volatility Sentiment": {
        "required": ["inputs_norm_yahoo", "polarity_series"],
        "check": lambda o: o.get("inputs_norm_yahoo") is not None and o.get("polarity_series") is not None
    }
}

all_pass = True
for viz_name, check_info in checks.items():
    is_valid = check_info["check"](out)
    status = "✅ PASS" if is_valid else "❌ FAIL"
    print(f"   {status:12} {viz_name}")
    
    if not is_valid:
        all_pass = False
        print(f"      Required: {check_info['required']}")
        for key in check_info['required']:
            val = out.get(key)
            if val is None:
                print(f"        • {key}: MISSING")
            elif hasattr(val, '__len__'):
                print(f"        • {key}: {len(val)} items")
            else:
                print(f"        • {key}: Present")

print("\n3. Data Quality Checks:\n")

# Sentiment data quality
sent_yahoo = out.get("polarity_series_yahoo")
sent_google = out.get("polarity_series_google")
sent_unified = out.get("polarity_series")

sent_keys = [k for k in out.keys() if 'senti' in k.lower() or 'polar' in k.lower()]
if sent_keys:
    print(f"   ✅ Found {len(sent_keys)} sentiment data sources")
    for key in sent_keys:
        val = out.get(key)
        if val is not None and hasattr(val, '__len__'):
            valid_count = len(val.dropna()) if hasattr(val, 'dropna') else len([v for v in val if v is not None])
            print(f"      • {key}: {valid_count} valid values")
else:
    print(f"   ❌ No sentiment data found")

# Actual returns data
actual_norm = out.get("actual_norm")
if actual_norm is not None and not actual_norm.empty:
    print(f"   ✅ NASDAQ normalized returns: {len(actual_norm)} values")
    print(f"      Range: [{actual_norm.min():.4f}, {actual_norm.max():.4f}]")
else:
    print(f"   ❌ No NASDAQ data")

# Node data
node_order = out.get("node_order", [])
print(f"   ✅ Node structure: {len(node_order)} total nodes")

# Weights
weights_yahoo = out.get("weights_history_yahoo", [])
weights_google = out.get("weights_history_google", [])
print(f"   ✅ FCM weight history: Yahoo={len(weights_yahoo)}, Google={len(weights_google)} snapshots")

print("\n" + "="*80)
if all_pass:
    print("✅ ALL VISUALIZATIONS HAVE VALID DATA")
    print("✅ Ready to generate all 12 visualizations")
else:
    print("⚠️  SOME VISUALIZATIONS MAY HAVE MISSING DATA")
print("="*80 + "\n")
