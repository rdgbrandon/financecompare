#!/usr/bin/env python
"""Test which visualization functions are displaying data"""
import sys
import os
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from fcm_nasdaq import prepare_data_and_run
from visualize_fcm import (
    plot_animated_fcm_structure,
    plot_sentiment_time_series,
    plot_model_predictions,
    plot_sentiment_vs_returns,
    plot_weight_evolution
)

logger = logging.getLogger("test")

print("\n" + "="*80)
print("TESTING VISUALIZATION FUNCTIONS - WHICH ONES HAVE DATA?")
print("="*80)

print("\n1. Fetching data...")
out = prepare_data_and_run()

test_functions = [
    ("plot_animated_fcm_structure", plot_animated_fcm_structure),
    ("plot_sentiment_time_series", plot_sentiment_time_series),
    ("plot_model_predictions", plot_model_predictions),
    ("plot_sentiment_vs_returns", plot_sentiment_vs_returns),
    ("plot_weight_evolution", plot_weight_evolution),
]

print("\n2. Testing each visualization function...\n")

for func_name, func in test_functions:
    print(f"   Testing {func_name}...", end=" ")
    try:
        # Check if required data exists first
        if func_name == "plot_animated_fcm_structure":
            needs = ["weights_history_yahoo", "weights_history_google", "inputs_norm_yahoo", "inputs_norm_google", "actual_norm"]
            has_data = all(out.get(k) is not None for k in needs)
            print(f"(Has required data: {has_data})", end=" ")
        elif func_name == "plot_sentiment_time_series":
            needs = ["polarity_series_yahoo", "polarity_series_google", "polarity_series"]
            has_data = any(out.get(k) is not None and len(out.get(k, []))  > 0 for k in needs)
            print(f"(Has sentiment data: {has_data})", end=" ")
        elif func_name == "plot_model_predictions":
            needs = ["actual_norm", "date_history_yahoo", "pred_history_yahoo"]
            has_data = all(out.get(k) is not None for k in needs[:1]) and len(out.get("date_history_yahoo", [])) > 0
            print(f"(Has prediction data: {has_data})", end=" ")
        elif func_name == "plot_sentiment_vs_returns":
            needs = ["polarity_series", "actual_norm"]
            has_data = out.get("polarity_series") is not None and out.get("actual_norm") is not None
            print(f"(Has correlation data: {has_data})", end=" ")
        elif func_name == "plot_weight_evolution":
            needs = ["weights_history_yahoo", "date_history_yahoo"]
            has_data = out.get("weights_history_yahoo") is not None and len(out.get("date_history_yahoo", [])) > 0
            print(f"(Has weight history: {has_data})", end=" ")
        
        # Call function (matplotlib will not display in batch mode)
        func(out)
        print("✅ SUCCESS")
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:80]}")

print("\n" + "="*80)
print("SUMMARY OF DATA IN OUT DICTIONARY:")
print("="*80)

# Summary of what data exists
sentiment_keys = [k for k in out.keys() if 'sentiment' in k.lower() or 'polarity' in k.lower()]
print(f"\nSentiment-related keys: {sentiment_keys}")

for key in sentiment_keys:
    val = out.get(key)
    if val is None:
        print(f"  {key}: None")
    elif hasattr(val, '__len__'):
        print(f"  {key}: {type(val).__name__} with {len(val)} items")
    else:
        print(f"  {key}: {type(val).__name__}")

print("\n✅ Test completed")
