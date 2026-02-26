#!/usr/bin/env python
"""Quick test to see what sentiment keys are available"""
import sys
import logging

# Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

try:
    from fcm_nasdaq import prepare_data_and_run
    print("Calling prepare_data_and_run()...",  file=sys.stderr)
    out = prepare_data_and_run()
    
    # Get sentiment-related keys
    sentiment_keys = [k for k in out.keys() if 'senti' in k.lower() or 'polar' in k.lower()]
    
    print("\n" + "="*60)
    print("SENTIMENT DATA KEYS FOUND:")
    print("="*60)
    print(f"Keys: {sentiment_keys}")
    print(f"Total output keys: {len(out.keys())}")
    
    # Check each sentiment key
    for key in sentiment_keys:
        val = out.get(key)
        if val is None:
            print(f"\n{key}: None")
        elif hasattr(val, '__len__'):
            print(f"\n{key}: {type(val).__name__} with {len(val)} items")
            if len(val) > 0:
                print(f"  First few: {str(val)[:100]}")
        else:
            print(f"\n{key}: {type(val).__name__}")
    
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
