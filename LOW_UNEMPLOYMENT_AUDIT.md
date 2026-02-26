# Low_Unemployment Data Availability Audit

## Problem Statement
User reported that Low_Unemployment data is not being picked up by all visuals in the system.

## Data Flow Analysis

### 1. **Data Source** (fcm_nasdaq.py)
- **Configuration** (line 134): `"Low_Unemployment": []` - No ticker, data from FRED API
- **Fetch Function** (lines 52-111): `fetch_unemployment_data_fred()`
  - Requires: `FRED_API_KEY` environment variable
  - Falls back to constant 0.5 if key not set
  - Returns: pandas Series normalized to [0, 1]

- **Integration** (lines 527-535 in build_proxies_from_prices):
  ```
  if node == "Low_Unemployment" and unemployment_data is not None:
      inputs[node] = unemployment_data.reindex(price_df.index).fillna(method='ffill').fillna(0.5)
      logger.info(f"Using FRED unemployment data for {node}")
  else:
      inputs[node] = pd.Series(0.5, index=price_df.index)
      logger.info(f"FRED unemployment not available; using constant 0.5 for {node}")
  ```

### 2. **Data Transformation** (fcm_nasdaq.py)
- **inputs_norm_yahoo** (line 1123): Contains Low_Unemployment column after normalization
- **inputs_norm_google** (line 1124): Contains Low_Unemployment column after normalization

### 3. **Visualization Consumption** (visualize_fcm.py)

#### Panel 1: Sentiment Time Series (lines 56-144)
- **Status**: ✅ May include Low_Unemployment if in 4 key nodes
- **Issue**: Uses hardcoded key_nodes list that includes "Oil_Prices" which doesn't exist
- **Fix needed**: Check if Low_Unemployment is actually in correlations_history nodes

#### Panel 2: Feature Importance (lines 693-746)
- **Status**: ✅ Displayed as correlation with target
- **Data source**: inputs_norm_yahoo.corrwith(actual_norm)
- **Note**: Will only show if Low_Unemployment > 0 variance

#### Panel 3: Dynamic FCM Structure (lines 468-649)
- **Status**: ❌ Uses hardcoded node lists (outer_nodes, inner_nodes)
- **Missing**: Low_Unemployment may not be in the hardcoded visualization config
- **Note**: Removed animated version - uses plot_dynamic_fcm_structure() instead

#### Panel 4+: Node Interaction Heatmap (lines 681-758)
- **Status**: ⚠️ Conditional display
- **Data source**: Low_Unemployment in node_names (correlations_history)
- **Issue**: Depends on whether Low_Unemployment has variance >0 for correlation calculation

#### Weight Evolution Plot (lines 805-877)
- **Status**: ⚠️ Only if in final weights dictionary
- **Issue**: Low_Unemployment only flows to inner nodes, not NASDAQ directly
- **Note**: May not appear if it has zero weight

#### Stock-level Visualizations (lines 1020+)
- **Status**: ❌ Don't use FCM nodes at all
- **Scope**: Only analyze individual stock prices

## Verification Checklist

### For Low_Unemployment to appear in visualizations:

1. **Environment Setup** ❌
   - [ ] FRED_API_KEY environment variable is set
   - Without this, Low_Unemployment is constant 0.5 (no variance, won't correlate)

2. **Data Pipeline** ✅
   - [x] fcm_nasdaq.py correctly builds inputs with Low_Unemployment
   - [x] Normalization happens correctly
   - [x] Node names should include "Low_Unemployment"

3. **Visualization Pipeline** ⚠️
   - [ ] plot_sentiment_time_series: Uses hardcoded key_nodes that may not include it
   - [ ] plot_dynamic_fcm_structure: Uses hardcoded node positions - LOW_UNEMPLOYMENT **IS** INCLUDED at (1.5, 8)
   - [ ] plot_node_interaction_heatmap: Depends on correlation availability
   - [ ] plot_weight_evolution: Depends on weights > 0

## Key Finding

**Root Cause**: Low_Unemployment will ONLY be picked up if:
1. FRED_API_KEY environment variable is set, AND
2. The data has variance (>0 std dev) to correlate with target

**Without FRED_API_KEY**: Low_Unemployment defaults to constant 0.5
- No variance
- Will not correlate with anything
- Will not appear in correlation-based visualizations
- Will have 0 weights in FCM

## Recommended Actions

1. **Set FRED_API_KEY**:
   ```bash
   export FRED_API_KEY=<your_fred_key>
   # Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
   ```

2. **Verify in visualize_fcm.py**:
   - Line 747: Check if 'Low_Unemployment' is in available_key_nodes
   - Add logging to show which nodes are actually being displayed

3. **Check plot_dynamic_fcm_structure()**:
   - Verify Low_Unemployment node is positioned correctly at (1.5, 8)
   - Should display in all quarterly snapshots

## Status Summary

| Component | Status | Data Source | Issue |
|-----------|--------|-------------|-------|
| fcm_nasdaq.py | ✅ | FRED API (requires key) | Needs FRED_API_KEY env var |
| inputs_norm_yahoo | ✅ | Built correctly | None |
| inputs_norm_google | ✅ | Built correctly | None |
| plot_sentiment_time_series | ⚠️ | Hardcoded list | May skip if not in key_nodes |
| plot_dynamic_fcm_structure | ✅ | Hardcoded position | Low_Unemployment at (1.5, 8) |
| plot_node_interaction_heatmap | ⚠️ | Correlations | Only if variance > 0 |
| plot_weight_evolution | ⚠️ | FCM weights | Only if weight > 0 |

