# NASDAQ Fuzzy Cognitive Map (FCM)

A Fuzzy Cognitive Map implementation for analyzing and predicting NASDAQ movements based on economic indicators and market factors.

## Overview

This FCM model integrates real-time data from Yahoo Finance (outer nodes) with computed internal factors (inner nodes) to model NASDAQ behavior over time.

### Node Structure

**Outer Nodes** (from Yahoo Finance):
- Large Deficits
- Foreign Demand
- Low Unemployment
- Supply Increase
- Energy Price Increase
- Consumer Demand
- Business Investment
- Market Volatility
- Equity Inflows

**Inner Nodes** (computed):
- Monetary Policy and Interest Rates
- Inflation
- Corporate Earnings
- Investor Sentiment
- NASDAQ (target prediction)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from fcm_nasdaq import NASDAQ_FCM
from datetime import datetime, timedelta

# Create FCM instance
fcm = NASDAQ_FCM()

# Visualize the weight matrix
fcm.visualize_weights()

# Run simulation for past year
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

results = fcm.simulate_time_series(start_date, end_date)

# Plot results
fcm.plot_results(results)
```

### Running from Command Line

```bash
python fcm_nasdaq.py
```

This will:
1. Generate a weight matrix visualization (`fcm_weight_matrix.png`)
2. Fetch data from Yahoo Finance for the past year
3. Run the FCM simulation
4. Save results to `fcm_nasdaq_results.csv`
5. Generate visualization plots (`fcm_simulation_results.png`)

## How It Works

### Time Series Integration

The FCM operates on time-series data:
1. **Data Fetching**: Yahoo Finance data is fetched for specified date range
2. **Normalization**: All values are normalized to [0, 1] range
3. **FCM Computation**: For each time point:
   - Outer nodes are set from normalized Yahoo Finance data
   - Inner nodes are computed iteratively until convergence
   - Results capture the state of all nodes at that point in time

### Weight Matrix

The weight matrix defines causal relationships:
- **Positive weights** (green arrows): Factor A increases → Factor B increases
- **Negative weights** (red arrows): Factor A increases → Factor B decreases
- **Weight magnitude**: Strength of the relationship (0.0 to 1.0)

### Yahoo Finance Proxies

| Node | Ticker | Description |
|------|--------|-------------|
| Large Deficits | ^IRX | 13 Week Treasury Bill |
| Foreign Demand | UUP | US Dollar Index |
| Supply Increase | XLI | Industrial Sector ETF |
| Energy Price Increase | XLE | Energy Sector ETF |
| Consumer Demand | XLY | Consumer Discretionary ETF |
| Business Investment | XLI | Industrial Sector ETF |
| Market Volatility | ^VIX | CBOE Volatility Index |
| Equity Inflows | SPY | S&P 500 ETF |

## Customization

### Adjusting Weights

Edit the `_initialize_weights()` method in `fcm_nasdaq.py`:

```python
# Example: Increase impact of inflation on NASDAQ
set_weight('Inflation', 'NASDAQ', -0.7)  # Changed from -0.4
```

### Adding New Nodes

1. Add node name to `self.nodes` list
2. Define connections in `_initialize_weights()`
3. If it's an outer node, add Yahoo Finance ticker proxy in `fetch_yahoo_finance_data()`

### Custom Date Ranges

```python
results = fcm.simulate_time_series('2020-01-01', '2024-12-31')
```

## Output Files

- `fcm_weight_matrix.png`: Heatmap of causal relationships
- `fcm_simulation_results.png`: Time series plots of predictions
- `fcm_nasdaq_results.csv`: Detailed numerical results

## Model Interpretation

The FCM captures complex interdependencies:
- Rising interest rates → Lower corporate earnings → Lower NASDAQ
- High inflation → Negative investor sentiment → Lower NASDAQ
- Strong corporate earnings + Positive sentiment → Higher NASDAQ

The model converges iteratively, capturing feedback loops and indirect effects.

## Limitations

- Yahoo Finance proxies may not perfectly represent conceptual nodes
- Weight values are set heuristically (can be optimized with historical data)
- Model assumes causal relationships are stable over time
- Some factors (e.g., "Low Unemployment") may require alternative data sources

## Future Enhancements

- Integration with FRED API for unemployment data
- Machine learning optimization of weights
- Sentiment analysis from news sources
- Multi-step ahead predictions
- Uncertainty quantification
