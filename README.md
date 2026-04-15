# FCM AI Trading Platform - Streamlit Edition

A production-grade, professional **Streamlit-based trading dashboard** featuring real-time market data, advanced sentiment analysis, FCM (Fuzzy Cognitive Map) predictions, deep learning models, and portfolio backtesting.

## ✨ Features

### Core Capabilities
- **Real-time Market Data**: Yahoo Finance integration for OHLCV data
- **News & Sentiment Analysis**: NLP-powered sentiment extraction from financial news
- **FCM Modeling**: Fuzzy Cognitive Map predictions with dynamic weight learning
- **Deep Learning Models**: LSTM/GRU integration for price predictions
- **Portfolio Backtesting**: Complete backtest engine with risk metrics
- **Professional UI**: Broker-style dashboard with multiple pages

### Technical Features
- **Snapshot Caching**: API-efficient data retrieval with TTL-based caching (30-60 sec for market, 5-10 min for news)
- **Rate Limiting Protection**: Built-in safeguards against API throttling
- **Hybrid Signal Generation**: Multi-model signal aggregation with configurable weights
- **Technical Indicators**: SMA, RSI, MACD, Bollinger Bands, ATR
- **Risk Metrics**: Sharpe ratio, Sortino ratio, max drawdown, win rate

## 📊 Dashboard Pages

1. **Dashboard**: Portfolio overview, real-time signals, ticker cards
2. **Market Data**: Candlestick charts with technical analysis
3. **News & Sentiment**: News feed with sentiment timeline
4. **FCM Insights**: Model performance and feature importance
5. **Backtesting**: Historical simulation with performance metrics
6. **Settings**: Configuration and app information

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

`ash
# Navigate to project directory
cd financecompare

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run dashboard.py
`

The app will start at http://localhost:8501

## 📁 Project Structure

`
financecompare/
├── dashboard.py                # Main Streamlit application (consolidated)
├── fcm_nasdaq.py               # FCM model with LSTM/GRU
├── visualize_fcm.py            # FCM visualization utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
`

### File Descriptions

- **dashboard.py**: Complete Streamlit application containing:
  - SnapshotManager: API caching with TTL-based snapshots
  - Portfolio: Backtesting engine with risk metrics
  - HybridSignalGenerator: Multi-model signal generation
  - Trade & Position management
  - All 6 dashboard pages and navigation
  - Data fetching and sentiment analysis functions

- **fcm_nasdaq.py**: Fuzzy Cognitive Map modeling module with LSTM/GRU deep learning

- **visualize_fcm.py**: Visualization utilities for FCM model outputs

## ⚙️ Configuration

### Model Weights (Sidebar)
Adjust how different prediction models contribute to signals:
- **FCM Weight**: 0.0-1.0 (default 0.5)
- **Deep Learning Weight**: 0.0-1.0 (default 0.3)
- **Sentiment Weight**: 0.0-1.0 (default 0.2)

### Snapshot Settings (Sidebar)
Configure cache TTLs to balance API efficiency:
- **Market Data TTL**: 30-300 seconds (default 60)
- **News TTL**: 300-3600 seconds (default 600)

## 🔄 How Snapshot Caching Works

The SnapshotManager caches API responses with configurable TTL (time-to-live):
- **Market Data**: Cached for 30-60 seconds (cheap to refresh frequently)
- **News Data**: Cached for 5-10 minutes (expensive, lower refresh rate)
- **Sentiment**: Cached with news data (single NLP computation)

This design balances real-time accuracy with API rate limit protection.

## 📈 Dashboard Features

### Dashboard Page
- Portfolio overview with current holdings
- Real-time buy/sell signals from hybrid model
- Sentiment timeline for selected ticker
- Market sentiment aggregation

### Market Data Page
- Interactive candlestick charts (Plotly)
- Technical indicators: SMA (20, 50, 200), RSI, MACD, Bollinger Bands
- Volume analysis with ATR

### News & Sentiment Page
- Live news feed for selected ticker
- Sentiment scores: positive/negative/neutral
- Historical sentiment timeline
- News source and publication date

### FCM Insights Page
- Model performance metrics over time
- Feature importance visualization
- FCM state variables and their values
- Model accuracy and prediction confidence

### Backtesting Page
- Historical performance simulation
- Configurable parameter optimization
- Risk metrics: Sharpe ratio, Sortino ratio, max drawdown, win rate
- Returns comparison vs. buy-and-hold

### Settings Page
- Application configuration guide
- Model weight explanations
- Data source information
- Cache TTL settings

## 🛠️ Technical Stack

- **Frontend**: Streamlit 1.28+
- **Data**: YFinance, Pandas, NumPy
- **ML/AI**: Scikit-learn, TensorFlow/Keras, TextBlob
- **Visualization**: Plotly, Matplotlib
- **Data Structure**: NetworkX for FCM graphs
- **NLP**: Transformers (optional), TextBlob

## 📝 Usage Example

After starting the application (streamlit run dashboard.py):

1. Open http://localhost:8501 in your browser
2. Select a stock ticker (e.g., AAPL)
3. View real-time signals on the Dashboard page
4. Analyze technical indicators on Market Data page
5. Review sentiment trends on News & Sentiment page
6. Backtest strategies on Backtesting page
7. Adjust model weights in sidebar to customize predictions

## ⚠️ Important Notes

- The application uses YFinance (free, but has rate limits)
- Sentiment analysis requires internet access for news
- First run may take longer due to model initialization
- TensorFlow/Keras are optional; LSTM predictions may not run without them
- Cache TTLs affect data freshness and API calls - adjust based on your needs

## 📞 Support

For issues or questions:
1. Check that all dependencies are installed: pip install -r requirements.txt
2. Ensure you're running with Streamlit CLI: streamlit run dashboard.py
3. Review the Settings page for configuration details
4. Check logs for error messages

## 📄 License

This project is provided as-is for educational and research purposes.
