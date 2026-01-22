"""
Fuzzy Cognitive Map for NASDAQ Analysis
Integrates external factors from Yahoo Finance with internal computed nodes
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class NASDAQ_FCM:
    def __init__(self):
        """
        Initialize FCM with nodes based on the diagram

        Outer nodes (from Yahoo Finance):
        - Large Deficits
        - Foreign Demand
        - Low Unemployment
        - Supply Increase
        - Energy Price Increase
        - Consumer Demand
        - Business Investment
        - Market Volatility
        - Equity Inflows

        Inner nodes (computed):
        - Monetary Policy and Interest Rates
        - Inflation
        - Corporate Earnings
        - Investor Sentiment
        - NASDAQ (target)
        """

        # Define all nodes
        self.nodes = [
            # Outer nodes (input from Yahoo Finance)
            'Large_Deficits',
            'Foreign_Demand',
            'Low_Unemployment',
            'Supply_Increase',
            'Energy_Price_Increase',
            'Consumer_Demand',
            'Business_Investment',
            'Market_Volatility',
            'Equity_Inflows',
            'AI_Technology_Advances',

            # Inner nodes (computed)
            'Monetary_Policy_Interest_Rates',
            'Inflation',
            'Corporate_Earnings',
            'Investor_Sentiment',
            'NASDAQ'
        ]

        self.n_nodes = len(self.nodes)
        self.node_index = {node: i for i, node in enumerate(self.nodes)}

        # Initialize weight matrix (adjacency matrix)
        self.W = np.zeros((self.n_nodes, self.n_nodes))

        # Define connections based on diagram
        self._initialize_weights()

        # Historical data storage
        self.historical_states = []
        self.timestamps = []

    def _initialize_weights(self):
        """Initialize weight matrix based on the diagram connections

        GREEN arrows = positive relationships
        RED arrows = negative relationships
        """

        # Helper function to set weight
        def set_weight(from_node, to_node, weight):
            i = self.node_index[from_node]
            j = self.node_index[to_node]
            self.W[i, j] = weight

        # ===== GREEN ARROWS (Positive relationships) =====

        # To Monetary Policy & Interest Rates
        set_weight('Large_Deficits', 'Monetary_Policy_Interest_Rates', 0.7)
        set_weight('Foreign_Demand', 'Monetary_Policy_Interest_Rates', 0.5)
        set_weight('Low_Unemployment', 'Monetary_Policy_Interest_Rates', 0.6)

        # To Inflation
        set_weight('Monetary_Policy_Interest_Rates', 'Inflation', 0.8)
        set_weight('Low_Unemployment', 'Inflation', 0.6)
        set_weight('Supply_Increase', 'Inflation', 0.6)
        set_weight('Energy_Price_Increase', 'Inflation', 0.7)
        set_weight('Consumer_Demand', 'Inflation', 0.5)

        # To Corporate Earnings
        set_weight('Business_Investment', 'Corporate_Earnings', 0.7)
        set_weight('Inflation', 'Corporate_Earnings', 0.4)
        set_weight('Consumer_Demand', 'Corporate_Earnings', 0.6)

        # To Investor Sentiment
        set_weight('Corporate_Earnings', 'Investor_Sentiment', 0.8)
        set_weight('Inflation', 'Investor_Sentiment', 0.3)
        set_weight('Market_Volatility', 'Investor_Sentiment', 0.3)
        set_weight('Equity_Inflows', 'Investor_Sentiment', 0.7)

        # To NASDAQ
        set_weight('Investor_Sentiment', 'NASDAQ', 0.9)
        set_weight('Corporate_Earnings', 'NASDAQ', 0.8)
        set_weight('Inflation', 'NASDAQ', 0.3)
        set_weight('AI_Technology_Advances', 'NASDAQ', 0.7)

        # ===== RED ARROWS (Negative relationships) =====

        # Monetary Policy -> Corporate Earnings (negative)
        set_weight('Monetary_Policy_Interest_Rates', 'Corporate_Earnings', -0.6)

        # Monetary Policy -> NASDAQ (negative)
        set_weight('Monetary_Policy_Interest_Rates', 'NASDAQ', -0.5)

        # Inflation -> NASDAQ (negative - the red arrow)
        set_weight('Inflation', 'NASDAQ', -0.4)

        # Inflation -> Investor Sentiment (negative - the red arrow)
        set_weight('Inflation', 'Investor_Sentiment', -0.5)

    def sigmoid(self, x):
        """Sigmoid activation function to normalize values between 0 and 1"""
        return 1 / (1 + np.exp(-x))

    def tanh_activation(self, x):
        """Hyperbolic tangent activation function (-1 to 1)"""
        return np.tanh(x)

    def fetch_yahoo_finance_data(self, start_date, end_date):
        """
        Fetch data from Yahoo Finance for outer nodes

        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format

        Returns:
        --------
        pd.DataFrame with normalized values for outer nodes over time
        """

        # Define proxies for each outer node
        proxies = {
            'Large_Deficits': '^IRX',  # 13 Week Treasury Bill (inverse proxy)
            'Foreign_Demand': 'UUP',  # Dollar index (inverse for foreign demand)
            'Low_Unemployment': None,  # Will use FRED data or manual input
            'Supply_Increase': 'XLI',  # Industrial sector ETF
            'Energy_Price_Increase': 'XLE',  # Energy sector ETF
            'Consumer_Demand': 'XLY',  # Consumer discretionary ETF
            'Business_Investment': 'XLI',  # Industrial sector ETF
            'Market_Volatility': '^VIX',  # VIX volatility index
            'Equity_Inflows': 'SPY',  # S&P 500 as proxy for equity inflows
            'AI_Technology_Advances': 'NVDA',  # NVIDIA as proxy for AI/tech advances
        }

        data = {}

        for node, ticker in proxies.items():
            if ticker:
                try:
                    ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    # Check if data was fetched successfully
                    if not ticker_data.empty:
                        # Use adjusted close or close price
                        if 'Adj Close' in ticker_data.columns:
                            price_data = ticker_data['Adj Close']
                        elif 'Close' in ticker_data.columns:
                            price_data = ticker_data['Close']
                        else:
                            print(f"No price data for {ticker} ({node})")
                            data[node] = pd.Series()
                            continue

                        # Handle multi-index columns (squeeze to Series if needed)
                        if isinstance(price_data, pd.DataFrame):
                            price_data = price_data.squeeze()

                        data[node] = price_data
                    else:
                        print(f"No data returned for {ticker} ({node})")
                        data[node] = pd.Series()
                except Exception as e:
                    print(f"Error fetching {ticker} for {node}: {e}")
                    data[node] = pd.Series()
            else:
                # For nodes without ticker (like Low_Unemployment), create empty series
                data[node] = pd.Series()

        # Fetch NASDAQ for comparison
        try:
            nasdaq_data = yf.download('^IXIC', start=start_date, end=end_date, progress=False)
            if not nasdaq_data.empty and 'Adj Close' in nasdaq_data.columns:
                price_data = nasdaq_data['Adj Close']
            elif not nasdaq_data.empty and 'Close' in nasdaq_data.columns:
                price_data = nasdaq_data['Close']
            else:
                print("No NASDAQ data available")
                data['NASDAQ_actual'] = pd.Series()
                price_data = None

            # Handle multi-index columns (squeeze to Series if needed)
            if price_data is not None:
                if isinstance(price_data, pd.DataFrame):
                    price_data = price_data.squeeze()
                data['NASDAQ_actual'] = price_data
        except Exception as e:
            print(f"Error fetching NASDAQ: {e}")
            data['NASDAQ_actual'] = pd.Series()

        # Create DataFrame - this handles empty series gracefully
        df = pd.DataFrame(data)

        # If dataframe is empty, raise an error
        if df.empty:
            raise ValueError("No data was fetched from Yahoo Finance. Check your date range and internet connection.")

        return df

    def normalize_data(self, df):
        """
        Normalize data to [0, 1] range using min-max normalization
        Also handle inverse relationships
        """
        normalized_df = df.copy()

        for col in df.columns:
            if df[col] is not None and not df[col].isna().all():
                min_val = df[col].min()
                max_val = df[col].max()

                if max_val > min_val:
                    normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0.5

                # Inverse for certain indicators
                if col in ['Large_Deficits', 'Market_Volatility']:
                    normalized_df[col] = 1 - normalized_df[col]

        return normalized_df

    def compute_state(self, outer_node_values, activation='sigmoid'):
        """
        Compute FCM state given outer node values

        Parameters:
        -----------
        outer_node_values : dict
            Dictionary mapping outer node names to their values [0, 1]
        activation : str
            Activation function: 'sigmoid' or 'tanh'

        Returns:
        --------
        dict with all node values
        """

        # Initialize state vector
        state = np.zeros(self.n_nodes)

        # Set outer node values
        for node, value in outer_node_values.items():
            if node in self.node_index:
                state[self.node_index[node]] = value

        # Iteratively compute inner node values until convergence
        max_iterations = 50
        threshold = 0.001

        # Indices of inner nodes
        inner_nodes_idx = [
            self.node_index['Monetary_Policy_Interest_Rates'],
            self.node_index['Inflation'],
            self.node_index['Corporate_Earnings'],
            self.node_index['Investor_Sentiment'],
            self.node_index['NASDAQ']
        ]

        for iteration in range(max_iterations):
            prev_state = state.copy()

            # Update inner nodes only
            for idx in inner_nodes_idx:
                # Compute weighted sum of inputs
                weighted_sum = np.dot(state, self.W[:, idx])

                # Apply activation function
                if activation == 'sigmoid':
                    state[idx] = self.sigmoid(weighted_sum)
                else:
                    state[idx] = self.tanh_activation(weighted_sum)

            # Check convergence
            if np.max(np.abs(state - prev_state)) < threshold:
                break

        # Create result dictionary
        result = {node: state[i] for i, node in enumerate(self.nodes)}

        return result

    def simulate_time_series(self, start_date, end_date, save_results=True):
        """
        Simulate FCM over time using Yahoo Finance data

        Parameters:
        -----------
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        save_results : bool
            Whether to save results to CSV

        Returns:
        --------
        pd.DataFrame with FCM predictions over time
        """

        print(f"Fetching Yahoo Finance data from {start_date} to {end_date}...")
        raw_data = self.fetch_yahoo_finance_data(start_date, end_date)

        print("Normalizing data...")
        normalized_data = self.normalize_data(raw_data)

        results = []

        print("Running FCM simulation...")
        for date, row in normalized_data.iterrows():
            # Prepare outer node values
            outer_values = {}
            for node in ['Large_Deficits', 'Foreign_Demand', 'Low_Unemployment',
                        'Supply_Increase', 'Energy_Price_Increase', 'Consumer_Demand',
                        'Business_Investment', 'Market_Volatility', 'Equity_Inflows',
                        'AI_Technology_Advances']:
                if node in row and not pd.isna(row[node]):
                    outer_values[node] = row[node]
                else:
                    outer_values[node] = 0.5  # neutral value

            # Compute FCM state
            state = self.compute_state(outer_values)
            state['Date'] = date

            # Add actual NASDAQ value if available
            if 'NASDAQ_actual' in row and not pd.isna(row['NASDAQ_actual']):
                state['NASDAQ_actual'] = row['NASDAQ_actual']

            results.append(state)

        results_df = pd.DataFrame(results)
        results_df.set_index('Date', inplace=True)

        if save_results:
            results_df.to_csv('fcm_nasdaq_results.csv')
            print("Results saved to fcm_nasdaq_results.csv")

        return results_df

    def visualize_weights(self, save_fig=True):
        """Visualize the weight matrix as a heatmap"""
        plt.figure(figsize=(14, 12))
        sns.heatmap(self.W,
                   xticklabels=self.nodes,
                   yticklabels=self.nodes,
                   cmap='RdYlGn',
                   center=0,
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Weight'})
        plt.title('FCM Weight Matrix - NASDAQ Model')
        plt.xlabel('To Node')
        plt.ylabel('From Node')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_fig:
            plt.savefig('fcm_weight_matrix.png', dpi=300, bbox_inches='tight')
            print("Weight matrix visualization saved to fcm_weight_matrix.png")

        plt.show()

    def plot_results(self, results_df, save_fig=True):
        """Plot FCM simulation results"""

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: NASDAQ prediction vs actual
        ax1 = axes[0]
        ax1.plot(results_df.index, results_df['NASDAQ'], label='FCM Prediction', linewidth=2)
        if 'NASDAQ_actual' in results_df.columns:
            # Normalize actual NASDAQ for comparison
            nasdaq_actual_norm = (results_df['NASDAQ_actual'] - results_df['NASDAQ_actual'].min()) / \
                                (results_df['NASDAQ_actual'].max() - results_df['NASDAQ_actual'].min())
            ax1.plot(results_df.index, nasdaq_actual_norm, label='Actual (Normalized)',
                    linewidth=2, alpha=0.7, linestyle='--')
        ax1.set_ylabel('NASDAQ Value')
        ax1.set_title('NASDAQ: FCM Prediction vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Inner nodes
        ax2 = axes[1]
        inner_nodes = ['Monetary_Policy_Interest_Rates', 'Inflation',
                      'Corporate_Earnings', 'Investor_Sentiment']
        for node in inner_nodes:
            ax2.plot(results_df.index, results_df[node], label=node.replace('_', ' '), linewidth=1.5)
        ax2.set_ylabel('Node Value')
        ax2.set_title('Inner Nodes (Computed)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Selected outer nodes
        ax3 = axes[2]
        outer_nodes = ['Energy_Price_Increase', 'Market_Volatility',
                      'Consumer_Demand', 'Equity_Inflows']
        for node in outer_nodes:
            if node in results_df.columns:
                ax3.plot(results_df.index, results_df[node], label=node.replace('_', ' '), linewidth=1.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Node Value')
        ax3.set_title('Selected Outer Nodes (from Yahoo Finance)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            plt.savefig('fcm_simulation_results.png', dpi=300, bbox_inches='tight')
            print("Simulation results plot saved to fcm_simulation_results.png")

        plt.show()


if __name__ == "__main__":
    # Create FCM instance
    fcm = NASDAQ_FCM()

    # Visualize weight matrix
    print("Generating weight matrix visualization...")
    fcm.visualize_weights()

    # Run simulation for past year
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"\nRunning FCM simulation from {start_date} to {end_date}...")
    results = fcm.simulate_time_series(start_date, end_date)

    # Plot results
    print("\nGenerating results plots...")
    fcm.plot_results(results)

    print("\nDone! Check the generated CSV and PNG files for results.")
