import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class NASDAQ_FCM:
    def __init__(self):
        self.nodes = [
            'Large_Deficits', 'Foreign_Demand', 'Low_Unemployment', 'Supply_Increase',
            'Energy_Price_Increase', 'Consumer_Demand', 'Business_Investment',
            'Market_Volatility', 'Equity_Inflows', 'AI_Technology_Advances',
            'Monetary_Policy_Interest_Rates', 'Inflation', 'Corporate_Earnings',
            'Investor_Sentiment', 'NASDAQ'
        ]
        self.n_nodes = len(self.nodes)
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.W = np.zeros((self.n_nodes, self.n_nodes))
        self._initialize_weights()

    def _initialize_weights(self):
        def set_weight(from_node, to_node, weight):
            self.W[self.node_index[from_node], self.node_index[to_node]] = weight

        # Positive relationships
        set_weight('Large_Deficits', 'Monetary_Policy_Interest_Rates', 0.7)
        set_weight('Foreign_Demand', 'Monetary_Policy_Interest_Rates', 0.5)
        set_weight('Low_Unemployment', 'Monetary_Policy_Interest_Rates', 0.6)
        set_weight('Monetary_Policy_Interest_Rates', 'Inflation', 0.8)
        set_weight('Low_Unemployment', 'Inflation', 0.6)
        set_weight('Supply_Increase', 'Inflation', 0.6)
        set_weight('Energy_Price_Increase', 'Inflation', 0.7)
        set_weight('Consumer_Demand', 'Inflation', 0.5)
        set_weight('Business_Investment', 'Corporate_Earnings', 0.7)
        set_weight('Inflation', 'Corporate_Earnings', 0.4)
        set_weight('Consumer_Demand', 'Corporate_Earnings', 0.6)
        set_weight('Corporate_Earnings', 'Investor_Sentiment', 0.8)
        set_weight('Inflation', 'Investor_Sentiment', 0.3)
        set_weight('Market_Volatility', 'Investor_Sentiment', 0.3)
        set_weight('Equity_Inflows', 'Investor_Sentiment', 0.7)
        set_weight('Investor_Sentiment', 'NASDAQ', 0.9)
        set_weight('Corporate_Earnings', 'NASDAQ', 0.8)
        set_weight('Inflation', 'NASDAQ', 0.3)
        set_weight('AI_Technology_Advances', 'NASDAQ', 0.7)

        # Negative relationships
        set_weight('Monetary_Policy_Interest_Rates', 'Corporate_Earnings', -0.6)
        set_weight('Monetary_Policy_Interest_Rates', 'NASDAQ', -0.5)
        set_weight('Inflation', 'NASDAQ', -0.4)
        set_weight('Inflation', 'Investor_Sentiment', -0.5)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fetch_data(self, start_date, end_date):
        proxies = {
            'Large_Deficits': '^IRX',
            'Foreign_Demand': 'UUP',
            'Supply_Increase': 'XLI',
            'Energy_Price_Increase': 'XLE',
            'Consumer_Demand': 'XLY',
            'Business_Investment': 'XLI',
            'Market_Volatility': '^VIX',
            'Equity_Inflows': 'SPY',
            'AI_Technology_Advances': 'NVDA',
        }

        data = {}
        for node, ticker in proxies.items():
            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not ticker_data.empty:
                    price_data = ticker_data['Adj Close'] if 'Adj Close' in ticker_data.columns else ticker_data['Close']
                    if isinstance(price_data, pd.DataFrame):
                        price_data = price_data.squeeze()
                    data[node] = price_data
            except:
                data[node] = pd.Series()

        nasdaq_data = yf.download('^IXIC', start=start_date, end=end_date, progress=False)
        if not nasdaq_data.empty:
            price_data = nasdaq_data['Adj Close'] if 'Adj Close' in nasdaq_data.columns else nasdaq_data['Close']
            if isinstance(price_data, pd.DataFrame):
                price_data = price_data.squeeze()
            data['NASDAQ_actual'] = price_data

        return pd.DataFrame(data)

    def normalize_data(self, df):
        normalized_df = df.copy()
        for col in df.columns:
            if not df[col].isna().all():
                min_val, max_val = df[col].min(), df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0.5
                if col in ['Large_Deficits', 'Market_Volatility']:
                    normalized_df[col] = 1 - normalized_df[col]
        return normalized_df

    def compute_state(self, outer_node_values):
        state = np.zeros(self.n_nodes)
        for node, value in outer_node_values.items():
            if node in self.node_index:
                state[self.node_index[node]] = value

        inner_nodes_idx = [self.node_index[n] for n in
            ['Monetary_Policy_Interest_Rates', 'Inflation', 'Corporate_Earnings', 'Investor_Sentiment', 'NASDAQ']]

        for _ in range(50):
            prev_state = state.copy()
            for idx in inner_nodes_idx:
                state[idx] = self.sigmoid(np.dot(state, self.W[:, idx]))
            if np.max(np.abs(state - prev_state)) < 0.001:
                break

        return {node: state[i] for i, node in enumerate(self.nodes)}

    def simulate(self, start_date, end_date):
        raw_data = self.fetch_data(start_date, end_date)
        normalized_data = self.normalize_data(raw_data)

        outer_nodes = ['Large_Deficits', 'Foreign_Demand', 'Low_Unemployment', 'Supply_Increase',
                       'Energy_Price_Increase', 'Consumer_Demand', 'Business_Investment',
                       'Market_Volatility', 'Equity_Inflows', 'AI_Technology_Advances']

        results = []
        for date, row in normalized_data.iterrows():
            outer_values = {node: row[node] if node in row and not pd.isna(row[node]) else 0.5 for node in outer_nodes}
            state = self.compute_state(outer_values)
            state['Date'] = date
            if 'NASDAQ_actual' in row and not pd.isna(row['NASDAQ_actual']):
                state['NASDAQ_actual'] = row['NASDAQ_actual']
            results.append(state)

        results_df = pd.DataFrame(results)
        results_df.set_index('Date', inplace=True)
        return results_df

    def visualize_weights(self):
        plt.figure(figsize=(14, 12))
        sns.heatmap(self.W, xticklabels=self.nodes, yticklabels=self.nodes,
                    cmap='RdYlGn', center=0, annot=True, fmt='.2f')
        plt.title('FCM Weight Matrix')
        plt.xlabel('To Node')
        plt.ylabel('From Node')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('fcm_weight_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_results(self, results_df):
        fig, ax = plt.subplots(figsize=(14, 6))

        nasdaq_pred_change = results_df['NASDAQ'].diff()
        ax.plot(results_df.index, nasdaq_pred_change, label='FCM Predicted Change', linewidth=2)

        if 'NASDAQ_actual' in results_df.columns:
            nasdaq_actual_norm = (results_df['NASDAQ_actual'] - results_df['NASDAQ_actual'].min()) / \
                                 (results_df['NASDAQ_actual'].max() - results_df['NASDAQ_actual'].min())
            nasdaq_actual_change = nasdaq_actual_norm.diff()
            ax.plot(results_df.index, nasdaq_actual_change, label='Actual Change (Normalized)',
                    linewidth=2, alpha=0.7, linestyle='--')

        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel('Change')
        ax.set_xlabel('Date')
        ax.set_title('NASDAQ: Actual vs Predicted Change')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fcm_change_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    fcm = NASDAQ_FCM()

    fcm.visualize_weights()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"Running simulation from {start_date} to {end_date}...")
    results = fcm.simulate(start_date, end_date)

    fcm.plot_results(results)
