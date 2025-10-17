import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# 20 popular tech stocks + S&P 500
tickers = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Google',
    'AMZN': 'Amazon',
    'NVDA': 'NVIDIA',
    'META': 'Meta',
    'TSLA': 'Tesla',
    'AVGO': 'Broadcom',
    'ORCL': 'Oracle',
    'ADBE': 'Adobe',
    'CRM': 'Salesforce',
    'CSCO': 'Cisco',
    'INTC': 'Intel',
    'AMD': 'AMD',
    'NFLX': 'Netflix',
    'QCOM': 'Qualcomm',
    'IBM': 'IBM',
    'UBER': 'Uber',
    'SHOP': 'Shopify',
    'SNOW': 'Snowflake',
    '^GSPC': 'SP500'
}

print("Downloading stock data...")
stock_data = {}
for ticker, name in tickers.items():
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty:
            stock_data[name] = data
            print(f"  ✓ {name} ({ticker})")
        else:
            print(f"  ✗ {name} ({ticker}) - No data")
    except Exception as e:
        print(f"  ✗ {name} ({ticker}) - Error: {e}")

print(f"\nSuccessfully downloaded {len(stock_data)} stocks\n")

# Calculate direction for each stock
comparison_data = {}
for name, data in stock_data.items():
    data['Change'] = data['Close'].diff()
    data['Direction'] = data['Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    comparison_data[name] = data['Direction']

comparison = pd.DataFrame(comparison_data)
comparison = comparison.dropna()

stocks = list(comparison.columns)
n = len(stocks)

matrix = np.zeros((n, n))

for i, stock1 in enumerate(stocks):
    for j, stock2 in enumerate(stocks):
        if i == j:
            matrix[i][j] = 0.0
        else:
            matching = (comparison[stock1] == comparison[stock2]).sum()
            total = len(comparison)
            match_percentage = (matching / total) * 100
            matrix[i][j] = (match_percentage - 50) / 50


print("=== Stock Direction Correlation Matrix ===")
print(f"Total trading days analyzed: {len(comparison)}")
print(f"Formula: (match_percentage - 50) / 50\n")

matrix_df = pd.DataFrame(matrix, index=stocks, columns=stocks)

# Print matrix with better formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)

print(matrix_df.to_string())

# Print top correlations
print("\n=== Top 10 Highest Correlations ===")
correlations = []
for i, stock1 in enumerate(stocks):
    for j, stock2 in enumerate(stocks):
        if i < j:
            matching = (comparison[stock1] == comparison[stock2]).sum()
            match_percentage = (matching / len(comparison)) * 100
            correlation = matrix[i][j]
            correlations.append((stock1, stock2, match_percentage, correlation))

correlations.sort(key=lambda x: x[3], reverse=True)
for stock1, stock2, match_pct, corr in correlations[:10]:
    print(f"{stock1:12} vs {stock2:12}: {match_pct:5.2f}% matching (corr: {corr:6.4f})")

print("\n=== Bottom 10 Lowest Correlations ===")
for stock1, stock2, match_pct, corr in correlations[-10:]:
    print(f"{stock1:12} vs {stock2:12}: {match_pct:5.2f}% matching (corr: {corr:6.4f})")

# Create Fuzzy Cognitive Map visualization
print("\nGenerating Fuzzy Cognitive Map...")

# Create directed graph
G = nx.DiGraph()

# Add all stocks as nodes
for stock in stocks:
    G.add_node(stock)

# Add edges with weights (only show significant correlations)
threshold = 0.3  # Only show correlations above this threshold
edge_weights = []
for i, stock1 in enumerate(stocks):
    for j, stock2 in enumerate(stocks):
        if i != j:
            weight = matrix[i][j]
            if abs(weight) >= threshold:
                G.add_edge(stock1, stock2, weight=weight)
                edge_weights.append(weight)

# Calculate mean correlation for color scaling
if edge_weights:
    mean_correlation = np.mean(edge_weights)
    min_correlation = min(edge_weights)
    max_correlation = max(edge_weights)
    print(f"  Correlation range: [{min_correlation:.4f}, {max_correlation:.4f}]")
    print(f"  Mean correlation: {mean_correlation:.4f}")
else:
    mean_correlation = 0
    min_correlation = -1
    max_correlation = 1

# Create figure
fig, ax = plt.subplots(figsize=(20, 20))

# Use circular layout for better visibility
pos = nx.circular_layout(G)

# Draw nodes
node_sizes = [3000 for _ in G.nodes()]
nx.draw_networkx_nodes(G, pos,
                       node_color='lightblue',
                       node_size=node_sizes,
                       alpha=0.9,
                       ax=ax)

# Draw node labels
nx.draw_networkx_labels(G, pos,
                        font_size=10,
                        font_weight='bold',
                        ax=ax)

# Function to map correlation to color (mean = yellow, above = green, below = red)
def correlation_to_color(weight, mean_corr, min_corr, max_corr):
    """Map correlation weight to RGB color with mean as midpoint"""
    if weight > mean_corr:
        # Above mean: yellow to green
        intensity = (weight - mean_corr) / (max_corr - mean_corr) if max_corr > mean_corr else 0
        # RGB: (1-intensity, 1, 0) gives yellow to green
        return (1 - intensity, 1, 0)
    else:
        # Below mean: yellow to red
        intensity = (mean_corr - weight) / (mean_corr - min_corr) if mean_corr > min_corr else 0
        # RGB: (1, 1-intensity, 0) gives yellow to red
        return (1, 1 - intensity, 0)

# Draw all edges with gradient colors
for (u, v, weight) in G.edges(data='weight'):
    color = correlation_to_color(weight, mean_correlation, min_correlation, max_correlation)
    width = abs(weight) * 3
    nx.draw_networkx_edges(G, pos,
                           edgelist=[(u, v)],
                           width=width,
                           edge_color=[color],
                           alpha=0.7,
                           arrows=True,
                           arrowsize=15,
                           connectionstyle='arc3,rad=0.1',
                           ax=ax)

# Add title and legend
plt.title(f'Fuzzy Cognitive Map - Tech Stock Direction Correlations\n' +
          f'Green = Above Mean | Yellow = Mean ({mean_correlation:.4f}) | Red = Below Mean\n' +
          f'Threshold: |{threshold}| or higher | Edge thickness = Correlation strength',
          fontsize=16, fontweight='bold', pad=20)

# Add custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', linewidth=3, label=f'Above Mean Correlation (> {mean_correlation:.4f})'),
    Line2D([0], [0], color='yellow', linewidth=3, label=f'Mean Correlation (≈ {mean_correlation:.4f})'),
    Line2D([0], [0], color='red', linewidth=3, label=f'Below Mean Correlation (< {mean_correlation:.4f})'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

plt.axis('off')
plt.tight_layout()

# Save the figure
output_file = 'stock_fcm.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Fuzzy Cognitive Map saved to: {output_file}")
print(f"  Total nodes: {G.number_of_nodes()}")
print(f"  Total edges (|correlation| >= {threshold}): {G.number_of_edges()}")

# Show the plot
plt.show()

print("\nAnalysis complete!")
