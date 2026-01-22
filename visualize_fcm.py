import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.set_aspect('equal')
ax.axis('off')

outer_nodes = [
    ('Large_Deficits', '^IRX', '13W Treasury Bill', 1.5, 12),
    ('Foreign_Demand', 'UUP', 'US Dollar Index', 1.5, 10),
    ('Low_Unemployment', 'N/A', 'No proxy (uses 0.5)', 1.5, 8),
    ('Supply_Increase', 'XLI', 'Industrial ETF', 1.5, 6),
    ('Energy_Price_Increase', 'XLE', 'Energy ETF', 1.5, 4),
    ('Consumer_Demand', 'XLY', 'Consumer Discr. ETF', 6, 1),
    ('Business_Investment', 'XLI', 'Industrial ETF (DUPLICATE)', 9, 1),
    ('Market_Volatility', '^VIX', 'Volatility Index', 12, 1),
    ('Equity_Inflows', 'SPY', 'S&P 500 ETF', 15, 1),
    ('AI_Technology', 'NVDA', 'NVIDIA', 18, 4),
]

inner_nodes = [
    ('Monetary_Policy', 8, 10),
    ('Inflation', 8, 7),
    ('Corporate_Earnings', 12, 10),
    ('Investor_Sentiment', 12, 7),
    ('NASDAQ', 15, 8.5),
]

def draw_outer_node(ax, name, ticker, desc, x, y):
    box = mpatches.FancyBboxPatch((x-1.4, y-0.6), 2.8, 1.2,
                                   boxstyle="round,pad=0.05",
                                   facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y+0.2, name.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x, y-0.35, f'{ticker}', ha='center', va='center', fontsize=7, color='darkblue')

def draw_inner_node(ax, name, x, y):
    circle = plt.Circle((x, y), 0.9, facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, name.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')

def draw_target_node(ax, name, x, y):
    circle = plt.Circle((x, y), 1.1, facecolor='lightgreen', edgecolor='darkgreen', linewidth=3)
    ax.add_patch(circle)
    ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')

for name, ticker, desc, x, y in outer_nodes:
    draw_outer_node(ax, name, ticker, desc, x, y)

for name, x, y in inner_nodes[:-1]:
    draw_inner_node(ax, name, x, y)

draw_target_node(ax, 'NASDAQ', 15, 8.5)

connections = [
    # To Monetary Policy (green = positive)
    (1.5, 12, 8, 10, 0.7, 'green'),      # Large_Deficits
    (1.5, 10, 8, 10, 0.5, 'green'),      # Foreign_Demand
    (1.5, 8, 8, 10, 0.6, 'green'),       # Low_Unemployment

    # To Inflation
    (8, 10, 8, 7, 0.8, 'green'),         # Monetary_Policy
    (1.5, 8, 8, 7, 0.6, 'green'),        # Low_Unemployment
    (1.5, 6, 8, 7, 0.6, 'green'),        # Supply_Increase
    (1.5, 4, 8, 7, 0.7, 'green'),        # Energy_Price_Increase
    (6, 1, 8, 7, 0.5, 'green'),          # Consumer_Demand

    # To Corporate Earnings
    (9, 1, 12, 10, 0.7, 'green'),        # Business_Investment
    (8, 7, 12, 10, 0.4, 'green'),        # Inflation
    (6, 1, 12, 10, 0.6, 'green'),        # Consumer_Demand
    (8, 10, 12, 10, -0.6, 'red'),        # Monetary_Policy (negative)

    # To Investor Sentiment
    (12, 10, 12, 7, 0.8, 'green'),       # Corporate_Earnings
    (8, 7, 12, 7, -0.5, 'red'),          # Inflation (negative)
    (12, 1, 12, 7, 0.3, 'green'),        # Market_Volatility
    (15, 1, 12, 7, 0.7, 'green'),        # Equity_Inflows

    # To NASDAQ
    (12, 7, 15, 8.5, 0.9, 'green'),      # Investor_Sentiment
    (12, 10, 15, 8.5, 0.8, 'green'),     # Corporate_Earnings
    (8, 7, 15, 8.5, -0.4, 'red'),        # Inflation (negative)
    (8, 10, 15, 8.5, -0.5, 'red'),       # Monetary_Policy (negative)
    (18, 4, 15, 8.5, 0.7, 'green'),      # AI_Technology
]

for x1, y1, x2, y2, weight, color in connections:
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)

    offset1 = 1.0 if x1 < 5 else 0.9
    offset2 = 1.1 if (x2, y2) == (15, 8.5) else 0.9

    start_x = x1 + (dx/dist) * offset1
    start_y = y1 + (dy/dist) * offset1
    end_x = x2 - (dx/dist) * offset2
    end_y = y2 - (dy/dist) * offset2

    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', color=color, lw=abs(weight)*2.5, alpha=0.7))

    mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
    ax.text(mid_x, mid_y, f'{weight}', fontsize=7, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.8))

legend_elements = [
    mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Outer Nodes (Yahoo Finance proxies)'),
    mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Inner Nodes (computed)'),
    mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Target Node (NASDAQ)'),
    plt.Line2D([0], [0], color='green', linewidth=3, label='Positive weight'),
    plt.Line2D([0], [0], color='red', linewidth=3, label='Negative weight'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.text(10, 13.5, 'FCM Node Structure & Proxy Assignments', ha='center', fontsize=16, fontweight='bold')
ax.text(10, 13, 'Blue boxes = Input nodes with Yahoo Finance ticker proxies', ha='center', fontsize=10)

proxy_text = """
PROXY ISSUES:
• XLI used for BOTH Supply_Increase AND Business_Investment
• Most proxies are stock prices, not economic indicators
• Low_Unemployment has NO proxy (defaults to 0.5)
• All data is same-day (no prediction lag)
"""
ax.text(0.5, 2.5, proxy_text, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('fcm_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved to fcm_structure.png")
