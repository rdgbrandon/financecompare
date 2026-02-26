#!/usr/bin/env python3
"""
Animated FCM Structure Visualization (1x2 format - DYNAMIC ONLY)
- Panel 1: Dynamic Yahoo FCM with real node values (animated)
- Panel 2: Dynamic Google FCM with real node values (animated)

Only shows DYNAMIC data - nodes display actual values and change size/color based on data
Uses real unemployment data, Yahoo News sentiment, and Google News sentiment
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import copy
from matplotlib.animation import FuncAnimation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fcm_animated_visual")

# Import the main analysis engine
from fcm_nasdaq import prepare_data_and_run

# ============================================================================
# NODE CONFIGURATION - Matches economic FCM structure
# ============================================================================

outer_nodes = [
    ('Large_Deficits', '^IRX', '13W Treasury Bill', 1.5, 12),
    ('Foreign_Demand', 'UUP', 'US Dollar Index', 1.5, 10),
    ('Low_Unemployment', 'FRED', 'Unemployment Rate', 1.5, 8),
    ('Supply_Increase', 'XLI', 'Industrial ETF', 1.5, 6),
    ('Energy_Price_Increase', 'XLE', 'Energy ETF', 1.5, 4),
    ('Consumer_Demand', 'XLY', 'Consumer Discr. ETF', 6, 1),
    ('Business_Investment', 'XLI', 'Industrial ETF', 9, 1),
    ('Market_Volatility', '^VIX', 'Volatility Index', 12, 1),
    ('Equity_Inflows', 'SPY', 'S&P 500 ETF', 15, 1),
    ('AI_Technology', 'NVDA', 'NVIDIA', 18, 4),
    ('News_Sentiment', 'Dynamic', 'News Sentiment', 18, 7),
]

inner_nodes = [
    ('Monetary_Policy', 8, 10),
    ('Inflation', 8, 7),
    ('Corporate_Earnings', 12, 10),
    ('Investor_Sentiment', 12, 7),
    ('NASDAQ', 15, 8.5),
]

# Canonical node order
node_order = [n[0] for n in outer_nodes] + [n[0] for n in inner_nodes]

# Position mapping
pos_to_name = {}
for name, _, _, x, y in outer_nodes:
    pos_to_name[(x, y)] = name
for name, x, y in inner_nodes:
    pos_to_name[(x, y)] = name

# ============================================================================
# VISUALIZATION HELPERS - DYNAMIC NODE DRAWING
# ============================================================================

def get_node_color_and_size(value, is_outer=True):
    """
    Get color and size for a node based on its normalized value (0-1).
    - Size: ranges from 0.5 to 1.5 times base radius
    - Color intensity: ranges from light to saturated
    """
    if value is None or np.isnan(value):
        value = 0.5

    # Clip value to [0, 1]
    value = float(np.clip(value, 0, 1))

    # Size scaling: smaller for low values, larger for high values
    base_radius = 1.0 if is_outer else 0.8
    size_multiplier = 0.6 + (0.9 * value)  # ranges 0.6 to 1.5
    radius = base_radius * size_multiplier

    # Color intensity: use value to set alpha and color saturation
    if is_outer:
        # Outer nodes: blue gradient (light to dark)
        intensity = 0.3 + (0.7 * value)  # ranges 0.3 to 1.0
        color = (intensity * 0.3, intensity * 0.5, intensity)  # blue gradient
    else:
        # Inner nodes: yellow gradient (light to orange)
        intensity = 0.3 + (0.7 * value)
        color = (intensity, intensity * 0.7, 0)  # yellow to orange gradient

    return color, radius

def draw_outer_node(ax, name, ticker, desc, x, y, value=None):
    """Draw outer input node (dynamic box with real value)"""
    color, radius = get_node_color_and_size(value, is_outer=True)

    # Dynamic box size based on value
    box_width = 2.4 + (0.8 * (value if value is not None else 0.5))
    box_height = 1.0 + (0.4 * (value if value is not None else 0.5))

    box = mpatches.FancyBboxPatch(
        (x - box_width/2, y - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='black',
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)

    # Display name
    ax.text(x, y+0.3, name.replace('_', '\n'), ha='center', va='center',
            fontsize=7, fontweight='bold')

    # Display actual value (0-1 normalized)
    if value is not None:
        ax.text(x, y-0.15, f'{value:.2f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white' if value > 0.6 else 'black')

    # Display ticker
    ax.text(x, y-0.55, f'{ticker}', ha='center', va='center', fontsize=6, color='darkblue')

def draw_inner_node(ax, name, x, y, value=None):
    """Draw inner computed node (dynamic circle with real value)"""
    color, radius = get_node_color_and_size(value, is_outer=False)

    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.8)
    ax.add_patch(circle)

    # Display name
    ax.text(x, y+0.15, name.replace('_', '\n'), ha='center', va='center',
            fontsize=7, fontweight='bold')

    # Display actual value (0-1 normalized)
    if value is not None:
        ax.text(x, y-0.25, f'{value:.2f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white' if value > 0.6 else 'black')

def draw_target_node(ax, name, x, y, value=None):
    """Draw target node - NASDAQ (dynamic circle with real value)"""
    # NASDAQ with gradient green
    if value is not None:
        value = np.clip(value, 0, 1)
        intensity = 0.4 + (0.6 * value)
        color = (0, intensity * 0.7, 0)  # green gradient
    else:
        color = (0, 0.7, 0)

    radius = 1.1 * (0.8 + (0.6 * (value if value is not None else 0.5)))

    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor='darkgreen',
                        linewidth=3, alpha=0.9)
    ax.add_patch(circle)

    # Display name
    ax.text(x, y+0.2, name, ha='center', va='center', fontsize=11, fontweight='bold')

    # Display actual value
    if value is not None:
        ax.text(x, y-0.4, f'{value:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white' if value > 0.6 else 'black')

def draw_nodes_layout(ax, node_values_dict=None):
    """
    Draw all nodes on axis with dynamic values.
    node_values_dict: dict mapping node names to their normalized values (0-1)
    """
    if node_values_dict is None:
        node_values_dict = {}

    for name, ticker, desc, x, y in outer_nodes:
        value = node_values_dict.get(name)
        draw_outer_node(ax, name, ticker, desc, x, y, value)

    for name, x, y in inner_nodes[:-1]:
        value = node_values_dict.get(name)
        draw_inner_node(ax, name, x, y, value)

    # NASDAQ (target)
    value = node_values_dict.get('NASDAQ')
    draw_target_node(ax, 'NASDAQ', 15, 8.5, value)

def draw_dynamic_arrows(ax, weights_dict, node_values_dict=None, annotate_weights=True, title_text=""):
    """Draw arrows representing weights on axis with dynamic nodes"""
    ax.cla()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw nodes with actual values
    draw_nodes_layout(ax, node_values_dict)

    # Define all connections
    connections = [
        (1.5, 12, 8, 10, 0.0, 'green'),  # Large_Deficits → Monetary_Policy
        (1.5, 10, 8, 10, 0.0, 'green'),  # Foreign_Demand → Monetary_Policy
        (1.5, 8, 8, 10, 0.0, 'green'),   # Low_Unemployment → Monetary_Policy
        (8, 10, 8, 7, 0.0, 'green'),     # Monetary_Policy → Inflation
        (1.5, 8, 8, 7, 0.0, 'green'),    # Low_Unemployment → Inflation
        (1.5, 6, 8, 7, 0.0, 'green'),    # Supply_Increase → Inflation
        (1.5, 4, 8, 7, 0.0, 'green'),    # Energy_Price_Increase → Inflation
        (6, 1, 8, 7, 0.0, 'green'),      # Consumer_Demand → Inflation
        (9, 1, 12, 10, 0.0, 'green'),    # Business_Investment → Corporate_Earnings
        (8, 7, 12, 10, 0.0, 'green'),    # Inflation → Corporate_Earnings
        (6, 1, 12, 10, 0.0, 'green'),    # Consumer_Demand → Corporate_Earnings
        (8, 10, 12, 10, 0.0, 'red'),     # Monetary_Policy → Corporate_Earnings
        (12, 10, 12, 7, 0.0, 'green'),   # Corporate_Earnings → Investor_Sentiment
        (8, 7, 12, 7, 0.0, 'red'),       # Inflation → Investor_Sentiment
        (12, 1, 12, 7, 0.0, 'green'),    # Market_Volatility → Investor_Sentiment
        (15, 1, 12, 7, 0.0, 'green'),    # Equity_Inflows → Investor_Sentiment
        (18, 7, 12, 7, 0.0, 'green'),    # News_Sentiment → Investor_Sentiment
        (12, 7, 15, 8.5, 0.0, 'green'),  # Investor_Sentiment → NASDAQ
        (12, 10, 15, 8.5, 0.0, 'green'), # Corporate_Earnings → NASDAQ
        (8, 7, 15, 8.5, 0.0, 'red'),     # Inflation → NASDAQ
        (8, 10, 15, 8.5, 0.0, 'red'),    # Monetary_Policy → NASDAQ
        (18, 4, 15, 8.5, 0.0, 'green'),  # AI_Technology → NASDAQ
        (18, 7, 15, 8.5, 0.0, 'green'),  # News_Sentiment → NASDAQ
    ]

    # Draw arrows with weight annotations
    for x1, y1, x2, y2, orig_w, _ in connections:
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)

        offset1 = 1.0 if x1 < 5 else 0.9
        offset2 = 1.1 if (x2, y2) == (15, 8.5) else 0.9

        start_x = x1 + (dx/dist) * offset1
        start_y = y1 + (dy/dist) * offset1
        end_x = x2 - (dx/dist) * offset2
        end_y = y2 - (dy/dist) * offset2

        src = pos_to_name.get((x1, y1))
        tgt = pos_to_name.get((x2, y2))

        # Get weight from dict
        w = orig_w
        if src and tgt and src in weights_dict and tgt in weights_dict[src]:
            w = weights_dict[src][tgt]

        color = 'green' if w >= 0 else 'red'
        line_width = max(0.5, abs(w) * 3.0)

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=line_width, alpha=0.85))

        if annotate_weights:
            mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            ax.text(mid_x, mid_y, f'{w:.3f}', fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.9))

    if title_text:
        ax.text(10, 13.2, title_text, ha='center', fontsize=11, fontweight='bold')

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def create_animated_fcm_visualization():
    """Create and display animated 1x2 FCM visualization (DYNAMIC ONLY)"""

    logger.info("Loading FCM analysis data...")
    out = prepare_data_and_run()

    # Extract data
    weights_history_yahoo = out.get("weights_history_yahoo", [])
    weights_history_google = out.get("weights_history_google", [])
    date_history_yahoo = out.get("date_history_yahoo", [])
    date_history_google = out.get("date_history_google", [])
    inputs_norm_yahoo = out.get("inputs_norm_yahoo", None)
    inputs_norm_google = out.get("inputs_norm_google", None)

    if not weights_history_yahoo or not weights_history_google:
        logger.error("No weight histories available!")
        return

    if inputs_norm_yahoo is None or inputs_norm_google is None:
        logger.error("No normalized inputs available!")
        return

    logger.info(f"Yahoo snapshots: {len(weights_history_yahoo)}")
    logger.info(f"Google snapshots: {len(weights_history_google)}")

    # Create figure with 1x2 layout (DYNAMIC ONLY - no static reference)
    fig, axs = plt.subplots(1, 2, figsize=(36, 14))
    ax_yahoo, ax_google = axs

    for ax in axs:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 14)
        ax.set_aspect('equal')
        ax.axis('off')

    # Initialize dynamic panels with first snapshot
    logger.info("Initializing dynamic panels...")
    initial_yahoo_weights = weights_history_yahoo[0] if weights_history_yahoo else {}
    initial_google_weights = weights_history_google[0] if weights_history_google else {}

    # Get initial node values from first row of normalized inputs
    initial_yahoo_values = inputs_norm_yahoo.iloc[0].to_dict() if len(inputs_norm_yahoo) > 0 else {}
    initial_google_values = inputs_norm_google.iloc[0].to_dict() if len(inputs_norm_google) > 0 else {}

    draw_dynamic_arrows(ax_yahoo, initial_yahoo_weights, initial_yahoo_values, annotate_weights=True,
                       title_text="Dynamic Yahoo FCM\n(Weights Learning Over Time)")
    draw_dynamic_arrows(ax_google, initial_google_weights, initial_google_values, annotate_weights=True,
                       title_text="Dynamic Google FCM\n(Weights Learning Over Time)")

    # Animation function
    frames = min(len(weights_history_yahoo), len(weights_history_google), len(inputs_norm_yahoo), len(inputs_norm_google))

    def update_frame(frame_idx):
        """Update animation frame with both weights and node values"""
        if frame_idx >= len(weights_history_yahoo) or frame_idx >= len(weights_history_google):
            return
        if frame_idx >= len(inputs_norm_yahoo) or frame_idx >= len(inputs_norm_google):
            return

        # Update Yahoo panel
        yahoo_weights = weights_history_yahoo[frame_idx]
        yahoo_date = date_history_yahoo[frame_idx] if frame_idx < len(date_history_yahoo) else None
        yahoo_values = inputs_norm_yahoo.iloc[frame_idx].to_dict()
        yahoo_title = f"Dynamic Yahoo FCM (Weights Learning)\nDate: {yahoo_date.strftime('%Y-%m-%d') if yahoo_date else 'N/A'}"
        draw_dynamic_arrows(ax_yahoo, yahoo_weights, yahoo_values, annotate_weights=True, title_text=yahoo_title)

        # Update Google panel
        google_weights = weights_history_google[frame_idx]
        google_date = date_history_google[frame_idx] if frame_idx < len(date_history_google) else None
        google_values = inputs_norm_google.iloc[frame_idx].to_dict()
        google_title = f"Dynamic Google FCM (Weights Learning)\nDate: {google_date.strftime('%Y-%m-%d') if google_date else 'N/A'}"
        draw_dynamic_arrows(ax_google, google_weights, google_values, annotate_weights=True, title_text=google_title)

    if frames > 0:
        logger.info(f"Creating animation with {frames} frames...")
        ani = FuncAnimation(fig, update_frame, frames=frames, interval=300, repeat=True)
    else:
        logger.warning("No frames to animate!")

    # Overall title
    fig.suptitle('Fuzzy Cognitive Map: Dynamic Evolution with Node Values (Yahoo vs Google Data Sources)',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('fcm_structure_animated.png', dpi=150, bbox_inches='tight', facecolor='white')
    logger.info("Saved static frame to fcm_structure_animated.png")

    plt.show()

    logger.info("Animation complete!")

    # Print final metrics
    logger.info("\n" + "="*80)
    logger.info("FINAL FCM WEIGHTS - YAHOO")
    logger.info("="*80)
    if weights_history_yahoo:
        final_yahoo = weights_history_yahoo[-1]
        for src in node_order[:11]:  # Only outer nodes
            if src in final_yahoo and 'NASDAQ' in final_yahoo[src]:
                w = final_yahoo[src]['NASDAQ']
                logger.info(f"{src:30s} → NASDAQ: {w:8.4f}")

    logger.info("\n" + "="*80)
    logger.info("FINAL FCM WEIGHTS - GOOGLE")
    logger.info("="*80)
    if weights_history_google:
        final_google = weights_history_google[-1]
        for src in node_order[:11]:  # Only outer nodes
            if src in final_google and 'NASDAQ' in final_google[src]:
                w = final_google[src]['NASDAQ']
                logger.info(f"{src:30s} → NASDAQ: {w:8.4f}")

if __name__ == "__main__":
    create_animated_fcm_visualization()

