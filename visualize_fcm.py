#!/usr/bin/env python3
"""
visualize_fcm.py

Comprehensive visualization suite for the dynamic FCM and sentiment analysis.
Visualizes:
 - Sentiment time series (Yahoo vs Google vs unified)
 - Model predictions vs actual normalized NASDAQ
 - Daily returns colored by sentiment
 - FCM network with dynamic weights and activations
 - Performance metrics (correlation, MSE) for both sources
 - Weight evolution over time
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from datetime import datetime
import networkx as nx
import logging

# Import the core pipeline
from fcm_nasdaq import prepare_data_and_run, compute_state_from_outer, CONFIG

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger("visualize_fcm")

# ===== Master Color Palette & Typography System =====
COLORS = {
    "yahoo_primary": "#1f77b4",      # Blue for Yahoo
    "google_primary": "#ff7f0e",     # Orange for Google
    "actual": "#2ca02c",              # Green for actual values
    "prediction": "#d62728",          # Red for predictions
    "lstm": "#9467bd",                # Purple for LSTM
    "gru": "#8c564b",                 # Brown for GRU
    "ensemble": "#e377c2",            # Pink for Ensemble
    "fcm_static": "#17becf"           # Cyan for static FCM
}

FONT_SIZES = {
    "title": 14,
    "subtitle": 12,
    "label": 11,
    "legend": 10,
    "annotation": 9,
    "small": 8
}

# ===== End Constants =====

def plot_sentiment_time_series(out):
    """Plot sentiment time series from available sources with improved clarity."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Top plot: Sentiment comparison
    ax = axes[0]
    actual_norm = out["actual_norm"]

    if out["polarity_series_yahoo"] is not None:
        # Dual source: Yahoo and Google
        pol_yahoo = out["polarity_series_yahoo"].reindex(actual_norm.index).fillna(0.0)
        pol_google = out["polarity_series_google"].reindex(actual_norm.index).fillna(0.0)

        # Plot with consistent colors from palette
        ax.plot(pol_yahoo.index, pol_yahoo.rolling(7, min_periods=1).mean(),
                label='Yahoo News Sentiment (7d MA)', color=COLORS["yahoo_primary"], lw=2.5, alpha=0.8)
        ax.plot(pol_google.index, pol_google.rolling(7, min_periods=1).mean(),
                label='Google News Sentiment (7d MA)', color=COLORS["google_primary"], lw=2.5, alpha=0.8)

        # Check for data availability
        yahoo_data_pct = (pol_yahoo != 0).sum() / len(pol_yahoo) * 100 if len(pol_yahoo) > 0 else 0
        google_data_pct = (pol_google != 0).sum() / len(pol_google) * 100 if len(pol_google) > 0 else 0

        # Add data availability indicators
        if yahoo_data_pct > 0:
            ax.text(0.01, 0.95, f"✓ Yahoo Data: {yahoo_data_pct:.0f}% available",
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        if google_data_pct > 0:
            ax.text(0.01, 0.88, f"✓ Google Data: {google_data_pct:.0f}% available",
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    else:
        # Unified sentiment
        pol = out["polarity_series"].reindex(actual_norm.index).fillna(0.0)
        ax.plot(pol.index, pol.rolling(7, min_periods=1).mean(),
                label='Unified News Sentiment (7d MA)', color=COLORS["yahoo_primary"], lw=2.5, alpha=0.8)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_ylabel("Sentiment Polarity [-1, 1]", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("News Sentiment Analysis (Yahoo vs Google)", fontsize=FONT_SIZES["title"], fontweight='bold')
    ax.legend(loc='upper left', fontsize=FONT_SIZES["legend"], framealpha=0.95)
    ax.grid(True, alpha=0.2)

    # Bottom plot: Actual vs normalized
    ax = axes[1]
    ax.plot(actual_norm.index, actual_norm, label='Actual NASDAQ Returns (normalized)',
            color=COLORS["actual"], lw=2.5, alpha=0.7)
    ax.fill_between(actual_norm.index, actual_norm, alpha=0.15, color=COLORS["actual"])
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_ylabel("Normalized Returns", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_xlabel("Date", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("Normalized NASDAQ Returns", fontsize=FONT_SIZES["title"], fontweight='bold')
    ax.legend(loc='upper left', fontsize=FONT_SIZES["legend"], framealpha=0.95)
    ax.grid(True, alpha=0.2)

    # Improve date label readability
    ax.tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('sentiment_timeseries.png', dpi=150, bbox_inches='tight')
    logger.info("Saved sentiment_timeseries.png")
    plt.show()


def plot_model_predictions(out):
    """Plot actual vs predicted NASDAQ comparing Yahoo and Google FCM models with metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    actual_norm = out["actual_norm"]

    # Yahoo Dynamic FCM Predictions
    ax = axes[0, 0]
    dates_yahoo = out["date_history_yahoo"]
    preds_yahoo = out["pred_history_yahoo"]
    actual_subset = actual_norm.loc[actual_norm.index.isin(dates_yahoo)]
    ax.plot(actual_norm.index, actual_norm, label='Actual NASDAQ', color=COLORS["actual"], lw=2.5, alpha=0.7)
    ax.plot(dates_yahoo, preds_yahoo, label='Yahoo FCM Predictions', color=COLORS["yahoo_primary"], lw=2, alpha=0.9)
    ax.fill_between(actual_norm.index, actual_norm, alpha=0.1, color=COLORS["actual"])

    # Metric display for Yahoo
    yahoo_corr = out['dynamic_corr_yahoo']
    yahoo_mse = out['dynamic_mse_yahoo']
    yahoo_corr_str = f"{yahoo_corr:.4f}" if not np.isnan(yahoo_corr) else "N/A"
    yahoo_mse_str = f"{yahoo_mse:.2e}" if not np.isnan(yahoo_mse) and yahoo_mse != float('inf') else "N/A"
    yahoo_avail = "✓" if (out['polarity_series_yahoo'] is not None) else "✗"

    ax.set_title(f"Yahoo Data Source - Dynamic FCM Model\nCorr: {yahoo_corr_str}  |  MSE: {yahoo_mse_str}  {yahoo_avail}",
                 fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    ax.set_ylabel("Normalized Value", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.legend(fontsize=FONT_SIZES["legend"], loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', rotation=45)

    # Google Dynamic FCM Predictions
    ax = axes[0, 1]
    dates_google = out["date_history_google"]
    preds_google = out["pred_history_google"]
    ax.plot(actual_norm.index, actual_norm, label='Actual NASDAQ', color=COLORS["actual"], lw=2.5, alpha=0.7)
    ax.plot(dates_google, preds_google, label='Google FCM Predictions', color=COLORS["google_primary"], lw=2, alpha=0.9)
    ax.fill_between(actual_norm.index, actual_norm, alpha=0.1, color=COLORS["actual"])

    # Metric display for Google
    google_corr = out['dynamic_corr_google']
    google_mse = out['dynamic_mse_google']
    google_corr_str = f"{google_corr:.4f}" if not np.isnan(google_corr) else "N/A"
    google_mse_str = f"{google_mse:.2e}" if not np.isnan(google_mse) and google_mse != float('inf') else "N/A"
    google_avail = "✓" if (out['polarity_series_google'] is not None) else "✗"

    ax.set_title(f"Google Data Source - Dynamic FCM Model\nCorr: {google_corr_str}  |  MSE: {google_mse_str}  {google_avail}",
                 fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    ax.set_ylabel("Normalized Value", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.legend(fontsize=FONT_SIZES["legend"], loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', rotation=45)

    # Model Comparison - Correlation
    ax = axes[1, 0]
    models = ['Static FCM', 'Yahoo FCM\n(Dynamic)', 'Google FCM\n(Dynamic)']
    correlations = [out['static_corr'],
                   out['dynamic_corr_yahoo'] if not np.isnan(out['dynamic_corr_yahoo']) else 0,
                   out['dynamic_corr_google'] if not np.isnan(out['dynamic_corr_google']) else 0]
    colors_bar = [COLORS["fcm_static"], COLORS["yahoo_primary"], COLORS["google_primary"]]
    bars = ax.bar(models, correlations, color=colors_bar, alpha=0.75, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel("Correlation with Actual", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("FCM Model Performance Comparison: Correlation", fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.2, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=FONT_SIZES["annotation"], fontweight='bold')

    # Model Comparison - MSE
    ax = axes[1, 1]
    mses = [out['static_mse'],
           out['dynamic_mse_yahoo'] if not np.isnan(out['dynamic_mse_yahoo']) and out['dynamic_mse_yahoo'] != float('inf') else 1.0,
           out['dynamic_mse_google'] if not np.isnan(out['dynamic_mse_google']) and out['dynamic_mse_google'] != float('inf') else 1.0]
    bars = ax.bar(models, mses, color=colors_bar, alpha=0.75, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel("Mean Squared Error (log scale)", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("FCM Model Performance Comparison: MSE", fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    if any(m > 0 for m in mses):
        ax.set_yscale('log')
    ax.grid(True, alpha=0.2, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, mses):
        height = bar.get_height()
        val_str = f'{val:.2e}' if val > 0 else 'N/A'
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                val_str, ha='center', va='bottom', fontsize=FONT_SIZES["annotation"], fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=150, bbox_inches='tight')
    logger.info("Saved model_predictions.png")
    plt.show()

def plot_sentiment_vs_returns(out):
    """Plot daily returns colored by sentiment."""
    # Determine which sentiment data to use
    polarity = None

    if out["polarity_series_yahoo"] is not None and not out["polarity_series_yahoo"].empty:
        polarity = out["polarity_series_yahoo"]
        title_suffix = "(Yahoo Source)"
    elif out["polarity_series_google"] is not None and not out["polarity_series_google"].empty:
        polarity = out["polarity_series_google"]
        title_suffix = "(Google Source)"
    elif out["polarity_series"] is not None and not out["polarity_series"].empty:
        polarity = out["polarity_series"]
        title_suffix = ""
    else:
        logger.warning("No sentiment data available for scatter plot")
        return

    actual_returns = out["actual_returns"]
    idx = polarity.index.intersection(actual_returns.index)

    if len(idx) == 0:
        logger.warning("No overlapping dates for sentiment vs returns plot")
        return

    pol = polarity.loc[idx]
    ret = actual_returns.loc[idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    scatter = ax.scatter(idx, ret.values, c=pol.values, cmap='RdYlGn', vmin=-1, vmax=1,
                        s=80, alpha=0.6, edgecolors='black', linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sentiment Polarity [-1, 1]", fontsize=10, fontweight='bold')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel("Daily Return", fontsize=11, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11, fontweight='bold')
    ax.set_title(f"Daily NASDAQ Returns Colored by News Sentiment {title_suffix}",
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('sentiment_vs_returns.png', dpi=150, bbox_inches='tight')
    logger.info("Saved sentiment_vs_returns.png")
    plt.show()

def plot_weight_evolution(out):
    """Plot how key edge weights evolve over time during training."""
    weights_history_yahoo = out["weights_history_yahoo"]
    dates_yahoo = out["date_history_yahoo"]

    if not weights_history_yahoo or len(weights_history_yahoo) == 0:
        logger.warning("No weight history available")
        return

    # Extract weights to NASDAQ from a few key nodes
    key_nodes = [n for n in out["node_order"] if n not in ['Monetary_Policy', 'Inflation', 'Corporate_Earnings', 'Investor_Sentiment', 'NASDAQ']][:5]

    fig, ax = plt.subplots(figsize=(15, 7))

    colors_list = plt.cm.tab20(np.linspace(0, 1, len(key_nodes)))

    for node, color in zip(key_nodes, colors_list):
        weights_to_nasdaq = []
        for w_dict in weights_history_yahoo:
            w = w_dict.get(node, {}).get('NASDAQ', 0.0)
            weights_to_nasdaq.append(w)
        ax.plot(range(len(weights_to_nasdaq)), weights_to_nasdaq, label=f'{node} → NASDAQ',
               color=color, lw=2.5, alpha=0.85, marker='o', markersize=4, markevery=max(1, len(weights_to_nasdaq)//20))

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel("Training Step", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_ylabel("Weight Value", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("FCM Weight Evolution During Training (Yahoo Source)", fontsize=FONT_SIZES["title"], fontweight='bold')
    ax.legend(loc='best', fontsize=FONT_SIZES["legend"], ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('weight_evolution.png', dpi=150, bbox_inches='tight')
    logger.info("Saved weight_evolution.png")
    plt.show()

def draw_fcm_graph(node_order, weights_snapshot, activations_snapshot=None, title="FCM Network"):
    """Draw the FCM network graph with improved visualization."""
    G = nx.DiGraph()

    # Add nodes
    for n in node_order:
        G.add_node(n)

    # Add edges from weights (only significant weights)
    edge_weights = []
    for src, targets in weights_snapshot.items():
        for tgt, w in targets.items():
            if abs(w) > 0.01:  # Filter small weights for clarity
                G.add_edge(src, tgt, weight=w)
                edge_weights.append((src, tgt, w))

    fig, ax = plt.subplots(figsize=(16, 10))

    # Layout: separate outer (input) nodes from inner nodes
    inner_nodes = ['Monetary_Policy', 'Inflation', 'Corporate_Earnings', 'Investor_Sentiment', 'NASDAQ']
    outer_nodes = [n for n in node_order if n not in inner_nodes]

    pos = {}
    # Place outer nodes in a circle on the left
    n_outer = len(outer_nodes)
    for i, node in enumerate(outer_nodes):
        angle = 2 * np.pi * i / max(n_outer, 1)
        pos[node] = (0 + np.cos(angle), np.sin(angle))

    # Place inner nodes on the right
    for i, node in enumerate(inner_nodes):
        angle = 2 * np.pi * i / len(inner_nodes)
        pos[node] = (3 + np.cos(angle), np.sin(angle))

    # Node colors and sizes based on activation values
    if activations_snapshot:
        node_vals = [activations_snapshot.get(n, 0.5) for n in G.nodes()]
    else:
        node_vals = [0.5 for _ in G.nodes()]

    sizes = [500 + 2000 * float(v) for v in node_vals]
    colors = [plt.cm.cool(float(v)) for v in node_vals]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                          edgecolors='black', linewidths=2, ax=ax)

    # Draw edges with color indicating sign
    if G.edges():
        edge_widths = [4.0 * abs(d['weight']) / max([abs(e[2]) for e in edge_weights]) for _, _, d in G.edges(data=True)]
        edge_colors = ['#2ca02c' if d['weight'] >= 0 else '#d62728' for _, _, d in G.edges(data=True)]

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                              arrowsize=25, arrowstyle='-|>', connectionstyle='arc3,rad=0.1',
                              ax=ax, alpha=0.7)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

    # Legend
    positive_patch = mpatches.Patch(color='#2ca02c', label='Positive weight')
    negative_patch = mpatches.Patch(color='#d62728', label='Negative weight')
    ax.legend(handles=[positive_patch, negative_patch], loc='upper left', fontsize=11)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    return fig

def plot_feature_importance(out):
    """Plot feature importance (correlation with target) with improved readability."""
    feature_importance = out["feature_importance"]

    fig, ax = plt.subplots(figsize=(13, 9))

    # Sort by importance
    feature_importance_sorted = feature_importance.sort_values(ascending=True)
    colors = [COLORS["actual"] if x > 0 else COLORS["prediction"] for x in feature_importance_sorted.values]

    bars = ax.barh(range(len(feature_importance_sorted)), feature_importance_sorted.values,
                   color=colors, alpha=0.75, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(feature_importance_sorted)))
    ax.set_yticklabels(feature_importance_sorted.index, fontsize=FONT_SIZES["label"])
    ax.set_xlabel("Absolute Correlation with NASDAQ Returns", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("Feature Importance: How Each Node Influences NASDAQ Predictions",
                fontsize=FONT_SIZES["title"], fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

    # Add value labels on bars
    for idx, (bar, val) in enumerate(zip(bars, feature_importance_sorted.values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
               f'{val:.3f}', ha='left', va='center', fontsize=FONT_SIZES["small"])

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    logger.info("Saved feature_importance.png")
    plt.show()

def plot_historical_node_flow(out):
    """
    Create historical node-flow visualization showing how nodes' correlations
    to NASDAQ evolved from Jan 2024 to Jan 2026.

    2-panel layout:
    Panel 1: Heatmap of node correlations over time
    Panel 2: Time series of 4 key nodes + NASDAQ returns
    """
    correlations_history = out["correlations_history_yahoo"]
    inputs_norm = out["inputs_norm_yahoo"]
    actual_norm = out["actual_norm"]

    if not correlations_history or len(correlations_history) < 2:
        logger.warning("Insufficient historical correlation data for node-flow visualization")
        return

    # Extract dates and build correlation matrix over time
    dates_list = sorted(list(correlations_history.keys()))
    node_names = list(correlations_history[dates_list[0]].index)

    # Create 2D array: rows=nodes, columns=dates
    corr_matrix = np.zeros((len(node_names), len(dates_list)))
    for date_idx, date in enumerate(dates_list):
        corr_series = correlations_history[date]
        for node_idx, node in enumerate(node_names):
            if node in corr_series.index:
                corr_matrix[node_idx, date_idx] = corr_series[node]

    # Create figure with 2 panels
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)

    # ===== PANEL 1: Node Correlation Evolution Heatmap =====
    ax1 = fig.add_subplot(gs[0])

    # Use seaborn for better heatmap
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1, interpolation='nearest')

    # Set ticks
    ax1.set_yticks(range(len(node_names)))
    ax1.set_yticklabels(node_names, fontsize=FONT_SIZES["label"])

    # Thin out x-axis labels to avoid crowding
    x_ticks = np.linspace(0, len(dates_list)-1, min(15, len(dates_list)), dtype=int)
    ax1.set_xticks(x_ticks)
    x_labels = [dates_list[i].strftime('%Y-%m-%d') for i in x_ticks]
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)

    # Labels
    ax1.set_xlabel("Date", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax1.set_ylabel("Nodes", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax1.set_title("Historical Node Influence Evolution: How Each Node's Correlation to NASDAQ Changed Over Time (Jan 2024 - Jan 2026)",
                 fontsize=FONT_SIZES["title"], fontweight='bold', pad=15)

    # Colorbar
    cbar1 = plt.colorbar(im, ax=ax1, pad=0.02, label='Correlation to NASDAQ')
    cbar1.set_label('Correlation Strength', fontsize=FONT_SIZES["label"], fontweight='bold')

    # ===== PANEL 2: Key Nodes Time Series + NASDAQ =====
    ax2 = fig.add_subplot(gs[1])

    # Select 4 key nodes if they exist
    key_nodes = ['Inflation', 'Low_Unemployment', 'Oil_Prices', 'Tech_News_Sentiment']
    # Filter to only nodes that actually exist
    available_key_nodes = [n for n in key_nodes if n in node_names]
    if not available_key_nodes:
        # Fall back to first 4 nodes
        available_key_nodes = node_names[:4]

    colors_panel2 = [COLORS["yahoo_primary"], COLORS["google_primary"],
                      COLORS["prediction"], COLORS["lstm"]]

    # Plot NASDAQ on secondary y-axis
    ax2_twin = ax2.twinx()
    line_nasdaq = ax2_twin.plot(actual_norm.index, actual_norm,
                                 label='NASDAQ Returns (normalized)',
                                 color=COLORS["actual"], lw=3, alpha=0.7, zorder=10)
    ax2_twin.set_ylabel("NASDAQ Normalized Returns", fontsize=FONT_SIZES["label"], fontweight='bold', color=COLORS["actual"])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS["actual"])
    ax2_twin.grid(True, alpha=0.1)

    # Plot key nodes
    for node_idx, (node, color) in enumerate(zip(available_key_nodes, colors_panel2)):
        if node in inputs_norm.columns:
            node_data = inputs_norm[node]
            ax2.plot(node_data.index, node_data, label=f'{node} (normalized)',
                    color=color, lw=2, alpha=0.8, marker='o', markersize=2, markevery=max(1, len(node_data)//30))

    ax2.set_xlabel("Date", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax2.set_ylabel("Node Values (normalized)", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax2.set_title("Key Node Values Over Time vs NASDAQ (Jan 2024 - Jan 2026)",
                 fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(axis='x', rotation=45)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              fontsize=FONT_SIZES["legend"], framealpha=0.95)

    plt.savefig('historical_node_flow.png', dpi=150, bbox_inches='tight')
    logger.info("Saved historical_node_flow.png")
    plt.show()

def plot_correlation_heatmap_snapshots(out):
    """Plot historical correlation snapshots over time."""
    correlations_history = out["correlations_history_yahoo"]

    if not correlations_history or len(correlations_history) < 2:
        logger.warning("Insufficient correlation history data")
        return

    # Select 4 snapshots for visualization (beginning, 1/3, 2/3, end)
    snapshot_dates = sorted(list(correlations_history.keys()))
    indices = [0, len(snapshot_dates)//3, 2*len(snapshot_dates)//3, -1]
    selected_dates = [snapshot_dates[i] for i in indices]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, date in enumerate(selected_dates):
        corr_series = correlations_history[date]

        ax = axes[idx]
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in corr_series.values]
        ax.barh(range(len(corr_series)), corr_series.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(corr_series)))
        ax.set_yticklabels(corr_series.index, fontsize=9)
        ax.set_xlabel("Correlation", fontsize=10, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title(f"Node Correlations with Target\n{date.strftime('%Y-%m-%d')}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        ax.set_xlim(-1, 1)

    plt.suptitle("Historical Correlation Evolution: How Nodes Influence NASDAQ Over Time",
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('correlation_snapshots.png', dpi=150, bbox_inches='tight')
    logger.info("Saved correlation_snapshots.png")
    plt.show()

def plot_lstm_comparison(out):
    """Plot comprehensive 5-model performance comparison: LSTM, GRU, Yahoo FCM, Google FCM, Ensemble."""
    lstm_metrics = out.get("lstm_metrics", {})
    gru_metrics = out.get("gru_metrics", {})
    metrics_comparison = out.get("metrics_comparison", {})

    if not lstm_metrics or all(v == 0 for v in lstm_metrics.values()):
        logger.warning("Deep learning metrics not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ===== Panel 1: Test MAE Comparison (5 models) =====
    ax = axes[0, 0]
    model_names = ['LSTM', 'GRU', 'Yahoo\nFCM', 'Google\nFCM', 'Ensemble']
    test_maes = [
        lstm_metrics.get('test_mae', 0),
        gru_metrics.get('test_mae', 0),
        abs(out.get('dynamic_corr_yahoo', 0)),
        abs(out.get('dynamic_corr_google', 0)),
        0  # Placeholder for ensemble
    ]
    colors_models = [COLORS["lstm"], COLORS["gru"], COLORS["yahoo_primary"], COLORS["google_primary"], COLORS["ensemble"]]
    bars = ax.bar(model_names, test_maes, color=colors_models, alpha=0.75, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel("Mean Absolute Error", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("Test Set MAE: Neural Networks vs FCM Models", fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, test_maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{val:.4f}', ha='center', va='bottom', fontsize=FONT_SIZES["annotation"], fontweight='bold')

    # ===== Panel 2: Test MSE Comparison (5 models) =====
    ax = axes[0, 1]
    test_mses = [
        lstm_metrics.get('test_mse', 0),
        gru_metrics.get('test_mse', 0),
        out.get('dynamic_mse_yahoo', 0),
        out.get('dynamic_mse_google', 0),
        0  # Placeholder for ensemble
    ]
    bars = ax.bar(model_names, test_mses, color=colors_models, alpha=0.75, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel("Mean Squared Error", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("Test Set MSE: Neural Networks vs FCM Models", fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    if any(m > 0 for m in test_mses):
        ax.set_yscale('log')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, test_mses):
        height = bar.get_height()
        val_str = f'{val:.2e}' if val > 0 else '0'
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.2 if height > 0 else 0.01,
               val_str, ha='center', va='bottom', fontsize=FONT_SIZES["annotation"], fontweight='bold')

    # ===== Panel 3: LSTM Deep Learning Metrics =====
    ax = axes[1, 0]
    lstm_metric_names = ['Train\nMAE', 'Test\nMAE', 'Train\nMSE', 'Test\nMSE', 'Test\nR²']
    lstm_metric_values = [
        lstm_metrics.get('train_mae', 0),
        lstm_metrics.get('test_mae', 0),
        lstm_metrics.get('train_mse', 0),
        lstm_metrics.get('test_mse', 0),
        lstm_metrics.get('test_r2', 0)
    ]
    colors_lstm = [COLORS["lstm"], COLORS["lstm"], COLORS["lstm"], COLORS["lstm"], COLORS["lstm"]]
    bars = ax.bar(lstm_metric_names, lstm_metric_values, color=colors_lstm, alpha=0.75, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel("Metric Value", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("LSTM Deep Learning Model: Complete Metrics", fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, lstm_metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{val:.4f}', ha='center', va='bottom', fontsize=FONT_SIZES["annotation"], fontweight='bold')

    # ===== Panel 4: GRU Deep Learning Metrics =====
    ax = axes[1, 1]
    gru_metric_names = ['Train\nMAE', 'Test\nMAE', 'Train\nMSE', 'Test\nMSE', 'Test\nR²']
    gru_metric_values = [
        gru_metrics.get('train_mae', 0),
        gru_metrics.get('test_mae', 0),
        gru_metrics.get('train_mse', 0),
        gru_metrics.get('test_mse', 0),
        gru_metrics.get('test_r2', 0)
    ]
    colors_gru = [COLORS["gru"], COLORS["gru"], COLORS["gru"], COLORS["gru"], COLORS["gru"]]
    bars = ax.bar(gru_metric_names, gru_metric_values, color=colors_gru, alpha=0.75, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel("Metric Value", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("GRU Deep Learning Model: Complete Metrics", fontsize=FONT_SIZES["subtitle"], fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, gru_metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{val:.4f}', ha='center', va='bottom', fontsize=FONT_SIZES["annotation"], fontweight='bold')

    plt.suptitle("Comprehensive Model Comparison: 5-Model Analysis", fontsize=FONT_SIZES["title"], fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('lstm_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("Saved lstm_comparison.png")
    plt.show()


def plot_ensemble_comparison(out):
    """Compare predictions from Yahoo FCM, Google FCM, LSTM, GRU, and Ensemble models."""
    fig, ax = plt.subplots(figsize=(18, 7))

    actual_norm = out["actual_norm"]
    dates_yahoo = out["date_history_yahoo"]

    # Plot actual NASDAQ (thick black line, highest z-order)
    ax.plot(actual_norm.index, actual_norm, label='Actual NASDAQ (normalized)',
           color=COLORS["actual"], lw=3, alpha=0.9, zorder=10)

    # Dynamic FCM Yahoo predictions
    ax.plot(dates_yahoo, out["pred_history_yahoo"], label='Dynamic FCM (Yahoo Source)',
           color=COLORS["yahoo_primary"], lw=2, alpha=0.7, linestyle='--', zorder=7)

    # Dynamic FCM Google predictions
    dates_google = out["date_history_google"]
    ax.plot(dates_google, out["pred_history_google"], label='Dynamic FCM (Google Source)',
           color=COLORS["google_primary"], lw=2, alpha=0.7, linestyle='--', zorder=7)

    # Ensemble predictions (if available)
    ensemble_pred = out.get("ensemble_predictions")
    if ensemble_pred is not None and len(ensemble_pred) > 0:
        # Align ensemble predictions with actual timeline
        ensemble_dates = actual_norm.index[-len(ensemble_pred):]
        ax.plot(ensemble_dates, ensemble_pred, label='Ensemble Predictions (LSTM+GRU+FCM)',
               color=COLORS["ensemble"], lw=2.5, alpha=0.85, linestyle='-', zorder=8, marker='o', markersize=3, markevery=5)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.set_xlabel("Date", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_ylabel("Normalized NASDAQ Returns", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("Multi-Model Ensemble Comparison: Yahoo vs Google FCM + Deep Learning",
                fontsize=FONT_SIZES["title"], fontweight='bold')
    ax.legend(loc='best', fontsize=FONT_SIZES["legend"], framealpha=0.96, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', rotation=45)

    # Add ensemble accuracy metrics if available
    if ensemble_pred is not None and len(ensemble_pred) > 0:
        ensemble_corr = out.get("metrics_comparison", {}).get("Ensemble", {}).get("corr", 0)
        ensemble_mse = out.get("metrics_comparison", {}).get("Ensemble", {}).get("mse", 0)
        metrics_text = f"Ensemble Accuracy: Corr={ensemble_corr:.4f}, MSE={ensemble_mse:.2e}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('ensemble_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("Saved ensemble_comparison.png")
    plt.show()


def plot_node_interaction_heatmap(out):
    """Create a heatmap showing final node-to-node interactions from weight matrix with debugging."""
    weights_final = out["weights_history_yahoo"][-1] if out["weights_history_yahoo"] else out["weights_init"]
    node_order = out["node_order"]

    # Build correlation matrix from weights
    n_nodes = len(node_order)
    weight_matrix = np.zeros((n_nodes, n_nodes))

    # Debug: Count non-zero weights
    non_zero_count = 0
    for i, src in enumerate(node_order):
        if src in weights_final:
            for j, tgt in enumerate(node_order):
                if tgt in weights_final[src]:
                    w = weights_final[src][tgt]
                    weight_matrix[i, j] = w
                    if abs(w) > 1e-6:
                        non_zero_count += 1

    logger.info(f"Node interaction heatmap: {non_zero_count} non-zero weights out of {n_nodes**2} total")

    fig, ax = plt.subplots(figsize=(16, 14))

    # Use better colormap and ensure proper normalization
    im = ax.imshow(weight_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(n_nodes))
    ax.set_yticks(range(n_nodes))
    ax.set_xticklabels(node_order, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticklabels(node_order, fontsize=10, fontweight='bold')

    ax.set_xlabel("Target Node (influences →)", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_ylabel("Source Node (← influences from)", fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title("FCM Final Weight Matrix: Node-to-Node Influence Strength", fontsize=FONT_SIZES["title"], fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Weight Value [-1 to +1]", fontsize=FONT_SIZES["label"], fontweight='bold')

    # Add text annotations with better threshold (show more weights)
    for i in range(n_nodes):
        for j in range(n_nodes):
            w = weight_matrix[i, j]
            # Show all weights ≥ 0.02 or all weights that are non-zero
            if abs(w) >= 0.02:
                # Color text based on background
                text_color = 'white' if abs(w) > 0.5 else 'black'
                ax.text(j, i, f'{w:.3f}',
                       ha="center", va="center", color=text_color,
                       fontsize=FONT_SIZES["annotation"], fontweight='bold')

    # Add grid for clarity
    ax.set_xticks(np.arange(n_nodes) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_nodes) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Add summary statistics
    max_weight = np.max(np.abs(weight_matrix))
    mean_weight = np.mean(np.abs(weight_matrix[weight_matrix != 0])) if non_zero_count > 0 else 0
    summary_text = f"Max Weight: {max_weight:.4f}  |  Mean Weight (non-zero): {mean_weight:.4f}"
    ax.text(0.5, -0.08, summary_text, transform=ax.transAxes,
           ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('node_interaction_heatmap.png', dpi=150, bbox_inches='tight')
    logger.info("Saved node_interaction_heatmap.png")
    plt.show()


def plot_individual_stock_prices(out):
    """Plot individual stock prices with sentiment overlay."""
    price_df = out["price_df_yahoo"]
    polarity = out.get("polarity_series")

    if polarity is None or polarity.empty:
        polarity = pd.Series(0, index=price_df.index)

    # Select major stocks
    major_stocks = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', '^VIX', '^IXIC']
    available_stocks = [s for s in major_stocks if s in price_df.columns]

    if not available_stocks:
        logger.warning("No major stocks found in price data")
        return

    # Create grid: 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, stock in enumerate(available_stocks):
        ax = axes[idx]

        # Normalize price for comparison (0-1 scale)
        price = price_df[stock]
        price_norm = (price - price.min()) / (price.max() - price.min() + 1e-8)

        # Reindex sentiment to match price dates
        sent_aligned = polarity.reindex(price.index).fillna(0)

        # Plot price
        ax.plot(price.index, price_norm, label='Normalized Price', color='#1f77b4', lw=2.5, alpha=0.8)

        # Add sentiment as filled area
        ax.fill_between(price.index, 0, (sent_aligned + 1) / 2, alpha=0.2, color='#ff7f0e', label='Sentiment (scaled)')

        ax.set_ylabel("Normalized Price", fontsize=10, fontweight='bold')
        ax.set_title(f"{stock} - Price & Sentiment", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.1, 1.1)

    plt.suptitle("Individual Stock Prices with News Sentiment Overlay",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('stock_prices_sentiment.png', dpi=150, bbox_inches='tight')
    logger.info("Saved stock_prices_sentiment.png")
    plt.show()

def plot_stock_correlation_matrix(out):
    """Plot correlation matrix between stocks."""
    price_df = out["price_df_yahoo"]

    # Calculate returns
    returns = price_df.pct_change().dropna()

    # Keep major stocks only
    major_stocks = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'XLY', 'XLI', '^VIX', '^IXIC']
    available = [s for s in major_stocks if s in returns.columns]

    if len(available) < 2:
        logger.warning("Insufficient stocks for correlation analysis")
        return

    corr_matrix = returns[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticklabels(available, fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation Coefficient", fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(len(available)):
        for j in range(len(available)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha="center", va="center",
                   color=color, fontsize=9, fontweight='bold')

    ax.set_title("Stock Return Correlations", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('stock_correlation_matrix.png', dpi=150, bbox_inches='tight')
    logger.info("Saved stock_correlation_matrix.png")
    plt.show()

def plot_stock_sentiment_impact(out):
    """Show sentiment correlation with individual stocks."""
    price_df = out["price_df_yahoo"]
    polarity = out.get("polarity_series")

    if polarity is None or polarity.empty:
        logger.warning("No sentiment data for impact analysis")
        return

    # Calculate returns and correlate with sentiment
    returns = price_df.pct_change()

    major_stocks = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'XLY', 'XLI', '^VIX', '^IXIC']
    available = [s for s in major_stocks if s in returns.columns]

    correlations = []
    for stock in available:
        common_idx = returns[stock].index.intersection(polarity.index)
        if len(common_idx) > 10:
            corr = returns[stock].loc[common_idx].corr(polarity.loc[common_idx])
            correlations.append((stock, corr if not np.isnan(corr) else 0))

    if not correlations:
        logger.warning("Could not compute sentiment correlations")
        return

    stocks, corrs = zip(*sorted(correlations, key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2ca02c' if c > 0 else '#d62728' for c in corrs]
    bars = ax.barh(range(len(stocks)), corrs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_yticks(range(len(stocks)))
    ax.set_yticklabels(stocks, fontsize=11, fontweight='bold')
    ax.set_xlabel("Correlation with News Sentiment", fontsize=11, fontweight='bold')
    ax.set_title("Stock Sensitivity to News Sentiment", fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.2, axis='x')

    # Add value labels
    for bar, val in zip(bars, corrs):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {val:.3f}', ha='left' if val > 0 else 'right', va='center',
               fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('stock_sentiment_impact.png', dpi=150, bbox_inches='tight')
    logger.info("Saved stock_sentiment_impact.png")
    plt.show()

def plot_stock_predictions_comparison(out):
    """Compare actual vs predicted for key stocks."""
    price_df_yahoo = out["price_df_yahoo"]
    actual_norm = out["actual_norm"]
    dates_yahoo = out["date_history_yahoo"]
    preds_yahoo = np.array(out["pred_history_yahoo"])

    # Key stocks
    key_stocks = ['NVDA', 'AAPL', 'MSFT', 'AMZN']
    available = [s for s in key_stocks if s in price_df_yahoo.columns]

    if not available:
        logger.warning("No stocks available for prediction comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, stock in enumerate(available):
        ax = axes[idx]

        # Get stock returns
        stock_price = price_df_yahoo[stock]
        stock_returns = stock_price.pct_change().fillna(0)

        # Normalize
        stock_norm = (stock_returns - stock_returns.min()) / (stock_returns.max() - stock_returns.min() + 1e-8)
        stock_norm = stock_norm.clip(-5, 5)

        # Plot actual returns
        ax.plot(stock_price.index, stock_norm, label='Actual Returns (normalized)',
               color='#2ca02c', lw=2, alpha=0.7)

        # Plot FCM predictions (scaled to stock range)
        dates_common = [d for d in dates_yahoo if d in stock_price.index]
        if dates_common:
            preds_subset = preds_yahoo[:len(dates_common)]
            ax.plot(dates_common, preds_subset, label='FCM Predictions',
                   color='#d62728', lw=1.5, alpha=0.7, linestyle='--')

        ax.set_ylabel("Normalized Returns", fontsize=10, fontweight='bold')
        ax.set_title(f"{stock} - Actual vs FCM Predictions", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Individual Stock Returns vs FCM Predictions Comparison",
                fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('stock_predictions_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("Saved stock_predictions_comparison.png")
    plt.show()

def plot_cumulative_returns_comparison(out):
    """Compare cumulative returns: stocks, NASDAQ, and FCM model."""
    price_df = out["price_df_yahoo"]
    actual_norm = out["actual_norm"]
    dates_yahoo = out["date_history_yahoo"]
    preds_yahoo = out["pred_history_yahoo"]

    # Key sectors
    sector_stocks = {
        'Tech (NVDA+AAPL+MSFT)': ['NVDA', 'AAPL', 'MSFT'],
        'Cloud (GOOGL+AMZN)': ['GOOGL', 'AMZN'],
        'Market (SPY)': ['SPY'],
        'NASDAQ': ['^IXIC']
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    for sector_name, stocks in sector_stocks.items():
        available = [s for s in stocks if s in price_df.columns]
        if not available:
            continue

        # Calculate cumulative returns
        prices = price_df[available]
        returns = prices.pct_change().fillna(0)
        cum_returns = (1 + returns).prod(axis=1).cumprod()

        ax.plot(cum_returns.index, cum_returns, label=sector_name, lw=2.5, alpha=0.8)

    # Add FCM predictions (cumulative)
    cum_fcm = (1 + np.array(preds_yahoo) / 100).cumprod()
    ax.plot(dates_yahoo, cum_fcm, label='FCM Predictions (scaled)',
           color='#d62728', lw=2, alpha=0.7, linestyle='--')

    ax.set_xlabel("Date", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cumulative Return Factor", fontsize=11, fontweight='bold')
    ax.set_title("Sector Cumulative Returns vs FCM Model Predictions", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('cumulative_returns_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("Saved cumulative_returns_comparison.png")
    plt.show()

def plot_stock_volatility_sentiment(out):
    """Show relationship between stock volatility and sentiment."""
    price_df = out["price_df_yahoo"]
    polarity = out.get("polarity_series")

    if polarity is None or polarity.empty:
        logger.warning("No sentiment data for volatility analysis")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    major_stocks = ['NVDA', 'AAPL', 'MSFT', 'AMZN']

    for idx, stock in enumerate(major_stocks):
        if stock not in price_df.columns:
            continue

        ax = axes[idx]

        # Calculate rolling volatility
        returns = price_df[stock].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized

        # Align with sentiment
        common_idx = volatility.index.intersection(polarity.index)
        vol_aligned = volatility.loc[common_idx]
        sent_aligned = polarity.loc[common_idx]

        # Scatter plot
        scatter = ax.scatter(sent_aligned.values, vol_aligned.values,
                           c=range(len(sent_aligned)), cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add trend line
        if len(sent_aligned) > 10:
            z = np.polyfit(sent_aligned.values, vol_aligned.values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(sent_aligned.min(), sent_aligned.max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", lw=2, alpha=0.8, label=f'Trend')

        ax.set_xlabel("Sentiment Polarity", fontsize=10, fontweight='bold')
        ax.set_ylabel("Annualized Volatility", fontsize=10, fontweight='bold')
        ax.set_title(f"{stock} - Volatility vs Sentiment", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Stock Volatility Relationship with News Sentiment",
                fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('stock_volatility_sentiment.png', dpi=150, bbox_inches='tight')
    logger.info("Saved stock_volatility_sentiment.png")
    plt.show()

def plot_residual_analysis(out):
    """Analyze and plot prediction errors (residuals)."""
    actual_norm = out["actual_norm"]
    dates_yahoo = out["date_history_yahoo"]
    preds_yahoo = np.array(out["pred_history_yahoo"])  # Convert to numpy array

    actual_subset = actual_norm.loc[actual_norm.index.isin(dates_yahoo)]
    residuals = actual_subset.values - preds_yahoo

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residuals over time
    ax = axes[0, 0]
    ax.scatter(dates_yahoo, residuals, alpha=0.6, color='#d62728', edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='--', lw=2)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=10, fontweight='bold')
    ax.set_title("Residuals Over Time", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # Residual histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=30, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(residuals), color='red', linestyle='--', lw=2, label=f'Mean: {np.mean(residuals):.4f}')
    ax.axvline(np.median(residuals), color='green', linestyle='--', lw=2, label=f'Median: {np.median(residuals):.4f}')
    ax.set_xlabel("Residual Value", fontsize=10, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=10, fontweight='bold')
    ax.set_title("Residual Distribution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Predicted vs Actual
    ax = axes[1, 0]
    ax.scatter(actual_subset.values, preds_yahoo, alpha=0.6, color='#1f77b4', edgecolors='black', linewidth=0.5)
    min_val = min(actual_subset.min(), np.min(preds_yahoo))
    max_val = max(actual_subset.max(), np.max(preds_yahoo))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel("Actual Normalized Returns", fontsize=10, fontweight='bold')
    ax.set_ylabel("Predicted Normalized Returns", fontsize=10, fontweight='bold')
    ax.set_title("Predicted vs Actual", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Q-Q plot for normality
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot: Residual Normality", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.suptitle("Residual Analysis: FCM Dynamic Model (Yahoo Source)",
                fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
    logger.info("Saved residual_analysis.png")
    plt.show()

def main():
    """Run all visualizations."""
    logger.info("Running FCM analysis and generating visualizations...")
    out = prepare_data_and_run()

    # Basic visualizations
    logger.info("Generating sentiment time series plot...")
    plot_sentiment_time_series(out)

    logger.info("Generating model predictions plot...")
    plot_model_predictions(out)

    logger.info("Generating sentiment vs returns plot...")
    plot_sentiment_vs_returns(out)

    logger.info("Generating weight evolution plot...")
    plot_weight_evolution(out)

    # FCM network
    logger.info("Generating FCM network visualization...")
    if out["weights_history_yahoo"]:
        last_weights = out["weights_history_yahoo"][-1]
    else:
        last_weights = out["weights_init"]

    idx = out["inputs_norm_yahoo"].index[-1]
    last_outer = out["inputs_norm_yahoo"].loc[idx]
    activ = compute_state_from_outer(last_outer, last_weights, out["node_order"])
    fig = draw_fcm_graph(out["node_order"], last_weights, activations_snapshot=activ,
                        title="FCM Network - Final State (Yahoo Source)")
    plt.savefig('fcm_network.png', dpi=150, bbox_inches='tight')
    logger.info("Saved fcm_network.png")
    plt.show()

    # Advanced visualizations
    logger.info("Generating feature importance plot...")
    plot_feature_importance(out)

    logger.info("Generating historical node-flow visualization...")
    plot_historical_node_flow(out)

    logger.info("Generating historical correlation snapshots...")
    plot_correlation_heatmap_snapshots(out)

    logger.info("Generating node interaction heatmap...")
    plot_node_interaction_heatmap(out)

    logger.info("Generating LSTM deep learning comparison...")
    plot_lstm_comparison(out)

    logger.info("Generating ensemble model comparison...")
    plot_ensemble_comparison(out)

    logger.info("Generating residual analysis...")
    plot_residual_analysis(out)

    # Stock-level visualizations
    logger.info("Generating individual stock prices plot...")
    plot_individual_stock_prices(out)

    logger.info("Generating stock correlation matrix...")
    plot_stock_correlation_matrix(out)

    logger.info("Generating stock sentiment impact plot...")
    plot_stock_sentiment_impact(out)

    logger.info("Generating stock predictions comparison...")
    plot_stock_predictions_comparison(out)

    logger.info("Generating cumulative returns comparison...")
    plot_cumulative_returns_comparison(out)

    logger.info("Generating stock volatility sentiment plot...")
    plot_stock_volatility_sentiment(out)

    logger.info("All visualizations completed!")

if __name__ == "__main__":
    main()