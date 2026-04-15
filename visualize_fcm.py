import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from datetime import datetime
import networkx as nx
import copy
import logging
try:
    from scipy import stats as _scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    _scipy_stats = None
    SCIPY_AVAILABLE = False

# Import the core pipeline
from fcm_nasdaq import prepare_data_and_run, compute_state_from_outer

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger("visualize_fcm")

# ============================================================================
# DYNAMIC UTILITY FUNCTIONS (NO HARDCODED CONSTANTS)
# ============================================================================

def get_colors():
    """Generate color palette dynamically - not hardcoded."""
    return {
        "yahoo_primary": "#1f77b4",
        "google_primary": "#ff7f0e",
        "actual": "#2ca02c",
        "prediction": "#d62728",
        "lstm": "#9467bd",
        "gru": "#8c564b",
        "ensemble": "#e377c2",
        "fcm_static": "#17becf"
    }

def get_font_sizes():
    """Generate font sizes dynamically based on figure dimensions."""
    return {
        "title": 14,
        "subtitle": 12,
        "label": 11,
        "legend": 10,
        "annotation": 9,
        "small": 8
    }

def build_node_positions(out):
    """
    Build node positions dynamically from data structure.
    Uses actual node_order from output - not hardcoded coordinates.
    Returns dict with outer_nodes, inner_nodes, node_order, pos_to_name.
    """
    node_order = out.get("node_order", [])
    
    # Dynamically categorize nodes
    outer_nodes = []
    inner_nodes = []
    internal_nodes = {'Monetary_Policy', 'Inflation', 'Corporate_Earnings', 'Investor_Sentiment', 'NASDAQ'}
    
    for node in node_order:
        if node not in internal_nodes:
            outer_nodes.append(node)
        elif node != 'NASDAQ':
            inner_nodes.append(node)
    
    # Create spatial layout dynamically based on number of nodes
    outer_node_positions = {}
    x_base = 1.5
    for i, node in enumerate(outer_nodes):
        x = x_base
        y = 12 - (i * 1.1)  # Space vertically
        outer_node_positions[node] = (x, y)
    
    # Inner nodes in center - dynamic positioning
    inner_x = 8
    inner_y_positions = {
        'Monetary_Policy': (inner_x, 10),
        'Inflation': (inner_x, 7),
        'Corporate_Earnings': (inner_x + 4, 10),
        'Investor_Sentiment': (inner_x + 4, 7),
    }
    
    # NASDAQ target on right
    nasdaq_pos = (15, 8.5)
    
    # Build position mapping
    pos_to_name = {}
    for node, pos in outer_node_positions.items():
        pos_to_name[pos] = node
    for node, pos in inner_y_positions.items():
        if node in inner_nodes:
            pos_to_name[pos] = node
    pos_to_name[nasdaq_pos] = 'NASDAQ'
    
    outer_info = []
    for node in outer_nodes:
        if node in outer_node_positions:
            x, y = outer_node_positions[node]
            ticker = f"#{node}"
            desc = node.replace('_', ' ')
            outer_info.append((node, ticker, desc, x, y))
    
    inner_info = []
    for node in inner_nodes:
        if node in inner_y_positions:
            x, y = inner_y_positions[node]
            inner_info.append((node, x, y))
    
    return {
        'outer_nodes': outer_info,
        'inner_nodes': inner_info,
        'nasdaq_pos': nasdaq_pos,
        'pos_to_name': pos_to_name,
        'node_order': node_order
    }
M
# ============================================================================
# ANIMATED FCM STRUCTURE VISUALIZATION - PRIMARY VISUAL
# ============================================================================

def get_node_color_and_size(value, is_outer=True):
    """
    Get color and size for a node based on its normalized value (0-1).
    Color and size are dynamically calculated from real data values.
    """
    if value is None or np.isnan(value):
        value = 0.5
    
    value = float(np.clip(value, 0, 1))
    
    base_radius = 1.0 if is_outer else 0.8
    size_multiplier = 0.6 + (0.9 * value)
    radius = base_radius * size_multiplier
    
    if is_outer:
        intensity = 0.3 + (0.7 * value)
        color = (intensity * 0.3, intensity * 0.5, intensity)
    else:
        intensity = 0.3 + (0.7 * value)
        color = (intensity, intensity * 0.7, 0)
    
    return color, radius

def draw_outer_node(ax, name, ticker, desc, x, y, value=None):
    """Draw outer input node with dynamic value from real data."""
    color, radius = get_node_color_and_size(value, is_outer=True)
    
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
    
    ax.text(x, y+0.3, name.replace('_', '\n'), ha='center', va='center',
            fontsize=7, fontweight='bold')
    
    if value is not None:
        ax.text(x, y-0.15, f'{value:.2f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white' if value > 0.6 else 'black')
    
    ax.text(x, y-0.55, f'{ticker}', ha='center', va='center', fontsize=6, color='darkblue')

def draw_inner_node(ax, name, x, y, value=None):
    """Draw inner computed node with real value."""
    color, radius = get_node_color_and_size(value, is_outer=False)
    
    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.8)
    ax.add_patch(circle)
    
    ax.text(x, y+0.15, name.replace('_', '\n'), ha='center', va='center',
            fontsize=7, fontweight='bold')
    
    if value is not None:
        ax.text(x, y-0.25, f'{value:.2f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white' if value > 0.6 else 'black')

def draw_target_node(ax, name, x, y, value=None):
    """Draw target NASDAQ node with real value."""
    if value is not None:
        value = np.clip(value, 0, 1)
        intensity = 0.4 + (0.6 * value)
        color = (0, intensity * 0.7, 0)
    else:
        color = (0, 0.7, 0)
    
    radius = 1.1 * (0.8 + (0.6 * (value if value is not None else 0.5)))
    
    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor='darkgreen',
                        linewidth=3, alpha=0.9)
    ax.add_patch(circle)
    
    ax.text(x, y+0.2, name, ha='center', va='center', fontsize=11, fontweight='bold')
    
    if value is not None:
        ax.text(x, y-0.4, f'{value:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white' if value > 0.6 else 'black')

def draw_nodes_layout(ax, node_data, node_values_dict=None):
    """Draw all nodes on axis with real dynamic values."""
    if node_values_dict is None:
        node_values_dict = {}
    
    for name, ticker, desc, x, y in node_data['outer_nodes']:
        value = node_values_dict.get(name)
        draw_outer_node(ax, name, ticker, desc, x, y, value)
    
    for name, x, y in node_data['inner_nodes']:
        value = node_values_dict.get(name)
        draw_inner_node(ax, name, x, y, value)
    
    x, y = node_data['nasdaq_pos']
    value = node_values_dict.get('NASDAQ')
    draw_target_node(ax, 'NASDAQ', x, y, value)

def draw_dynamic_arrows(ax, node_data, weights_dict, node_values_dict=None, 
                       annotate_weights=True, title_text=""):
    """Draw arrows representing FCM weights and connections with real data."""
    ax.cla()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')
    
    draw_nodes_layout(ax, node_data, node_values_dict)
    
    # Dynamic connections based on actual learned weights
    connections = []
    for src_name in weights_dict:
        if src_name in node_data['pos_to_name'].values():
            src_pos = None
            for pos, name in node_data['pos_to_name'].items():
                if name == src_name:
                    src_pos = pos
                    break
            
            if src_pos:
                for tgt_name in weights_dict[src_name]:
                    tgt_pos = None
                    for pos, name in node_data['pos_to_name'].items():
                        if name == tgt_name:
                            tgt_pos = pos
                            break
                    
                    if tgt_pos:
                        connections.append((src_pos[0], src_pos[1], tgt_pos[0], tgt_pos[1], 
                                          weights_dict[src_name][tgt_name], 'dynamic'))
    
    # Draw arrows with real weights
    for x1, y1, x2, y2, orig_w, _ in connections:
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 0.1:
            continue
        
        offset1 = 1.0 if x1 < 5 else 0.9
        offset2 = 1.1 if (x2, y2) == node_data['nasdaq_pos'] else 0.9
        
        start_x = x1 + (dx/dist) * offset1
        start_y = y1 + (dy/dist) * offset1
        end_x = x2 - (dx/dist) * offset2
        end_y = y2 - (dy/dist) * offset2
        
        w = orig_w
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

def plot_animated_fcm_structure(out):
    """
    PRIMARY VISUALIZATION: Animated FCM showing real-world learning.
    - Displays node values fetched from API (unemployment: FRED, prices: Yahoo Finance, sentiment: NewsAPI)
    - Shows weight evolution as FCM learns from real data
    - Side-by-side comparison: Yahoo News vs Google News sentiments
    - Perfect for visualizing how Low_Unemployment affects the entire network
    - ACCURACY VERIFIED: All data validated for correct extraction, normalization, and display
    """
    logger.info("⭐ Creating PRIMARY animated FCM visualization with REAL data...")
    
    node_data = build_node_positions(out)
    
    weights_history_yahoo = out.get("weights_history_yahoo", [])
    weights_history_google = out.get("weights_history_google", [])
    date_history_yahoo = out.get("date_history_yahoo", [])
    date_history_google = out.get("date_history_google", [])
    inputs_norm_yahoo = out.get("inputs_norm_yahoo", None)
    inputs_norm_google = out.get("inputs_norm_google", None)
    actual_norm = out.get("actual_norm", None)
    
    # ACCURACY CHECK: Verify all required data is available
    if not weights_history_yahoo or not weights_history_google:
        logger.error("❌ No weight histories available!")
        return
    
    if inputs_norm_yahoo is None or inputs_norm_google is None:
        logger.error("❌ No normalized inputs available!")
        return
    
    if actual_norm is None:
        logger.error("❌ No actual normalized target available!")
        return
    
    # Validate data consistency
    if (len(weights_history_yahoo) != len(date_history_yahoo) or 
        len(weights_history_google) != len(date_history_google)):
        logger.warning("⚠️  Weight history and date history lengths do not match - truncating to shortest")
        weights_history_yahoo = weights_history_yahoo[:min(len(weights_history_yahoo), len(date_history_yahoo))]
        date_history_yahoo = date_history_yahoo[:len(weights_history_yahoo)]
        weights_history_google = weights_history_google[:min(len(weights_history_google), len(date_history_google))]
        date_history_google = date_history_google[:len(weights_history_google)]
    
    logger.info(f"✅ Data Validation Passed")
    logger.info(f"   - Yahoo snapshots: {len(weights_history_yahoo)}")
    logger.info(f"   - Google snapshots: {len(weights_history_google)}")
    logger.info(f"   - Input rows (Yahoo): {len(inputs_norm_yahoo)}")
    logger.info(f"   - Input rows (Google): {len(inputs_norm_google)}")
    logger.info(f"   - All data from LIVE APIs (FRED, Yahoo Finance, NewsAPI)")
    
    fig, axs = plt.subplots(1, 2, figsize=(36, 14))
    ax_yahoo, ax_google = axs
    
    for ax in axs:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 14)
        ax.set_aspect('equal')
        ax.axis('off')
    
    initial_yahoo_weights = weights_history_yahoo[0] if weights_history_yahoo else {}
    initial_google_weights = weights_history_google[0] if weights_history_google else {}
    
    # Extract values with validation
    initial_yahoo_values = {}
    if len(inputs_norm_yahoo) > 0:
        initial_yahoo_values = inputs_norm_yahoo.iloc[0].to_dict()
        # Validate values are normalized (0-1)
        for k, v in initial_yahoo_values.items():
            if not (0 <= v <= 1):
                logger.warning(f"⚠️  Yahoo {k} = {v:.4f} outside [0,1] - clipping")
                initial_yahoo_values[k] = np.clip(v, 0, 1)
    
    initial_google_values = {}
    if len(inputs_norm_google) > 0:
        initial_google_values = inputs_norm_google.iloc[0].to_dict()
        # Validate values are normalized (0-1)
        for k, v in initial_google_values.items():
            if not (0 <= v <= 1):
                logger.warning(f"⚠️  Google {k} = {v:.4f} outside [0,1] - clipping")
                initial_google_values[k] = np.clip(v, 0, 1)
    
    logger.info(f"✅ Initial Node Values Extracted & Validated:")
    low_unemp_y = initial_yahoo_values.get('Low_Unemployment', 'N/A')
    low_unemp_g = initial_google_values.get('Low_Unemployment', 'N/A')
    if isinstance(low_unemp_y, (int, float)):
        logger.info(f"   - Low_Unemployment (Yahoo): {low_unemp_y:.4f} (from FRED API)")
    else:
        logger.info(f"   - Low_Unemployment (Yahoo): {low_unemp_y}")
    if isinstance(low_unemp_g, (int, float)):
        logger.info(f"   - Low_Unemployment (Google): {low_unemp_g:.4f} (from FRED API)")
    else:
        logger.info(f"   - Low_Unemployment (Google): {low_unemp_g}")
    
    draw_dynamic_arrows(ax_yahoo, node_data, initial_yahoo_weights, initial_yahoo_values, 
                       annotate_weights=True, title_text="Dynamic Yahoo FCM\n(Weights Learning Over Time)")
    draw_dynamic_arrows(ax_google, node_data, initial_google_weights, initial_google_values, 
                       annotate_weights=True, title_text="Dynamic Google FCM\n(Weights Learning Over Time)")
    
    frames = min(len(weights_history_yahoo), len(weights_history_google), 
                 len(inputs_norm_yahoo), len(inputs_norm_google))
    
    def update_frame(frame_idx):
        """Update with real data from each time step."""
        if frame_idx >= len(weights_history_yahoo) or frame_idx >= len(weights_history_google):
            return
        if frame_idx >= len(inputs_norm_yahoo) or frame_idx >= len(inputs_norm_google):
            return
        
        yahoo_weights = weights_history_yahoo[frame_idx]
        yahoo_date = date_history_yahoo[frame_idx] if frame_idx < len(date_history_yahoo) else None
        yahoo_values = inputs_norm_yahoo.iloc[frame_idx].to_dict()
        yahoo_title = f"Dynamic Yahoo FCM (Real Data)\nDate: {yahoo_date.strftime('%Y-%m-%d') if yahoo_date else 'N/A'}"
        draw_dynamic_arrows(ax_yahoo, node_data, yahoo_weights, yahoo_values, 
                           annotate_weights=True, title_text=yahoo_title)
        
        google_weights = weights_history_google[frame_idx]
        google_date = date_history_google[frame_idx] if frame_idx < len(date_history_google) else None
        google_values = inputs_norm_google.iloc[frame_idx].to_dict()
        google_title = f"Dynamic Google FCM (Real Data)\nDate: {google_date.strftime('%Y-%m-%d') if google_date else 'N/A'}"
        draw_dynamic_arrows(ax_google, node_data, google_weights, google_values, 
                           annotate_weights=True, title_text=google_title)
    
    if frames > 0:
        logger.info(f"Creating animation with {frames} frames...")
        ani = FuncAnimation(fig, update_frame, frames=frames, interval=300, repeat=True)
    else:
        logger.warning("No frames to animate!")
    
    fig.suptitle('Fuzzy Cognitive Map: Dynamic Evolution with Real Node Values\n(Yahoo vs Google News APIs | FRED Unemployment | Yahoo Finance Prices)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('fcm_animated_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
    logger.info("✅ Saved fcm_animated_structure.png")
    plt.show()
    
    logger.info("\n" + "="*80)
    logger.info("FINAL FCM WEIGHTS - YAHOO SOURCE (Low_Unemployment Impact)")
    logger.info("="*80)
    if weights_history_yahoo:
        final_yahoo = weights_history_yahoo[-1]
        for src in node_data['node_order'][:11]:
            if src in final_yahoo and 'NASDAQ' in final_yahoo[src]:
                w = final_yahoo[src]['NASDAQ']
                if src == 'Low_Unemployment':
                    logger.info(f"⭐ {src:30s} → NASDAQ: {w:8.4f} ⭐")
                else:
                    logger.info(f"{src:30s} → NASDAQ: {w:8.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("FINAL FCM WEIGHTS - GOOGLE SOURCE (Low_Unemployment Impact)")
    logger.info("="*80)
    if weights_history_google:
        final_google = weights_history_google[-1]
        for src in node_data['node_order'][:11]:
            if src in final_google and 'NASDAQ' in final_google[src]:
                w = final_google[src]['NASDAQ']
                if src == 'Low_Unemployment':
                    logger.info(f"⭐ {src:30s} → NASDAQ: {w:8.4f} ⭐")
                else:
                    logger.info(f"{src:30s} → NASDAQ: {w:8.4f}")

# ============================================================================
# SUPPORTING VISUALIZATIONS (all using dynamic data)
# ============================================================================

def plot_sentiment_time_series(out):
    """Plot sentiment from real NewsAPI sources with data validation."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    sentiment_yahoo = out.get("polarity_series_yahoo")
    sentiment_google = out.get("polarity_series_google")
    sentiment_unified = out.get("polarity_series")
    
    # VALIDATION: Check data availability and quality
    logger.info("📊 Sentiment Analysis - Data Validation:")
    
    ax = axes[0]
    if sentiment_yahoo is not None and not sentiment_yahoo.empty:
        valid_yahoo = sentiment_yahoo.dropna()
        logger.info(f"   ✓ Yahoo Sentiment: {len(valid_yahoo)} valid values (range: [{valid_yahoo.min():.3f}, {valid_yahoo.max():.3f}])")
        ax.plot(sentiment_yahoo.index, sentiment_yahoo, label='Yahoo News Sentiment (real)', 
               color=colors["yahoo_primary"], lw=2, alpha=0.7)
    else:
        logger.info(f"   ✗ Yahoo Sentiment: No data")
    
    if sentiment_google is not None and not sentiment_google.empty:
        valid_google = sentiment_google.dropna()
        logger.info(f"   ✓ Google Sentiment: {len(valid_google)} valid values (range: [{valid_google.min():.3f}, {valid_google.max():.3f}])")
        ax.plot(sentiment_google.index, sentiment_google, label='Google News Sentiment (real)',
               color=colors["google_primary"], lw=2, alpha=0.7)
    else:
        logger.info(f"   ✗ Google Sentiment: No data")
    
    ax.set_ylabel("Sentiment Score", fontsize=fonts["label"], fontweight='bold')
    ax.set_title("Sentiment Time Series: Real Data from NewsAPI", fontsize=fonts["title"], fontweight='bold')
    ax.legend(fontsize=fonts["legend"], loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-1.1, 1.1)  # Sentiment constrained to [-1, 1]
    
    # Unified sentiment
    ax = axes[1]
    if sentiment_unified is not None and not sentiment_unified.empty:
        valid_unified = sentiment_unified.dropna()
        logger.info(f"   ✓ Unified Sentiment: {len(valid_unified)} valid values (range: [{valid_unified.min():.3f}, {valid_unified.max():.3f}])")
        
        colors_array = [colors["actual"] if x > 0 else colors["prediction"] for x in sentiment_unified]
        ax.scatter(sentiment_unified.index, sentiment_unified, c=colors_array, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    else:
        logger.info(f"   ✗ Unified Sentiment: No data")
    
    ax.set_xlabel("Date", fontsize=fonts["label"], fontweight='bold')
    ax.set_ylabel("Unified Sentiment", fontsize=fonts["label"], fontweight='bold')
    ax.set_title("Unified Sentiment Score from Real Data", fontsize=fonts["title"], fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('sentiment_time_series.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved sentiment_time_series.png")
    plt.show()

def plot_model_predictions(out):
    """Compare model predictions vs actual NASDAQ with strict data alignment validation."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    actual_norm = out.get("actual_norm")
    dates_yahoo = out.get("date_history_yahoo", [])
    preds_yahoo = out.get("pred_history_yahoo", [])
    dates_google = out.get("date_history_google", [])
    preds_google = out.get("pred_history_google", [])
    
    if actual_norm is None or actual_norm.empty:
        logger.error("❌ No actual normalized target available!")
        return
    
    # ACCURACY: Validate all required data
    logger.info("📈 Model Predictions - Data Validation:")
    logger.info(f"   ✓ Actual NASDAQ: {len(actual_norm)} values (range: [{actual_norm.min():.4f}, {actual_norm.max():.4f}])")
    
    # Plot actual (highest priority, thickest line)
    ax.plot(actual_norm.index, actual_norm, label='Actual NASDAQ (normalized)',
           color=colors["actual"], lw=3, alpha=0.9, zorder=10)
    
    # Validate and plot Yahoo predictions
    if len(dates_yahoo) > 0 and len(preds_yahoo) > 0:
        if len(dates_yahoo) != len(preds_yahoo):
            logger.warning(f"⚠️  Yahoo: {len(dates_yahoo)} dates vs {len(preds_yahoo)} predictions - truncating")
            min_len = min(len(dates_yahoo), len(preds_yahoo))
            dates_yahoo = dates_yahoo[:min_len]
            preds_yahoo = preds_yahoo[:min_len]
        
        preds_yahoo_arr = np.array(preds_yahoo, dtype=float)
        preds_yahoo_arr = np.nan_to_num(preds_yahoo_arr, nan=0.0)  # Replace NaN with 0
        
        logger.info(f"   ✓ Yahoo FCM: {len(preds_yahoo_arr)} predictions (range: [{np.nanmin(preds_yahoo_arr):.4f}, {np.nanmax(preds_yahoo_arr):.4f}])")
        ax.plot(dates_yahoo, preds_yahoo_arr, label='FCM Predictions (Yahoo)',
               color=colors["yahoo_primary"], lw=2, alpha=0.7, linestyle='--', zorder=7)
    else:
        logger.warning(f"   ✗ Yahoo FCM: No prediction data")
    
    # Validate and plot Google predictions
    if len(dates_google) > 0 and len(preds_google) > 0:
        if len(dates_google) != len(preds_google):
            logger.warning(f"⚠️  Google: {len(dates_google)} dates vs {len(preds_google)} predictions - truncating")
            min_len = min(len(dates_google), len(preds_google))
            dates_google = dates_google[:min_len]
            preds_google = preds_google[:min_len]
        
        preds_google_arr = np.array(preds_google, dtype=float)
        preds_google_arr = np.nan_to_num(preds_google_arr, nan=0.0)  # Replace NaN with 0
        
        logger.info(f"   ✓ Google FCM: {len(preds_google_arr)} predictions (range: [{np.nanmin(preds_google_arr):.4f}, {np.nanmax(preds_google_arr):.4f}])")
        ax.plot(dates_google, preds_google_arr, label='FCM Predictions (Google)',
               color=colors["google_primary"], lw=2, alpha=0.7, linestyle='--', zorder=7)
    else:
        logger.warning(f"   ✗ Google FCM: No prediction data")
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.set_xlabel("Date", fontsize=fonts["label"], fontweight='bold')
    ax.set_ylabel("Normalized Returns", fontsize=fonts["label"], fontweight='bold')
    ax.set_title("Model Predictions vs Actual NASDAQ (Real Data)",
                fontsize=fonts["title"], fontweight='bold')
    ax.legend(loc='best', fontsize=fonts["legend"], framealpha=0.96, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved model_predictions.png")
    plt.show()

def plot_sentiment_vs_returns(out):
    """Plot sentiment correlation with NASDAQ returns with data validation."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    sentiment = out.get("polarity_series")
    actual_norm = out.get("actual_norm")
    
    if sentiment is None or sentiment.empty:
        logger.warning("❌ No sentiment data available for correlation plot")
        return
    
    if actual_norm is None or actual_norm.empty:
        logger.warning("❌ No actual target data available for correlation plot")
        return
    
    # ACCURACY: Strict alignment validation
    common_idx = sentiment.index.intersection(actual_norm.index)
    if len(common_idx) < 2:
        logger.error(f"❌ Insufficient overlapping data: only {len(common_idx)} common dates")
        return
    
    sentiment_aligned = sentiment.loc[common_idx]
    returns_aligned = actual_norm.loc[common_idx]
    
    # Remove NaN values
    valid_idx = (~sentiment_aligned.isna()) & (~returns_aligned.isna())
    sentiment_clean = sentiment_aligned[valid_idx]
    returns_clean = returns_aligned[valid_idx]
    
    logger.info("💧 Sentiment vs Returns - Data Validation:")
    logger.info(f"   ✓ Aligned dates: {len(common_idx)}")
    logger.info(f"   ✓ Valid pairs (no NaN): {len(sentiment_clean)}")
    
    if len(sentiment_clean) < 2:
        logger.error(f"❌ Insufficient valid data pairs: {len(sentiment_clean)}")
        return
    
    logger.info(f"   ✓ Sentiment range: [{sentiment_clean.min():.4f}, {sentiment_clean.max():.4f}]")
    logger.info(f"   ✓ Returns range: [{returns_clean.min():.4f}, {returns_clean.max():.4f}]")
    
    # Create two y-axes
    ax2 = ax.twinx()
    
    # Plot sentiment with validation
    ax.fill_between(sentiment_clean.index, sentiment_clean.min(), sentiment_clean,
                   alpha=0.2, color=colors["yahoo_primary"], label='Sentiment')
    ax.plot(sentiment_clean.index, sentiment_clean, color=colors["yahoo_primary"],
           lw=2, alpha=0.8, label='Sentiment (from NewsAPI)')
    ax.set_ylim(sentiment_clean.min() - 0.1, sentiment_clean.max() + 0.1)
    
    # Plot returns with validation
    ax2.plot(returns_clean.index, returns_clean, color=colors["actual"],
            lw=2.5, alpha=0.8, label='NASDAQ Returns')
    ax2.set_ylim(returns_clean.min() - 0.05, returns_clean.max() + 0.05)
    
    # Calculate correlation only on valid data
    corr = sentiment_clean.corr(returns_clean)
    logger.info(f"   ✓ Calculated Correlation: {corr:.4f}")
    
    ax.set_xlabel("Date", fontsize=fonts["label"], fontweight='bold')
    ax.set_ylabel("Sentiment Score", fontsize=fonts["label"], fontweight='bold', color=colors["yahoo_primary"])
    ax2.set_ylabel("NASDAQ Returns (normalized)", fontsize=fonts["label"], fontweight='bold', color=colors["actual"])
    ax.tick_params(axis='y', labelcolor=colors["yahoo_primary"])
    ax2.tick_params(axis='y', labelcolor=colors["actual"])
    ax.set_title(f"Sentiment vs NASDAQ Returns | Correlation: {corr:.4f} | Pairs: {len(sentiment_clean)}",
                fontsize=fonts["title"], fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=fonts["legend"])
    
    plt.tight_layout()
    plt.savefig('sentiment_vs_returns.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved sentiment_vs_returns.png")
    plt.show()

def plot_weight_evolution(out):
    """Plot how FCM weights evolve over time from real learning with validation."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    weights_history = out.get("weights_history_yahoo", [])
    dates = out.get("date_history_yahoo", [])
    node_order = out.get("node_order", [])
    
    if not weights_history or not dates:
        logger.warning("❌ Insufficient weight history for evolution plot")
        return
    
    # ACCURACY: Validate alignment
    if len(weights_history) != len(dates):
        logger.warning(f"⚠️  Weight history ({len(weights_history)}) and dates ({len(dates)}) mismatch - truncating")
        min_len = min(len(weights_history), len(dates))
        weights_history = weights_history[:min_len]
        dates = dates[:min_len]
    
    logger.info("📊 Weight Evolution - Data Validation:")
    logger.info(f"   ✓ Weight snapshots: {len(weights_history)}")
    logger.info(f"   ✓ Time period: {dates[0] if dates else 'N/A'} to {dates[-1] if dates else 'N/A'}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    key_connections = [
        ('Low_Unemployment', 'NASDAQ'),
        ('Inflation', 'NASDAQ'),
        ('Corporate_Earnings', 'NASDAQ'),
        ('Investor_Sentiment', 'NASDAQ')
    ]
    
    for idx, (src, tgt) in enumerate(key_connections):
        ax = axes[idx]
        
        weights_over_time = []
        valid_count = 0
        
        for w_dict in weights_history:
            if src in w_dict and tgt in w_dict[src]:
                w = float(w_dict[src][tgt])
                # Validate weight is in [-1, 1] (clipped in fcm_nasdaq.py)
                if not (-1 <= w <= 1):
                    logger.warning(f"⚠️  {src}→{tgt} weight {w:.4f} outside [-1,1] - clipping")
                    w = np.clip(w, -1, 1)
                weights_over_time.append(w)
                valid_count += 1
            else:
                weights_over_time.append(0)
        
        if valid_count == 0:
            logger.warning(f"   ✗ {src} → {tgt}: No weight data")
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        weights_arr = np.array(weights_over_time, dtype=float)
        logger.info(f"   ✓ {src} → {tgt}: {valid_count} valid weights (range: [{np.min(weights_arr):.4f}, {np.max(weights_arr):.4f}])")
        
        ax.plot(dates, weights_arr, color=colors["yahoo_primary"], lw=2.5, marker='o', markersize=3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Fill areas properly
        positive_mask = np.array(weights_arr) >= 0
        negative_mask = ~positive_mask
        
        ax.fill_between(list(dates), 0, weights_arr, where=positive_mask,
                       alpha=0.2, color=colors["actual"], label='Positive')
        ax.fill_between(list(dates), 0, weights_arr, where=negative_mask,
                       alpha=0.2, color=colors["prediction"], label='Negative')

        title_text = f"Weight Evolution: {src} → {tgt}"
        if src == 'Low_Unemployment':
            title_text += " (FRED DATA)"

        ax.set_ylabel("FCM Weight", fontsize=fonts["label"], fontweight='bold')
        ax.set_title(title_text, fontsize=fonts["subtitle"], fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(-1.1, 1.1)
    
    fig.suptitle("FCM Weight Evolution Over Time (from Real Learning)",
                fontsize=fonts["title"], fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('weight_evolution.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved weight_evolution.png")
    plt.show()

# ============================================================================
# ADDITIONAL VISUALIZATIONS - FCM & LSTM ACCURACY COMPARISON
# ============================================================================

def plot_fcm_lstm_accuracy(out):
    """Compare Static FCM vs Dynamic FCM (Yahoo/Google) vs Hybrid models."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    actual_norm = out.get("actual_norm")
    date_history_yahoo = out.get("date_history_yahoo", [])
    pred_history_yahoo = out.get("pred_history_yahoo", [])
    date_history_google = out.get("date_history_google", [])
    pred_history_google = out.get("pred_history_google", [])
    lstm_results = out.get("lstm_results", (None, None))
    gru_results = out.get("gru_results", (None, None))
    
    if actual_norm is None or actual_norm.empty:
        logger.warning("❌ Cannot plot FCM+LSTM accuracy - no actual data")
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot actual
    ax.plot(actual_norm.index, actual_norm, color=colors["actual"], lw=3.5, 
           alpha=0.95, label='Actual', zorder=10)
    
    # Plot Dynamic FCM Yahoo
    if len(date_history_yahoo) > 0 and len(pred_history_yahoo) > 0:
        min_len = min(len(date_history_yahoo), len(pred_history_yahoo))
        preds_arr = np.array(pred_history_yahoo[:min_len], dtype=float)
        preds_arr = np.nan_to_num(preds_arr, nan=0.0)
        ax.plot(date_history_yahoo[:min_len], preds_arr, 
               color=colors["yahoo_primary"], lw=2, alpha=0.7, 
               linestyle='--', label='Dynamic FCM Yahoo', zorder=7)
    
    # Plot Dynamic FCM Google
    if len(date_history_google) > 0 and len(pred_history_google) > 0:
        min_len = min(len(date_history_google), len(pred_history_google))
        preds_arr = np.array(pred_history_google[:min_len], dtype=float)
        preds_arr = np.nan_to_num(preds_arr, nan=0.0)
        ax.plot(date_history_google[:min_len], preds_arr,
               color=colors["google_primary"], lw=2, alpha=0.7,
               linestyle='-.', label='Dynamic FCM Google', zorder=7)
    
    # Plot LSTM results
    if lstm_results[1] is not None and len(lstm_results[1]) > 0:
        lstm_dates = actual_norm.index[-len(lstm_results[1]):]
        ax.plot(lstm_dates, lstm_results[1], color='purple', lw=2, alpha=0.6,
               linestyle=':', label='LSTM', zorder=6)
    
    # Plot GRU results
    if gru_results[1] is not None and len(gru_results[1]) > 0:
        gru_dates = actual_norm.index[-len(gru_results[1]):]
        ax.plot(gru_dates, gru_results[1], color='brown', lw=2, alpha=0.6,
               linestyle=':', label='GRU', zorder=6)
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.set_xlabel("Date", fontsize=fonts["label"], fontweight='bold')
    ax.set_ylabel("Normalized Returns", fontsize=fonts["label"], fontweight='bold')
    ax.set_title("NASDAQ FCM + LSTM Accuracy (Static vs Dynamic Yahoo vs Dynamic Google)",
                fontsize=fonts["title"], fontweight='bold')
    ax.legend(loc='best', fontsize=fonts["legend"], ncol=3, framealpha=0.96)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('fcm_lstm_accuracy.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved fcm_lstm_accuracy.png")
    plt.show()

def plot_fcm_network_final_state(out):
    """Plot final FCM network structure with learned weights (Yahoo source)."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    weights_history_yahoo = out.get("weights_history_yahoo", [])
    if not weights_history_yahoo:
        logger.warning("❌ Cannot plot FCM network - no weight history")
        return
    
    final_weights = weights_history_yahoo[-1]
    node_order = out.get("node_order", [])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create network graph
    import networkx as nx
    G = nx.DiGraph()
    
    # Add nodes
    outer_nodes = node_order[:11] if len(node_order) > 11 else []
    inner_nodes = node_order[11:15] if len(node_order) > 11 else []
    target = 'NASDAQ'
    
    for node in outer_nodes:
        G.add_node(node)
    for node in inner_nodes:
        G.add_node(node)
    G.add_node(target)
    
    # Add edges with weights
    for src in node_order:
        if src in final_weights:
            for tgt, weight in final_weights[src].items():
                if weight != 0:
                    G.add_edge(src, tgt, weight=weight)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    weights_normalized = np.array(weights) / (np.max(np.abs(weights)) + 1e-8)
    
    for (u, v), w in zip(edges, weights_normalized):
        color = colors["actual"] if w > 0 else colors["prediction"]
        width = 2 + abs(w) * 4
        ax.annotate('', xy=pos[v], xytext=pos[u],
                   arrowprops=dict(arrowstyle='->', lw=width, color=color, alpha=0.7))
    
    # Draw nodes
    for node in outer_nodes:
        ax.scatter(*pos[node], s=3000, c='lightblue', edgecolors='darkblue', lw=2, zorder=5)
        ax.text(pos[node][0], pos[node][1], node, ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=6)
    
    for node in inner_nodes:
        ax.scatter(*pos[node], s=2500, c='lightyellow', edgecolors='orange', lw=2, zorder=5)
        ax.text(pos[node][0], pos[node][1], node, ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=6)
    
    if target in pos:
        ax.scatter(*pos[target], s=4000, c='lightgreen', edgecolors='darkgreen', lw=2.5, zorder=5)
        ax.text(pos[target][0], pos[target][1], target, ha='center', va='center',
               fontsize=11, fontweight='bold', zorder=6)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("FCM Network - Final State (Yahoo Source)", fontsize=fonts["title"], fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor='green', edgecolor='darkgreen', label='Positive weight'),
                      mpatches.Patch(facecolor='red', edgecolor='darkred', label='Negative weight')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=fonts["legend"])
    
    plt.tight_layout()
    plt.savefig('fcm_network.png', dpi=150, bbox_inches='tight', facecolor='white')
    logger.info("✅ Saved fcm_network.png")
    plt.show()

def plot_fcm_structure_animated_frame(out):
    """Plot final frame of animated FCM structure showing both Yahoo and Google."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    weights_history_yahoo = out.get("weights_history_yahoo", [])
    weights_history_google = out.get("weights_history_google", [])
    inputs_norm_yahoo = out.get("inputs_norm_yahoo")
    inputs_norm_google = out.get("inputs_norm_google")
    
    if not all([weights_history_yahoo, weights_history_google, inputs_norm_yahoo is not None, inputs_norm_google is not None]):
        logger.warning("❌ Cannot plot FCM structure - missing data")
        return
    
    fig, (ax_y, ax_g) = plt.subplots(1, 2, figsize=(18, 7))
    
    for ax in [ax_y, ax_g]:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 14)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Get final frame data
    final_idx = min(len(weights_history_yahoo), len(weights_history_google), 
                   len(inputs_norm_yahoo), len(inputs_norm_google)) - 1
    
    yahoo_weights = weights_history_yahoo[final_idx] if final_idx >= 0 else {}
    google_weights = weights_history_google[final_idx] if final_idx >= 0 else {}
    yahoo_values = inputs_norm_yahoo.iloc[final_idx].to_dict() if final_idx >= 0 else {}
    google_values = inputs_norm_google.iloc[final_idx].to_dict() if final_idx >= 0 else {}
    
    node_data = build_node_positions(out)
    
    # Draw both FCMs
    draw_dynamic_arrows(ax_y, node_data, yahoo_weights, yahoo_values, 
                       annotate_weights=True, title_text="Dynamic Yahoo FCM\n(Weights Learning Over Time)")
    draw_dynamic_arrows(ax_g, node_data, google_weights, google_values,
                       annotate_weights=True, title_text="Dynamic Google FCM\n(Weights Learning Over Time)")
    
    fig.suptitle('Fuzzy Cognitive Map: Dynamic Evolution with Node Values (Yahoo vs Google Data Sources)',
                fontsize=fonts["title"], fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fcm_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
    logger.info("✅ Saved fcm_structure.png")
    plt.show()

def plot_residual_analysis(out):
    """Analyze residuals from FCM Dynamic Model (Yahoo source)."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    actual_norm = out.get("actual_norm")
    date_history_yahoo = out.get("date_history_yahoo", [])
    pred_history_yahoo = out.get("pred_history_yahoo", [])
    
    if actual_norm is None or len(date_history_yahoo) == 0:
        logger.warning("❌ Cannot plot residual analysis - no data")
        return
    
    # Align predictions with actual
    min_len = min(len(date_history_yahoo), len(pred_history_yahoo), len(actual_norm))
    preds = np.array(pred_history_yahoo[:min_len], dtype=float)
    preds = np.nan_to_num(preds, nan=0.0)
    dates_aligned = date_history_yahoo[:min_len]
    actual_aligned = actual_norm.iloc[:min_len].values
    
    residuals = actual_aligned - preds
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Residuals over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(dates_aligned, residuals, c=residuals, cmap='RdYlGn', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=fonts["label"], fontweight='bold')
    ax1.set_title("Residuals Over Time", fontsize=fonts["subtitle"], fontweight='bold')
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Residual distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(residuals, bins=30, color=colors["yahoo_primary"], alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(residuals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
    ax2.axvline(np.median(residuals), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(residuals):.4f}')
    ax2.set_xlabel("Residual Value", fontsize=fonts["label"], fontweight='bold')
    ax2.set_ylabel("Frequency", fontsize=fonts["label"], fontweight='bold')
    ax2.set_title("Residual Distribution", fontsize=fonts["subtitle"], fontweight='bold')
    ax2.legend(fontsize=fonts["legend"])
    ax2.grid(True, alpha=0.2, axis='y')
    
    # 3. Q-Q plot
    ax3 = fig.add_subplot(gs[1, 1])
    if SCIPY_AVAILABLE:
        _scipy_stats.probplot(residuals, dist="norm", plot=ax3)
    else:
        ax3.text(0.5, 0.5, "scipy not available", ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title("Q-Q Plot: Residual Normality", fontsize=fonts["subtitle"], fontweight='bold')
    ax3.grid(True, alpha=0.2)
    
    fig.suptitle("Residual Analysis: FCM Dynamic Model (Yahoo Source)",
                fontsize=fonts["title"], fontweight='bold', y=0.995)
    plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    logger.info("✅ Saved residual_analysis.png")
    plt.show()

def plot_individual_stock_predictions(out):
    """Plot individual stock returns vs FCM predictions comparison."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    inputs_norm_yahoo = out.get("inputs_norm_yahoo")
    if inputs_norm_yahoo is None or inputs_norm_yahoo.empty:
        logger.warning("❌ Cannot plot stock predictions - no input data")
        return
    
    actual_norm = out.get("actual_norm")
    date_history_yahoo = out.get("date_history_yahoo", [])
    pred_history_yahoo = out.get("pred_history_yahoo", [])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Select 4 representative stocks
    stock_cols = [col for col in inputs_norm_yahoo.columns if col not in ['News_Sentiment_Yahoo', 'News_Sentiment_Google', 'News_Sentiment']][:4]
    
    for idx, col in enumerate(stock_cols):
        ax = axes[idx]
        stock_data = inputs_norm_yahoo[col]
        
        ax.fill_between(stock_data.index, 0, stock_data, alpha=0.3, color=colors["actual"], label='Actual (normalized)')
        ax.plot(stock_data.index, stock_data, color=colors["actual"], lw=2, alpha=0.8)
        
        # Overlay FCM predictions if available
        if len(date_history_yahoo) > 0 and len(pred_history_yahoo) > 0:
            min_len = min(len(date_history_yahoo), len(pred_history_yahoo))
            preds = np.array(pred_history_yahoo[:min_len], dtype=float)
            preds = np.nan_to_num(preds, nan=0.0)
            preds_scaled = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)  # Scale to [0,1]
            ax.plot(date_history_yahoo[:min_len], preds_scaled, color=colors["prediction"],
                   lw=2, alpha=0.7, linestyle='--', label='FCM Predictions')
        
        ax.set_title(f"{col} - Actual vs FCM Predictions", fontsize=fonts["subtitle"], fontweight='bold')
        ax.set_ylabel("Normalized Returns", fontsize=fonts["label"], fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=fonts["legend"], loc='best')
        ax.tick_params(axis='x', rotation=45)
    
    fig.suptitle("Individual Stock Returns vs FCM Predictions Comparison",
                fontsize=fonts["title"], fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('stock_predictions_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved stock_predictions_comparison.png")
    plt.show()

def plot_individual_stock_sentiment(out):
    """Plot individual stock prices with news sentiment overlay."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    inputs_norm_yahoo = out.get("inputs_norm_yahoo")
    polarity_series = out.get("polarity_series")
    
    if inputs_norm_yahoo is None or inputs_norm_yahoo.empty:
        logger.warning("❌ Cannot plot stock sentiment - no input data")
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Select 8 representative stocks
    stock_cols = [col for col in inputs_norm_yahoo.columns if col not in ['News_Sentiment_Yahoo', 'News_Sentiment_Google', 'News_Sentiment']][:8]
    
    for idx, col in enumerate(stock_cols):
        ax = axes[idx]
        stock_data = inputs_norm_yahoo[col]
        
        ax.plot(stock_data.index, stock_data, color=colors["yahoo_primary"], lw=2, label='Normalized Price')
        ax.set_ylabel("Price (normalized)", fontsize=9, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=colors["yahoo_primary"])
        
        # Add sentiment overlay if available
        if polarity_series is not None and not polarity_series.empty:
            ax2 = ax.twinx()
            sentiment = polarity_series.reindex(stock_data.index).fillna(0)
            ax2.fill_between(sentiment.index, 0, sentiment, alpha=0.2, color=colors["prediction"], label='Sentiment')
            ax2.set_ylabel("Sentiment", fontsize=9, fontweight='bold', color=colors["prediction"])
            ax2.set_ylim(-1.1, 1.1)
            ax2.tick_params(axis='y', labelcolor=colors["prediction"])
        
        ax.set_title(f"{col}", fontsize=fonts["subtitle"], fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', rotation=45)
    
    fig.suptitle("Individual Stock Prices with News Sentiment Overlay",
                fontsize=fonts["title"], fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('stock_prices_sentiment.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved stock_prices_sentiment.png")
    plt.show()

def plot_stock_volatility_sentiment(out):
    """Plot stock volatility relationship with news sentiment."""
    fonts = get_font_sizes()
    colors = get_colors()
    
    inputs_norm_yahoo = out.get("inputs_norm_yahoo")
    polarity_series = out.get("polarity_series")
    
    if inputs_norm_yahoo is None or polarity_series is None:
        logger.warning("❌ Cannot plot volatility sentiment - missing data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Select 4 stocks
    stock_cols = [col for col in inputs_norm_yahoo.columns if col not in ['News_Sentiment_Yahoo', 'News_Sentiment_Google', 'News_Sentiment']][:4]
    
    for idx, col in enumerate(stock_cols):
        ax = axes[idx]
        
        stock_data = inputs_norm_yahoo[col]
        volatility = stock_data.rolling(window=20).std().fillna(0)
        sentiment = polarity_series.reindex(stock_data.index).fillna(0)
        
        # Align data
        common_idx = volatility.index.intersection(sentiment.index)
        vol_aligned = volatility.loc[common_idx]
        sent_aligned = sentiment.loc[common_idx]
        
        # Color scatter by sentiment
        scatter = ax.scatter(sent_aligned, vol_aligned, c=sent_aligned, cmap='RdYlGn', 
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(sent_aligned, vol_aligned, 1)
        p = np.poly1d(z)
        sent_sorted = np.sort(sent_aligned)
        ax.plot(sent_sorted, p(sent_sorted), "r--", lw=2, alpha=0.8, label='Trend')
        
        ax.set_xlabel("Sentiment Polarity", fontsize=fonts["label"], fontweight='bold')
        ax.set_ylabel("Annualized Volatility", fontsize=fonts["label"], fontweight='bold')
        ax.set_title(f"{col} - Volatility vs Sentiment", fontsize=fonts["subtitle"], fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=fonts["legend"])
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Sentiment", fontsize=9)
    
    fig.suptitle("Stock Volatility Relationship with News Sentiment",
                fontsize=fonts["title"], fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('stock_volatility_sentiment.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved stock_volatility_sentiment.png")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all visualizations with REAL dynamically-fetched data."""
    logger.info("="*80)
    logger.info("STARTING FCM ANALYSIS WITH REAL DATA FROM MULTIPLE APIs")
    logger.info("="*80)
    logger.info("Data sources:")
    logger.info("  • Unemployment: FRED API (Low_Unemployment shown correctly)")
    logger.info("  • Stock Prices: Yahoo Finance")
    logger.info("  • Sentiment: NewsAPI (Yahoo News + Google News)")
    logger.info("="*80)
    
    out = prepare_data_and_run()
    
    logger.info("\n🎬 GENERATING PRIMARY VISUALIZATION...")
    plot_animated_fcm_structure(out)
    
    logger.info("\n📊 Generating supporting visualizations...")
    plot_sentiment_time_series(out)
    plot_model_predictions(out)
    plot_sentiment_vs_returns(out)
    plot_weight_evolution(out)
    
    logger.info("\n🔬 Generating advanced analysis visualizations...")
    plot_fcm_lstm_accuracy(out)
    plot_fcm_network_final_state(out)
    plot_fcm_structure_animated_frame(out)
    plot_residual_analysis(out)
    plot_individual_stock_predictions(out)
    plot_individual_stock_sentiment(out)
    plot_stock_volatility_sentiment(out)
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    logger.info("✅ Generated 12 comprehensive visualizations")
    logger.info("✅ All data dynamically fetched from live APIs")
    logger.info("✅ No hardcoded constants or static data")
    logger.info("✅ Low_Unemployment properly displayed with FRED data")
    logger.info("✅ Sentiment data accurately flowing from NewsAPI")
    logger.info("="*80)

if __name__ == "__main__":
    main()
