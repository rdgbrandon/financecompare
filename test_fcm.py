"""
Simple test script for NASDAQ FCM
"""

from fcm_nasdaq import NASDAQ_FCM
from datetime import datetime, timedelta

# Create FCM instance
print("Creating NASDAQ FCM...")
fcm = NASDAQ_FCM()

# Visualize weight matrix
print("\nGenerating weight matrix visualization...")
fcm.visualize_weights()

# Use historical data from the past year (2024)
end_date = '2024-12-31'
start_date = '2024-01-01'

print(f"\n{'='*60}")
print(f"Running FCM simulation from {start_date} to {end_date}")
print(f"{'='*60}\n")

try:
    results = fcm.simulate_time_series(start_date, end_date)

    print("\n" + "="*60)
    print("Simulation completed successfully!")
    print("="*60)

    print("\nSample Results (First 5 days):")
    print(results[['NASDAQ', 'Investor_Sentiment', 'Inflation', 'Corporate_Earnings']].head())

    print("\nSample Results (Last 5 days):")
    print(results[['NASDAQ', 'Investor_Sentiment', 'Inflation', 'Corporate_Earnings']].tail())

    print(f"\nResults saved to: fcm_nasdaq_results.csv")

    # Plot results
    print("\nGenerating plots...")
    fcm.plot_results(results)

    print("\n" + "="*60)
    print("All visualizations saved!")
    print("="*60)
    print("\nGenerated files:")
    print("  - fcm_weight_matrix.png")
    print("  - fcm_simulation_results.png")
    print("  - fcm_nasdaq_results.csv")

except Exception as e:
    print(f"\nError during simulation: {e}")
    import traceback
    traceback.print_exc()
