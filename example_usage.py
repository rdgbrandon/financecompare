"""
Example usage scripts for the NASDAQ FCM
"""

from fcm_nasdaq import NASDAQ_FCM
from datetime import datetime, timedelta
import pandas as pd


def example_1_basic_simulation():
    """Example 1: Basic simulation with visualization"""
    print("=" * 60)
    print("Example 1: Basic FCM Simulation")
    print("=" * 60)

    fcm = NASDAQ_FCM()

    # Simulate for past 6 months
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    print(f"\nSimulating from {start_date} to {end_date}...")
    results = fcm.simulate_time_series(start_date, end_date)

    print("\nSample results (last 5 days):")
    print(results[['NASDAQ', 'Investor_Sentiment', 'Inflation', 'Corporate_Earnings']].tail())

    fcm.plot_results(results)


def example_2_custom_weights():
    """Example 2: FCM with custom weight adjustments"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Weight Adjustments")
    print("=" * 60)

    fcm = NASDAQ_FCM()

    # Modify weights to test different scenarios
    # Scenario: Stronger inflation impact
    idx_inflation = fcm.node_index['Inflation']
    idx_nasdaq = fcm.node_index['NASDAQ']
    fcm.W[idx_inflation, idx_nasdaq] = -0.8  # Increased from -0.4

    print("\nModified weight: Inflation -> NASDAQ = -0.8 (was -0.4)")
    print("This simulates a scenario where inflation has stronger negative impact on NASDAQ")

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    results = fcm.simulate_time_series(start_date, end_date, save_results=False)

    print("\nAverage NASDAQ prediction:", results['NASDAQ'].mean())
    print("Average Inflation level:", results['Inflation'].mean())


def example_3_single_point_analysis():
    """Example 3: Analyze FCM for a single point in time"""
    print("\n" + "=" * 60)
    print("Example 3: Single Time Point Analysis")
    print("=" * 60)

    fcm = NASDAQ_FCM()

    # Define a specific economic scenario
    scenario = {
        'Large_Deficits': 0.7,
        'Foreign_Demand': 0.5,
        'Low_Unemployment': 0.8,
        'Supply_Increase': 0.4,
        'Energy_Price_Increase': 0.6,
        'Consumer_Demand': 0.7,
        'Business_Investment': 0.6,
        'Market_Volatility': 0.3,  # Low volatility
        'Equity_Inflows': 0.8
    }

    print("\nInput scenario:")
    for node, value in scenario.items():
        print(f"  {node}: {value:.2f}")

    # Compute FCM state
    result = fcm.compute_state(scenario)

    print("\nComputed inner nodes:")
    print(f"  Monetary Policy & Interest Rates: {result['Monetary_Policy_Interest_Rates']:.3f}")
    print(f"  Inflation: {result['Inflation']:.3f}")
    print(f"  Corporate Earnings: {result['Corporate_Earnings']:.3f}")
    print(f"  Investor Sentiment: {result['Investor_Sentiment']:.3f}")
    print(f"  NASDAQ Prediction: {result['NASDAQ']:.3f}")


def example_4_scenario_comparison():
    """Example 4: Compare different economic scenarios"""
    print("\n" + "=" * 60)
    print("Example 4: Scenario Comparison")
    print("=" * 60)

    fcm = NASDAQ_FCM()

    # Scenario 1: Bullish economy
    bullish = {
        'Large_Deficits': 0.3,
        'Foreign_Demand': 0.8,
        'Low_Unemployment': 0.9,
        'Supply_Increase': 0.7,
        'Energy_Price_Increase': 0.3,
        'Consumer_Demand': 0.8,
        'Business_Investment': 0.8,
        'Market_Volatility': 0.2,
        'Equity_Inflows': 0.9
    }

    # Scenario 2: Bearish economy
    bearish = {
        'Large_Deficits': 0.8,
        'Foreign_Demand': 0.3,
        'Low_Unemployment': 0.4,
        'Supply_Increase': 0.3,
        'Energy_Price_Increase': 0.8,
        'Consumer_Demand': 0.3,
        'Business_Investment': 0.3,
        'Market_Volatility': 0.8,
        'Equity_Inflows': 0.2
    }

    # Scenario 3: Mixed signals
    mixed = {
        'Large_Deficits': 0.5,
        'Foreign_Demand': 0.5,
        'Low_Unemployment': 0.7,
        'Supply_Increase': 0.5,
        'Energy_Price_Increase': 0.6,
        'Consumer_Demand': 0.6,
        'Business_Investment': 0.5,
        'Market_Volatility': 0.5,
        'Equity_Inflows': 0.5
    }

    scenarios = {
        'Bullish': bullish,
        'Bearish': bearish,
        'Mixed': mixed
    }

    results_comparison = []

    for name, scenario in scenarios.items():
        result = fcm.compute_state(scenario)
        results_comparison.append({
            'Scenario': name,
            'NASDAQ': result['NASDAQ'],
            'Investor_Sentiment': result['Investor_Sentiment'],
            'Corporate_Earnings': result['Corporate_Earnings'],
            'Inflation': result['Inflation'],
            'Monetary_Policy_IR': result['Monetary_Policy_Interest_Rates']
        })

    df_comparison = pd.DataFrame(results_comparison)
    print("\nScenario Comparison Results:")
    print(df_comparison.to_string(index=False))

    # Visualize comparison
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(scenarios))
    width = 0.15

    metrics = ['NASDAQ', 'Investor_Sentiment', 'Corporate_Earnings', 'Inflation', 'Monetary_Policy_IR']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, metric in enumerate(metrics):
        values = df_comparison[metric].values
        ax.bar([xi + i * width for xi in x], values, width, label=metric.replace('_', ' '), color=colors[i])

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Value')
    ax.set_title('FCM Predictions Across Different Economic Scenarios')
    ax.set_xticks([xi + width * 2 for xi in x])
    ax.set_xticklabels(scenarios.keys())
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison chart saved to scenario_comparison.png")
    plt.show()


def example_5_sensitivity_analysis():
    """Example 5: Sensitivity analysis - how changes in one factor affect NASDAQ"""
    print("\n" + "=" * 60)
    print("Example 5: Sensitivity Analysis")
    print("=" * 60)

    fcm = NASDAQ_FCM()

    # Base scenario (neutral)
    base_scenario = {
        'Large_Deficits': 0.5,
        'Foreign_Demand': 0.5,
        'Low_Unemployment': 0.5,
        'Supply_Increase': 0.5,
        'Energy_Price_Increase': 0.5,
        'Consumer_Demand': 0.5,
        'Business_Investment': 0.5,
        'Market_Volatility': 0.5,
        'Equity_Inflows': 0.5
    }

    # Test sensitivity to Energy Price
    print("\nSensitivity to Energy Price Increase:")
    print("-" * 40)

    energy_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    nasdaq_predictions = []

    for energy in energy_values:
        scenario = base_scenario.copy()
        scenario['Energy_Price_Increase'] = energy
        result = fcm.compute_state(scenario)
        nasdaq_predictions.append(result['NASDAQ'])
        print(f"Energy Price = {energy:.1f} → NASDAQ = {result['NASDAQ']:.3f}")

    # Plot sensitivity
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Test multiple factors
    factors_to_test = ['Energy_Price_Increase', 'Equity_Inflows', 'Market_Volatility', 'Consumer_Demand']

    for idx, factor in enumerate(factors_to_test):
        ax = axes[idx // 2, idx % 2]
        factor_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        nasdaq_results = []

        for val in factor_values:
            scenario = base_scenario.copy()
            scenario[factor] = val
            result = fcm.compute_state(scenario)
            nasdaq_results.append(result['NASDAQ'])

        ax.plot(factor_values, nasdaq_results, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel(f'{factor.replace("_", " ")} Value')
        ax.set_ylabel('NASDAQ Prediction')
        ax.set_title(f'Sensitivity to {factor.replace("_", " ")}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSensitivity analysis chart saved to sensitivity_analysis.png")
    plt.show()


if __name__ == "__main__":
    # Run all examples
    print("NASDAQ FCM - Example Usage Scripts")
    print("=" * 60)

    # Uncomment the examples you want to run:

    example_1_basic_simulation()
    # example_2_custom_weights()
    # example_3_single_point_analysis()
    # example_4_scenario_comparison()
    # example_5_sensitivity_analysis()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
