import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

google = yf.download('GOOGL', start=start_date, end=end_date, progress=False)
apple = yf.download('AAPL', start=start_date, end=end_date, progress=False)
sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)


google['Change'] = google['Close'].diff()
apple['Change'] = apple['Close'].diff()
sp500['Change'] = sp500['Close'].diff()

google['Direction'] = google['Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
apple['Direction'] = apple['Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
sp500['Direction'] = sp500['Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

comparison = pd.DataFrame({
    'Google': google['Direction'],
    'Apple': apple['Direction'],
    'SP500': sp500['Direction']
})

comparison = comparison.dropna()

stocks = ['Google', 'Apple', 'SP500']

matrix = np.zeros((3, 3))

for i, stock1 in enumerate(stocks):
    for j, stock2 in enumerate(stocks):
        if i == j:
            matrix[i][j] = 0.0
        else:
            matching = (comparison[stock1] == comparison[stock2]).sum()
            total = len(comparison)
            match_percentage = (matching / total) * 100
            matrix[i][j] = (match_percentage - 50) / 50


print("=== Stock Direction Comparison ===")
print(f"Total trading days analyzed: {len(comparison)}\n")


for i, stock1 in enumerate(stocks):
    for j, stock2 in enumerate(stocks):
        if i < j:  
            matching = (comparison[stock1] == comparison[stock2]).sum()
            match_percentage = (matching / len(comparison)) * 100
            print(f"{stock1} vs {stock2}:")
            print(f"  Matching days: {matching}/{len(comparison)} ({match_percentage:.2f}%)")

print("\n=== Correlation Matrix ===")
print("Formula: (match_percentage - 50) / 50\n")

matrix_df = pd.DataFrame(matrix, index=stocks, columns=stocks)
print(matrix_df.to_string())

print("\nMatrix values:")
for i, stock1 in enumerate(stocks):
    for j, stock2 in enumerate(stocks):
        if i != j:
            print(f"  [{stock1}, {stock2}]: {matrix[i][j]:.4f}")
