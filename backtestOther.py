import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to backtest a given pair of EMAs with different portfolio allocations
def backtest_ema(short_window, long_window, allocation_percentage):
    df = btc_df.copy()
    df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()

    df['Position'] = np.where(df['EMA_Short'] > df['EMA_Long'], 1, -1)
    df['Position'] = df['Position'].shift()
    df['Signal'] = df['Position'].diff()

    df['Daily_Return'] = df['Close'].pct_change()
    df['Portfolio_Balance'] = initial_balance * 1.0  # Ensure this is float
    df['Allocated_Balance'] = df['Portfolio_Balance'] * (allocation_percentage / 100)

    for i in range(1, len(df)):
        position_change = df['Daily_Return'].iloc[i] * df['Position'].iloc[i] * (allocation_percentage / 100)
        df.loc[df.index[i], 'Portfolio_Balance'] = df.loc[df.index[i-1], 'Portfolio_Balance'] * (1 + position_change)

    final_balance = df['Portfolio_Balance'].iloc[-1]
    profit_loss = final_balance - initial_balance
    percent_profit_loss = (profit_loss / initial_balance) * 100

    return short_window, long_window, allocation_percentage, percent_profit_loss, final_balance

# Fetch historical data for Bitcoin
ticker_data = yf.Ticker('BTC-USD')
btc_df = ticker_data.history(period='1h', start='2020-01-01', end='2023-01-01')

# Set initial balance
initial_balance = 10000.0  # Ensure this is float

# Define range of MA windows
ma_range = range(5, 105)  # Change as needed for different ranges

# Generate unique combinations of MAs
ma_combinations = list(combinations(ma_range, 2))

# Define allocation percentages
allocation_percentages = [10, 25, 50, 100]

# Store results
results = []

# Total combinations for progress tracking
total_combinations = len(ma_combinations) * len(allocation_percentages)

# Multi-threaded execution
with ThreadPoolExecutor(max_workers=16) as executor:
    future_to_combination = {
        executor.submit(backtest_ema, short_window, long_window, allocation_percentage): (short_window, long_window, allocation_percentage)
        for short_window, long_window in ma_combinations
        for allocation_percentage in allocation_percentages
    }

    for idx, future in enumerate(as_completed(future_to_combination)):
        short_window, long_window, allocation_percentage = future_to_combination[future]
        try:
            result = future.result()
            results.append(result)
            # Log progress
            print(f"Tested EMA {short_window} vs EMA {long_window} with {allocation_percentage}% allocation: ROI {result[3]:.2f}%. {total_combinations - (idx + 1)} combinations left.")
        except Exception as exc:
            print(f"EMA {short_window} vs EMA {long_window} with {allocation_percentage}% allocation generated an exception: {exc}")

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['Short_Window', 'Long_Window', 'Allocation_Percentage', 'ROI', 'Final_Balance'])

# Sort results by ROI
sorted_results_df = results_df.sort_values(by='ROI', ascending=False)

# Log top 50 results
top_50_results = sorted_results_df.head(50)
print("Top 50 EMA Combinations by ROI:")
print(top_50_results)

# Display top 10 results
print("Top 100 EMA Combinations by ROI:")
print(sorted_results_df.head(100))

# Plot the top results
plt.figure(figsize=(14, 7))
markers = ['o', 's', 'D', '^']  # Different markers for each allocation
for i, allocation_percentage in enumerate(allocation_percentages):
    allocation_results = sorted_results_df[sorted_results_df['Allocation_Percentage'] == allocation_percentage].head(10)
    plt.plot(allocation_results['Short_Window'], allocation_results['ROI'], marker=markers[i], label=f'{allocation_percentage}% Allocation')
    for j in range(len(allocation_results)):
        plt.text(allocation_results['Short_Window'].iloc[j], allocation_results['ROI'].iloc[j], f"${allocation_results['Final_Balance'].iloc[j]:.2f}")

plt.xlabel('EMA Short Window')
plt.ylabel('ROI (%)')
plt.title('Top 100 EMA Combinations by ROI for Different Allocations')
plt.legend()
plt.grid(True)
plt.show()
