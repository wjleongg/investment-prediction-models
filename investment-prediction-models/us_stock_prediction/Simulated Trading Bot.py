import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

#Set up and load input Data
train_df = pd.read_csv(r'Insert Training File Path (xlsx)')
test_df = pd.read_csv(r'Insert Testing File Path (xlsx)')

#Initialize an empty DataFrame for trade logging
trade_log_df = pd.DataFrame(columns=['Trade Number', 'Symbol', 'Entry Price', 'Size of Position','Predicted Return', 'Exit Price', 'Profit & Loss'])

#Set up output directory
output_file = r"Insert Output File Location (xlsx)"
trade_log_df.to_excel(output_file, index=False)
print(f"Trade log saved to {output_file}")

# Ensure Date column is datetime
train_df['Date'] = pd.to_datetime(train_df['Date'], dayfirst=True)
test_df['Date'] = pd.to_datetime(test_df['Date'], dayfirst=True)

# Sort data
train_df.sort_values(['Symbol', 'Date'], inplace=True)
test_df.sort_values(['Symbol', 'Date'], inplace=True)

# Feature columns
features = ['RSI_14D', 'BB_Upper_Band', 'BB_Lower_Band', 'STOK', 'STOD', 'plusDI', 'minusDI', 'ADX', 'Money_Flow_Index']

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_test = scaler.transform(test_df[features])

# Learning Condition
train_df['Future_Return'] = train_df.groupby('Symbol')['Close'].shift(-100) / train_df['Close'] - 1
y_train = train_df['Future_Return'].fillna(0)

# Train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

#Show feature importance
feature_importance = model.feature_importances_

# Predict returns
test_df['Predicted_Return'] = model.predict(X_test)

# Portfolio simulation parameters
portfolio_value = 100000
max_position = 0.20 * portfolio_value
#max_trades = 0 # Remove Comment to Set limit on number of trades
trade_log = []
open_positions = {}

# Simulate trading
for symbol, stock_data in test_df.groupby('Symbol'):
    stock_data = stock_data.sort_values('Date')
    open_trade = None

    for i, row in stock_data.iterrows():
        current_date = row['Date']
        entry_price = row['Close']
        predicted_return = row['Predicted_Return']

        # Check if we should enter a trade (Based on Technical Indicators)
        if symbol not in open_positions and predicted_return > 0:  #Modify based on Strategy & Certainty of Profits (Low ~ 0 to 0.05, High ~ > 0.1)
            position_size = min(max_position, portfolio_value * 0.20)

            # Open trade
            open_positions[symbol] = {
                'Trade Number': len(trade_log) + 1,
                'Symbol': symbol,
                'Entry Date': current_date,
                'Entry Price': entry_price,
                'Size of Position': position_size,
                'Predicted Return': predicted_return,
                'Exit Date': None,
                'Exit Price': None,
                'Profit & Loss': 0
            }
        
        # If already in a trade, look for exit signals (Based on Technical Indicators)
        elif symbol in open_positions:
            entry_price = open_positions[symbol]['Entry Price']
            entry_date = open_positions[symbol]['Entry Date']

            # Identify exit signal (e.g., downtrend, overbought conditions)
            exit_condition = row['Predicted_Return'] < 0  #Modify based on Strategy & Risk Appetite (Low ~ -0 to -0.05, High ~ > -0.1)

            if exit_condition:
                exit_price = row['Close']
                exit_date = current_date
                profit_loss = (exit_price - entry_price) * (open_positions[symbol]['Size of Position'] / entry_price)

                #Update portfolio value after trade closure
                portfolio_value += profit_loss

                # Finalize trade
                open_positions[symbol].update({
                    'Exit Date': exit_date,
                    'Exit Price': exit_price,
                    'Profit & Loss': profit_loss
                })

                trade_log.append(open_positions[symbol])
                del open_positions[symbol]

# Ensure all trades are exited on the last test day
final_test_date = test_df['Date'].max()

for symbol, trade in open_positions.items():
    last_price = test_df[(test_df['Symbol'] == symbol) & (test_df['Date'] == final_test_date)]['Close']
    if not last_price.empty:
        trade['Exit Price'] = last_price.values[0]
        trade['Exit Date'] = final_test_date
        trade['Profit & Loss'] = (trade['Exit Price'] - trade['Entry Price']) * (trade['Size of Position'] / trade['Entry Price'])

    trade_log.append(trade)

# Final calculations
portfolio_value += sum(trade['Profit & Loss'] for trade in trade_log)
win_trades = [trade for trade in trade_log if trade['Profit & Loss'] > 0]
loss_trades = [trade for trade in trade_log if trade['Profit & Loss'] <= 0]

# Output trade log
trade_log_df = pd.DataFrame(trade_log)
trade_log_df.to_excel(output_file, index=False)

# Summary statistics
print(f"Final Portfolio Value: ${portfolio_value:.2f}")
print(f"Total Profit: ${sum(trade['Profit & Loss'] for trade in trade_log):.2f}")
print(f"Win/Loss Ratio: {len(win_trades)} / {len(loss_trades)}")

best_trade = max(trade_log, key=lambda x: x['Profit & Loss'])
worst_trade = min(trade_log, key=lambda x: x['Profit & Loss'])

print("Best Performing Trade:", best_trade)
print("Worst Performing Trade:", worst_trade)

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.title("Feature Importance in Stock Prediction Model")
plt.xlabel("Technical Indicators")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()