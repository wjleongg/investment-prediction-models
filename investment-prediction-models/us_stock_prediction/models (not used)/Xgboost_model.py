import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

pd.set_option('display.max_columns', None)


# In[2]:


df = pd.read_csv("indicators_data_train.csv")

df['Date'] = pd.to_datetime(df['Date'])

df.head()


# In[3]:


dft = pd.read_csv('indicators_data_test.csv')
dft


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming df is your original DataFrame
# Define features and target
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI_14D', 
         'BB_Middle_Band', 'BB_Upper_Band', 'BB_Lower_Band', 
         'Aroon_Oscillator', 'STOK', 'STOD', 'VWAP', 'Momentum', 
         'OBV', 'TEMA', 'NATR', 'plusDI', 'minusDI', 'ADX', 
         'MACD', 'Money_Flow_Index', 'MIN_Volume', 'MAX_Volume']]
y = df['Close']

# Set the random seed
seed = 42

# Split the data into training, validation and test sets
seed = 42

## First, split the data into a training set and a temporary set using a 80-20 split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# not using 'Date' and 'Symbol' columns! (test)
X_test = dft[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI_14D', 'BB_Middle_Band', 'BB_Upper_Band', 'BB_Lower_Band', 'Aroon_Oscillator', 'STOK', 'STOD', 'VWAP', 'Momentum', 'OBV', 'TEMA', 'NATR', 'plusDI', 'minusDI', 'ADX', 'MACD', 'Money_Flow_Index', 'MIN_Volume', 'MAX_Volume']]
y_test = dft['Close']

# Create DataFrames for train, validation, and test sets
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['Close'] = y_train.values

valid_df = pd.DataFrame(X_val, columns=X.columns)
valid_df['Close'] = y_val.values

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['Close'] = y_test.values

# Print the shapes of the resulting datasets
print("Training set shape:", train_df.shape)
print("Validation set shape:", valid_df.shape)
print("Test set shape:", test_df.shape)

# Optionally, print the first few rows of each DataFrame
print("\nTraining set preview:")
print(train_df.head())

print("\nValidation set preview:")
print(valid_df.head())

print("\nTest set preview:")
print(test_df.head())


# In[5]:


pip install scikit-learn


# In[6]:


get_ipython().run_cell_magic('time', '', "parameters = {\n    'n_estimators': [100, 200, 300, 400],\n    'learning_rate': [0.001, 0.005, 0.01, 0.05],\n    'max_depth': [8, 10, 12, 15],\n    'gamma': [0.001, 0.005, 0.01, 0.02],\n    'random_state': [42]\n}\n\n# Initialize the model without 'eval_set' and 'verbose'\nmodel = xgb.XGBRegressor(objective='reg:squarederror')\n\n# Set up GridSearchCV\nclf = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=3)\n\n# Fit the model\nclf.fit(X_train, y_train)\n\n# Print the best parameters and score\nprint(f'Best params: {clf.best_params_}')\nprint(f'Best validation score = {clf.best_score_}')\n")


# In[7]:


get_ipython().run_cell_magic('time', '', "\n# Modify XGBoost initialization to use GPU\nmodel = xgb.XGBRegressor(tree_method='hist', objective='reg:squarederror')\n\n# Fit the model, passing 'eval_set' within the fit method\nmodel.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)\n\n# You can also use 'early_stopping_rounds' if you want early stopping\n# model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)\n")


# In[8]:


plot_importance(model);


# In[9]:


y_pred = model.predict(X_test)

# Print the first 5 actual and predicted values
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')


# In[10]:


get_ipython().run_cell_magic('time', '', "parameters = {\n    'n_estimators': [100, 200, 300, 400],\n    'learning_rate': [0.001, 0.005, 0.01, 0.05],\n    'max_depth': [8, 10, 12, 15],\n    'gamma': [0.001, 0.005, 0.01, 0.02],\n    'random_state': [42]\n}\n\n# Prepare the evaluation set\neval_set = [(X_train, y_train), (X_val, y_val)]  # Use X_val instead of X_valid\n\n# Initialize the model\nmodel = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)\n\n# Set up GridSearchCV\nclf = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=3)\n\n# Fit the model\nclf.fit(X_train, y_train)\n\n# Print the best parameters and score\nprint(f'Best params: {clf.best_params_}')\nprint(f'Best validation score = {clf.best_score_}')\n")


# #### a lower mse score will result in a better result
# 
# we have a low mse score, so this is the best model to predict stock prices

# In[11]:


print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')


# In[12]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go  # Add this line for 'go' to be defined
import pandas as pd
from sklearn.model_selection import train_test_split

predicted_prices = pd.DataFrame(X_test.copy())
predicted_prices['Close'] = y_pred  # Predictions

# Plotting
fig = make_subplots(rows=2, cols=1)

# Plot actual Close prices (truth) on the entire dataset
fig.add_trace(go.Scatter(x=df.Date, y=df.Close,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)

# Plot predicted Close prices on the test set
fig.add_trace(go.Scatter(x=predicted_prices.index,
                         y=predicted_prices.Close,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)

# Plot actual Close prices (truth) on the test set in the second subplot
fig.add_trace(go.Scatter(x=predicted_prices.index,
                         y=y_test,
                         name='Truth',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

# Plot predictions for the test set in the second subplot
fig.add_trace(go.Scatter(x=predicted_prices.index,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

# Show the plot
fig.show()

# Optionally, check the shapes of the train, validation, and test sets
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)


# In[13]:


# Assuming y_pred is defined somewhere above this block
# If not, you'll need to add: y_pred = clf.predict(X_test)

# Plotting and evaluation metrics
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Prepare data for plotting
predicted_prices = pd.DataFrame(X_test.copy())
predicted_prices['Close'] = y_pred

# Create subplots
fig = make_subplots(rows=2, cols=1)

# Plot actual Close prices on the entire dataset
fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Truth', marker_color='LightSkyBlue'), row=1, col=1)
# Plot predicted Close prices on the test set
fig.add_trace(go.Scatter(x=predicted_prices.index, y=predicted_prices.Close, name='Prediction', marker_color='MediumPurple'), row=1, col=1)
# Plot actual Close prices (truth) on the test set in the second subplot
fig.add_trace(go.Scatter(x=predicted_prices.index, y=y_test, name='Truth', marker_color='LightSkyBlue', showlegend=False), row=2, col=1)
# Plot predictions for the test set in the second subplot
fig.add_trace(go.Scatter(x=predicted_prices.index, y=y_pred, name='Prediction', marker_color='MediumPurple', showlegend=False), row=2, col=1)

# Display the plot
fig.show()

# Calculate and display evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Percentage Error: {mape}%')


# In[41]:


import pickle

# Initialize XGBoost model using CPU
model = XGBRegressor(objective='reg:squarederror', tree_method='hist')  # No need for device='cuda'

# Define a smaller parameter grid for faster optimization
parameters = {
    'n_estimators': [50, 100],   
    'max_depth': [3, 5],         
    'learning_rate': [0.1, 0.2], 
    'subsample': [0.8],          
    'colsample_bytree': [0.8],   
    'min_child_weight': [1, 5]   
}

# Set up GridSearchCV with cross-validation and verbose output (cv=3 for a good balance)
clf = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=3, verbose=2)

# Fit the model
clf.fit(X_train, y_train)

# Save the model to a file using pickle
with open('models/Xgboost_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Check if the file is able to load
with open('models/Xgboost_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)

# Print the best parameters found by GridSearchCV
print(clf_loaded.best_params_)


# In[42]:


# check if file is able to load

# Load the model from file
with open('models/Xgboost_model.pkl', 'rb') as f:
    clf = pickle.load(f)

clf


# In[43]:


# Calculate Directional Accuracy
directional_accuracy = np.mean((np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred))).astype(int)) * 100
print("Directional Accuracy (%):", directional_accuracy)


# In[ ]: