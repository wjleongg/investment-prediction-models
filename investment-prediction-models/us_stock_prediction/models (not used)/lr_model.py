# Import necessary libraries
import numpy as np  # Linear algebra
import pandas as pd  # Data processing
import matplotlib.pyplot as plt  # Data visualization
from matplotlib import style
import datetime

from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

import math

pd.set_option('display.max_columns', None)


# In[2]:


df = pd.read_csv("data_files/indicators_data_train.csv") 
df


# In[3]:


dft = pd.read_csv('../data_files/indicators_data_test.csv')
dft


# ## data exploration

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


# check for null values
df.isnull().sum()


# ## data modification

# In[7]:


# # change datatype of Date
# df["Date"]=pd.to_datetime(df["Date"])
# df.head(2)


# In[8]:


## remove!!!!
forecast_col = 'Close'

forecast_out = int(math.ceil(0.2 * len(df)))
print('number of data that will be forcasted:', forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.head(2)


# In[9]:


#X = np.array(df.drop(['label'], 1))

X=np.array(df.drop(['label','Symbol','Date'], axis=1))
#print(X)
X = preprocessing.scale(X)
#print(X)
X_lately = X[-forecast_out:]
#print(X_lately)
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])


# ## train, validation, test split

# In[10]:


# not using 'Date' and 'Symbol' columns! (train)
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI_14D', 'BB_Middle_Band', 'BB_Upper_Band', 'BB_Lower_Band', 'Aroon_Oscillator', 'STOK', 'STOD', 'VWAP', 'Momentum', 'OBV', 'TEMA', 'NATR', 'plusDI', 'minusDI', 'ADX', 'MACD', 'Money_Flow_Index', 'MIN_Volume', 'MAX_Volume']]
y = df['Close']


# In[11]:


# Split the data into training, validation and test sets
seed = 42

## First, split the data into a training set and a temporary set using a 80-20 split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed) 


# In[12]:


# not using 'Date' and 'Symbol' columns! (test)
X_test = dft[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI_14D', 'BB_Middle_Band', 'BB_Upper_Band', 'BB_Lower_Band', 'Aroon_Oscillator', 'STOK', 'STOD', 'VWAP', 'Momentum', 'OBV', 'TEMA', 'NATR', 'plusDI', 'minusDI', 'ADX', 'MACD', 'Money_Flow_Index', 'MIN_Volume', 'MAX_Volume']]
y_test = dft['Close']


# In[13]:


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test", y_test.shape)


# ## model training

# In[14]:


model = LinearRegression(n_jobs=-1)
# model = Lasso()
model.fit(X_train, y_train)


# In[15]:


# Use Polynomial Regression if needed
poly = PolynomialFeatures(degree=2)  # Change degree as necessary
X_poly = poly.fit_transform(X_train)

# Model selection with regularization
model = Lasso()  # Use Lasso or Ridge
# model = Ridge()

# Hyperparameter tuning
params = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X_poly, y_train)

# Best model
best_model = grid_search.best_estimator_


# In[16]:


# best parameters
print(f'Best Parameters: {grid_search.best_params_}')


# In[17]:


# Predictions
X_test_poly = poly.transform(X_test)
predictions = best_model.predict(X_test_poly)


# ## model evaluation

# In[18]:


# # Import required metrics functions
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Predictions are already done here:
# # Assuming you have 'predictions' for your model and 'y_test' as the actual values

# # Calculate evaluation metrics
# mse = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)

# # Print the metrics
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"R² Score: {r2}")


# In[19]:


from sklearn.metrics import mean_absolute_error

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# In[20]:


from sklearn.metrics import mean_squared_error

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


# In[21]:


import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[22]:


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[23]:


from sklearn.metrics import r2_score

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# In[24]:


def calculate_mase(y_true, y_pred):
    naive_forecast = np.roll(y_true, 1)  # Shift the actual values by one time step
    naive_forecast[0] = np.nan  # Set the first value to NaN (or you could drop it)
    
    mae_model = np.mean(np.abs(y_true[1:] - y_pred[1:]))
    mae_naive = np.mean(np.abs(y_true[1:] - naive_forecast[1:]))
    
    return mae_model / mae_naive


# In[27]:


mae = calculate_mae(y_test, predictions)
mse = calculate_mse(y_test, predictions)
rmse = calculate_rmse(y_test, predictions)
mape = calculate_mape(y_test, predictions)
r2 = calculate_r2(y_test, predictions)
mase = calculate_mase(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Scaled Error (MASE): {mase}")


# ## save model into pickle file

# In[26]:


import pickle
from sklearn.ensemble import RandomForestClassifier

# Save the model to a file using pickle
with open('models\lr_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models\lr_model.pkl', 'rb') as f:
    clf = pickle.load(f)

clf


# In[ ]: