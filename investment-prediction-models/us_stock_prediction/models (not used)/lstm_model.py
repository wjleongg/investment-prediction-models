import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import warnings
warnings.filterwarnings("ignore")

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

pd.set_option('display.max_columns', None)


# In[2]:


df = pd.read_csv('data_files/indicators_data_train.csv')
df.head(2)


# In[3]:


dft = pd.read_csv('data_files/indicators_data_test.csv')
dft.head(2)


# ## data exploration

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


# check for null values
df.isnull().sum()


# In[7]:


# change datatype of Date
df["Date"]=pd.to_datetime(df["Date"])
df.head(2)


# ## training data

# ### train test split

# In[8]:


# not using 'Date' and 'Symbol' columns! (train)
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI_14D', 'BB_Middle_Band', 'BB_Upper_Band', 'BB_Lower_Band', 'Aroon_Oscillator', 'STOK', 'STOD', 'VWAP', 'Momentum', 'OBV', 'TEMA', 'NATR', 'plusDI', 'minusDI', 'ADX', 'MACD', 'Money_Flow_Index', 'MIN_Volume', 'MAX_Volume']]
y = df['Close']


# In[9]:


# Split the data into training, validation and test sets
seed = 42

## First, split the data into a training set and a temporary set using a 80-20 split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)


# In[10]:


# not using 'Date' and 'Symbol' columns! (test)
X_test = dft[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI_14D', 'BB_Middle_Band', 'BB_Upper_Band', 'BB_Lower_Band', 'Aroon_Oscillator', 'STOK', 'STOD', 'VWAP', 'Momentum', 'OBV', 'TEMA', 'NATR', 'plusDI', 'minusDI', 'ADX', 'MACD', 'Money_Flow_Index', 'MIN_Volume', 'MAX_Volume']]
y_test = dft['Close']


# ### normalising values

# In[11]:


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler with the desired feature range (default is 0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the training data and transform the training data
X_train = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test = scaler.transform(X_test)


# In[12]:


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val", y_val.shape)
print("X_test:", X_test.shape)
print("y_test", y_test.shape)


# In[13]:


batch_size = 128

# Create a Dataset from the training data
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(batch_size)

# Create a Dataset from the validation data
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(batch_size)

# Create a Dataset from the test data
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(batch_size)


# # Building LSTM model

# In[14]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # 1 feature, adapt as needed
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Initialize the model
model = Sequential()

# First Bidirectional LSTM layer with dropout
model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Second regular LSTM layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Output layer
model.add(Dense(units=1))  # Single output unit for regression task


# In[16]:


# Compile the model with Adam optimizer and a customized learning rate
optimizer = Adam(learning_rate=0.0005)  # Optimized learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# Fit the model on training data with validation split
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val,y_val), verbose=1)


# In[17]:


# Summary of the model architecture
model.summary()


# ## metric analysis on test data

# In[18]:


test_loss, test_acc = model.evaluate(test_ds)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)


# In[19]:


y_pred = model.predict(X_test)
y_pred.shape


# In[20]:


# Make predictions on the validation and test datasets
# y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
# print(y_val_pred.shape)
print(y_test_pred.shape)


# In[21]:


# Convert y_val_pred to a 1D array or Series
# y_val_pred = y_val_pred.flatten()
y_test_pred = y_test_pred.flatten()
y_test_pred.shape


# In[22]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Mean Absolute Error (MAE) - lower better
mae = mean_absolute_error(y_test, y_test_pred)

# Mean Squared Error (MSE) - lower better
mse = mean_squared_error(y_test, y_test_pred)

# Root Mean Squared Error (RMSE) - closer to 0 better
rmse = np.sqrt(mse)

# Mean Absolute Percentage Error (MAPE) - lower better
mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

# Explained Variance Score, 1 will be perfect fit
evs = explained_variance_score(y_test, y_test_pred)

# Directional Accuracy - closer to 1 better
direction_accuracy = np.mean((np.sign(np.diff(y_test)) == np.sign(np.diff(y_test_pred))).astype(int)) * 100

# Display the results
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
print('Explained Variance Score:', evs)
print("Directional Accuracy (%):", direction_accuracy)


# In[23]:


import matplotlib.pyplot as plt

# Extract loss values from the history object
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ## save prediction model into a pickle file

# In[24]:


import pickle
from sklearn.ensemble import RandomForestClassifier

# Save the model to a file using pickle
with open('models/lstm_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[26]:


# check if file is able to load

# Load the model from file
with open('models/lstm_model.pkl', 'rb') as f:
    clf = pickle.load(f)

clf


# In[ ]: