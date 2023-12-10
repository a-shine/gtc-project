import os

os.environ["KERAS_BACKEND"] = "torch"

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras_core as keras
from keras_core import Sequential
from keras_core.layers import Dense, LSTM
import numpy as np
from sklearn.preprocessing import StandardScaler

from loading_data import get_data

# exp = 'b726'
exp = 'b698'
# exp = 'i417'
# exp = 'p4679'
# exp = 'p4581'
# exp = 'cascadia'
# exp = 'sim_b726'
# exp = 'sim_b698'
# exp = 'sim_i417'

# Load your data
data, _ = get_data(exp, visualise=False)

# Normalize your data
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, 1))

# Define the lookback period and prepare input-output pairs
lookback = 10
inputs = np.array([data[i:i+lookback] for i in range(len(data)-lookback)])
outputs = data[lookback:]

# Split the data into training and testing sets
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, shuffle=False, test_size=0.2)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(lookback, 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_inputs, train_outputs, epochs=10, verbose=1)

# Evaluate the model
mse = model.evaluate(test_inputs, test_outputs, verbose=0)
print(f'Test MSE: {mse}')

# Generate predictions for the test inputs
predictions = model.predict(test_inputs)

# Plot the ground truth and the predictions
plt.figure(figsize=(10, 6))
plt.plot(test_outputs, label='Ground Truth')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()