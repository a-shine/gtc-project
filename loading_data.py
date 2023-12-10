import os

import matplotlib.pyplot as plt
import numpy as np

from utils.load import load_data, set_param

dirs = {'main': os.getcwd()}


def get_data(experiment, visualise=False):
    params = set_param(experiment)

    dirs['data'] = dirs['main'] + '/' + params['dir_data']

    # Load data
    X, _, t, _, _ = load_data(experiment, dirs, params)

    if visualise:
        # plot X signal in time domain
        plt.plot(t, X)
        plt.xlabel('Time [s]')
        plt.ylabel('Shear stress [N/m^2, Pa]')
        plt.show()

    return X, t


def train_val_test_split_prediction(X, t):
    # time series forcasting train val test split
    pass

# FIXME: This function is a black box - figure out how it is sampling the signal to create the input and output sequences
def train_test_split_forecasting(X, t, input_sequence_length, forecast_horizon, test_ratio=0.1,
                                 visualise=False):
    """
    Time series forecasting train, validation, and test split.

    Parameters:
    - X: Time series data
    - t: Time information (optional, not used in this function)
    - input_sequence_length: Size of the input sequence (window size)
    - forecast_horizon: Number of future points to predict
    - val_ratio: Ratio of the data to be used for validation (default is 0.1)
    - test_ratio: Ratio of the data to be used for testing (default is 0.1)

    Returns:
    - X_train, y_train: Training data and labels
    - X_val, y_val: Validation data and labels
    - X_test, y_test: Test data and labels
    """

    # Calculate the sizes for the test sets
    test_size = int(len(X) * test_ratio)

    # Create input-output pairs for training
    input_sequences_train = []
    ground_truth_values_train = []

    for i in range(val_start):
        input_sequence = X[i:i + input_sequence_length]
        ground_truth = X[i + input_sequence_length:i + input_sequence_length + forecast_horizon]

        input_sequences_train.append(input_sequence)
        ground_truth_values_train.append(ground_truth)

    # Convert lists to NumPy arrays for training
    X_train = np.array(input_sequences_train)
    y_train = np.array(ground_truth_values_train)

    # Reshape X_train to match the input shape expected by the model
    X_train = X_train.reshape(-1, input_sequence_length, 1)

    # Create input-output pairs for testing
    input_sequences_test = []
    ground_truth_values_test = []

    for i in range(len(X) - test_size, len(X) - input_sequence_length - forecast_horizon + 1):
        input_sequence = X[i:i + input_sequence_length]
        ground_truth = X[i + input_sequence_length:i + input_sequence_length + forecast_horizon]

        input_sequences_test.append(input_sequence)
        ground_truth_values_test.append(ground_truth)

    # Convert lists to NumPy arrays for testing
    X_test = np.array(input_sequences_test)
    y_test = np.array(ground_truth_values_test)

    # Reshape X_test to match the input shape expected by the model
    X_test = X_test.reshape(-1, input_sequence_length, 1)

    if visualise:
        # take a random sample from the training set
        i = np.random.randint(len(X_train))
        # plot the input sequence and expected output
        plt.plot(X_train[i], label='Input sequence')
        plt.plot(np.arange(input_sequence_length, input_sequence_length + forecast_horizon), y_train[i],
                 label='Ground truth')
        plt.show()

    return X_train, y_train, X_test, y_test
