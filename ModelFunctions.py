import keras.backend
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.metrics import MeanSquaredError
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def train_nn(learning_rate, batch_size, epochs, neurons, csv_path, layers, activations):
    keras.backend.clear_session()
    # load data from CSV file
    data = pd.read_csv(csv_path)

    # split data into features and target variable
    X = data.drop('Y', axis=1).values
    y = data['Y'].values

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # define the model architecture
    model = Sequential()
    for it in range(layers):
        model.add(Dense(neurons[it], activation=activations[it]))
    model.add(Dense(1))

    # compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # evaluate the model on the test set
    preds = model.predict(X_test)
    if math.isnan(preds[0]):
        return model, 'N', 'N'
    else:

        r_squared = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        # return the trained model and evaluation metrics
        return model, mse, r_squared


def train_cnn(learning_rate, batch_size, epochs, neurons, csv_path, layers, activations, filters, kernels):
    # load data from CSV file
    keras.backend.clear_session()
    data = pd.read_csv(csv_path)

    # split data into features and target variable
    X = data.drop('Y', axis=1).values
    y = data['Y'].values

    # reshape input data to 3D tensor
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # define the model architecture
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernels, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=filters, kernel_size=kernels, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    for it in range(layers):
        model.add(Dense(neurons[it], activation=activations[it]))
    model.add(Dense(1))
    model.add(Dense(1))

    # compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    preds = model.predict(X_test)
    if math.isnan(preds[0]):
        return model, 'N', 'N'
    else:

        r_squared = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        # return the trained model and evaluation metrics
        return model, mse, r_squared

def train_cnn_noPool(learning_rate, batch_size, epochs, neurons, csv_path, layers, activations, kernels, filters):
    # load data from CSV file
    keras.backend.clear_session()
    data = pd.read_csv(csv_path)

    # split data into features and target variable
    X = data.drop('Y', axis=1).values
    y = data['Y'].values

    # reshape input data to 3D tensor
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # define the model architecture
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernels, activation='relu', padding='same'))
    model.add(Flatten())
    for it in range(layers):
        model.add(Dense(neurons[it], activation=activations[it]))
    model.add(Dense(1))

    # compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    preds = model.predict(X_test)
    if math.isnan(preds[0]):
        return model, 'N', 'N'
    else:

        r_squared = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        # return the trained model and evaluation metrics
        return model, mse, r_squared




def train_rl(learning_rate, neurons, csv_path, layers, activations):
    # load data from CSV file
    data = pd.read_csv(csv_path)

    # split data into features and target variable
    X = data.drop('target_variable_name', axis=1).values
    y = data['target_variable_name'].values

    # standardize the input data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # define the model architecture
    model = Sequential()
    for it in range(layers):
        model.add(Dense(neurons[it], activation=activations[it]))
    model.add(Dense(1))

    # compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)

    # define the reinforcement learning agent
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=10000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=1, memory=memory, nb_steps_warmup=1000,
                   target_model_update=1e-2, policy=policy)

    # define the reward function
    def get_reward(y_true, y_pred):
        error = abs((y_true - y_pred) / y_true)
        if error < 0.1:
            reward = 0.8 + 0.2 * (1 - error / 0.1)
        elif error < 0.2:
            reward = 0.8 * (1 - 10 * (error - 0.1) ** 2)
            if reward < 0:
                reward = 0
        else:
            reward = 0
        return reward

    # train the agent
    dqn.compile(optimizer=Adam(), metrics=['mae'])
    dqn.fit(X, y, nb_steps=5000, visualize=False, verbose=1, callbacks=[])

    # evaluate the agent
    y_pred = dqn.predict(X)
    rewards = np.array([get_reward(y[i], y_pred[i]) for i in range(len(y))])
    accuracy = np.mean(rewards)

    # return the trained model and evaluation metrics
    return model, accuracy


