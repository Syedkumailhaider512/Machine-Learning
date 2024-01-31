import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Sequential
import tensorflow as tf


plot = plt.style.use('fivethirtyeight')
data = pd.read_csv('durhamtemp_1901_2019.csv')
#data.info()
#print(data.shape)
data = data.dropna(how='any', axis=0)
#print(data.shape)
data['Date'] =  pd.to_datetime(data['Date'], format='%d/%m/%Y')
data.sort_values('Date', inplace=True)

#print(data)
#data = data.rename(columns={'avg_temp':'Avg_Temp'})
temperature = data.loc[:,["Av temp"]]
temperature.columns = ["Avg_Temp"]
temperature.index = data["Date"]

#print(temperature.describe())

'''
plt.figure(figsize=(10,7))
plt.plot(temperature, label="average temperature", linewidth=1.5, alpha=0.6)
plt.xlabel("Years")
plt.ylabel("Average Temperature")
plt.title("Durham Daily Average Temperature (1901-2019)")
plt.show()


plt.figure(figsize=(10,10))
plt.plot(data[data["Year"] == 2019]["Av temp"].values, label="Avg temp (2019)", linewidth=1.2)
plt.ylabel("Avg. temp")
plt.title("Avg. Durham temperature (2019)")
plt.show()
'''


time_series = temperature['Avg_Temp']
#print(time_series)


def rolling_window(dataset, lag):
    X, Y = [], []

    for i in range(len(dataset) - lag):
        X.append(dataset[i:(i + lag)])
        Y.append(dataset[i + lag])

    X, Y = np.array(X), np.array(Y)
    X, Y = np.reshape(X, (X.shape[0], X.shape[1], 1)), np.reshape(Y, (Y.shape[0], 1))

    return X, Y

lag = 10
x, y = rolling_window(time_series, lag=lag)

#print(x.shape)

train_size = int(len(x) * 0.85)
test_size = len(x) - train_size

# Split the data into the train and test set
X_train = x[0:train_size]
Y_train = y[0:train_size]

X_test = x[train_size:len(x)]
Y_test = y[train_size:len(y)]

epochs = 10
num_layers = 100

# Define the LSTM model
model = Sequential()
model.add(LSTM(num_layers, activation='relu', input_shape=(lag, 1)))
model.add(Dense(1))

# Setting the optimizer and the loss function
model.compile(optimizer='adam', loss='mse')

model.summary()

# Training the model for 20 epochs


model.fit(
    X_train,
    Y_train,
    batch_size=32,
    epochs=epochs,
    validation_data=(X_test, Y_test),
)


def prediction_test(model, X_test):
    return [y[0] for y in model.predict(X_test)]


y_pred = prediction_test(model, X_test)

print(y_pred)


'''
plt.figure(figsize=(14,14))
plt.plot(Y_test, label="actual", linewidth=1)
plt.plot(y_pred, label="predicted", linewidth=1, alpha=0.8)
plt.ylabel("Avg. temperature")
plt.title("Test set data")
plt.legend()
plt.show()
'''

def predict_nsteps(model, dataset, nsteps, lag):
    X = dataset[-lag:].values.reshape(-1, 1)
    X = np.reshape(X, (1, X.shape[0], 1))

    # Making the prediction list
    yhat = []

    for _ in range(nsteps):
        prediction = model.predict(X)
        yhat.append(prediction)

        X = np.append(X, prediction)
        X = np.delete(X, 0)
        X = np.reshape(X, (1, len(X), 1))

    return yhat

nsteps = 2000
n_ahead = predict_nsteps(model=model, dataset=time_series, nsteps=nsteps, lag=lag)
y_ahead = [y[0][0] for y in n_ahead]

print(y_ahead)

plt.figure(figsize=(14,14))
plt.plot(y_ahead, label="predicted", alpha=0.6, marker='o')
plt.ylabel("Avg. temperature")
plt.title("2020 weather prediction")
plt.legend()
plt.show()

print("Prediction Completed!")