import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


data = pd.read_csv('datasets/austin_weather.csv')

data_t = data[['TempAvgF']]
data_t.index = pd.to_datetime(data[['Date']].stack(), format='%Y%m%d', errors='ignore') # datetime merge doar pe series si .stack face series
trend = np.linspace(0, len(data_t)-1, 50, dtype='int64')
print(data_t.head())


plt.figure(figsize=(20, 8))
data_t.plot()
plt.title('Time series')
plt.xlabel('data')
plt.ylabel('temperature')
#plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

copy_data = data_t
scaler = MinMaxScaler(feature_range=(-1, 1))

data_antrenare = data_t.iloc[:1000]
data_test = data_t.iloc[1000:]


def make_data(data_frame, history):
    sequences = []
    sequ_pred = []
    values = data_frame['TempAvgF'].values
    for i in range(len(values)-history-1):
        sequences.append(values[i:i+history])
        sequ_pred.append(values[i+history+1])
    return np.array(sequences), np.array(sequ_pred)

hist_size = 20
train_x, train_y = make_data(data_antrenare, hist_size)
test_x, test_y = make_data(data_test, hist_size)


#for i in range(len(train_x)):
#    print(train_x[i], train_y[i])


#building model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(20, 1)))
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(train_x, train_y, epochs=200, verbose=0, validation_data=(test_x, test_y), batch_size=20)

print('Loss ', history.history['loss'][199])

predictions = model.predict(test_x)

plt.figure(figsize=(20, 9))
plt.plot(predictions, color='red', linewidth=3)
plt.plot(test_y, color='blue')
plt.legend(('Predicted', 'Actual'))
plt.show()

