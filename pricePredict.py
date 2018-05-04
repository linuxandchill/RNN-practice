import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#import dataset
dataset = pd.read_csv("Google_Stock_Price_Train.csv")
#we need a numpy arr shaped (rows, cols)
training_set = dataset.iloc[:, 1:2].values
#print(training_set.shape)

#create scaler
scaler = MinMaxScaler(feature_range = (0,1)) 
#fit scaler to data
scaled_data = scaler.fit_transform(training_set)
#print(scaled_data.ndim)
#print(scaled_data.shape)
#print(scaled_data[:10])
#print(scaled_data[1100:])

#input//t-step previous prices
x_train = []
#output//t-step+1
y_train = []

#print(scaled_data.shape[0])
for t in range(60, scaled_data.shape[0]): #[rows, columns] --> we only have 1 column in scaled_data
    #it is index 0
    x_train.append(scaled_data[t-60:t, 0])
    y_train.append(scaled_data[t,0])

#each row is a day, cols are prev prices for each day
x_train = np.array(x_train)
y_train = np.array(y_train)

#print(x_train.shape)
#print(y_train.shape)

#add new dim to x_train --> becomes 3D tensor 
#required by Keras (batch_size, timesteps, input_dim)
#last arg (1) is the number of features
x_train = np.reshape(x_train,
            (x_train.shape[0], 
            x_train.shape[1],1))

print(x_train.shape)

###########BUILDING MODEL###########################
model = Sequential()

########### LSTM LAYER 1 ###########################
#args: #of neurons in layer, if stacking use ret_seq=True, input_shape 
#last arg (1) is the number of features
model.add(LSTM(units = 50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
#drop 20% of neurons in layer
model.add(Dropout(0.2))

########### LSTM LAYER 2 ###########################
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))

########### LSTM LAYER 3 ###########################
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))

########### LSTM LAYER 4 ###########################
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

########### OUTPUT LAYER ###########################
model.add(Dense(units=1))

########### COMPILE ###########################
#can use RMSprop or adam optimizer
#use MSE for regression
model.compile(optimizer='adam', 
        loss='mean_squared_error',
        metrics=['acc'])

model.summary()

########### FIT MODEL ###########################
history = model.fit(x_train, 
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2)

model.save('stock_price_RNN.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label="Training accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation accuracy")
plt.title("ACCURACY")
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label="training loss")
plt.plot(epochs, val_loss, 'b', label="validation loss")
plt.title("LOSS")
plt.legend()

plt.show()






