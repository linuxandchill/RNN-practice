from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

#load pretrained model w/ weights
model = load_model('stock_price_RNN.h5')

########### RUN ON TEST SET ###########################
training_data = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = training_data.iloc[:, 1:2].values
scaler = MinMaxScaler(feature_range = (0,1)) 
training_set = scaler.fit_transform(training_set)

test_data = pd.read_csv("Google_Stock_Price_Test.csv")
test_set = test_data.iloc[:, 1:2].values

#axis = 0 for vertical concatenation
complete_dataset = pd.concat((training_data['Open'], test_data['Open']), axis=0)
#predict for January
inputs = complete_dataset[len(complete_dataset) - len(test_data) - 60 : ].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
