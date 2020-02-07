import numpy as np
import pandas as pd

samsung = np.load('./samsung_stock/data/samsung.npy')
kospi200 = np.load('./samsung_stock/data/kospi200.npy')

# print(samsung) #(426, 5)
# print(samsung.shape) #(426, 5)
# print(kospi200) #(426, 5)
# print(kospi200.shape) #(426, 5)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps   
        y_end_number = x_end_number + y_column   
        
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :] # x값(5x5)
        tmp_y = dataset[x_end_number : y_end_number, 3] # y값(종가)
        x.append(tmp_x) 
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(samsung, 5, 1)
# print(x.shape) #(421, 5, 5)
# print(y.shape) #(421, 1)
# print(x[0, :], '\n', y[0])

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, test_size = 0.3, shuffle = False)

print(x_train.shape) #(294, 5, 5)
print(x_test.shape)  #(127, 5, 5)

# 데이터 전처리
# StandardScaler
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])  # 3차원 -> 2차원
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
# x_test = scaler.transform(x_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0, :])

x_train = x_train.reshape(294, 5, 5)
x_test = x_test.reshape(127, 5, 5)

# 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, input_shape=(5, 5)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x_train, y_train, epochs=200, batch_size = 1, validation_split = 0.2) 

loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print('loss:', loss)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# y_pred = model.predict(x_test)

# for i in range(5):
#     print('종가:', y_test[i], 'y예측값:', y_pred[i])
    
# # RMSE
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_pred):
#     return np.sqrt(mean_squared_error(y_test, y_pred))
# print('RMSE : ', RMSE(y_test, y_pred))
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# loss: 2081114.4385755644
# 종가: [47200] y예측값: [46644.277]
# 종가: [47150] y예측값: [47117.145]
# 종가: [46100] y예측값: [46880.24]
# 종가: [46550] y예측값: [46467.582]
# 종가: [45350] y예측값: [46585.594]
# RMSE :  1442.6069037524778
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
x_input = np.array([[53000,53900,51800,51900,39565391],
                    [55500,57400,55200,57200,23995260],
                    [57100,59000,56800,58900,21800192],
                    [60000,60200,58900,59500,19278165],
                    [60100,61100,59700,61100,14727159]])
x_input = x_input.reshape(1,25)
x_input = scaler.transform(x_input)
x_input = x_input.reshape(1,5,5)
y_pred = model.predict(x_input)
print(y_pred)  # 종가[[52640.973]]

