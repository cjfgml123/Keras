import numpy as np
import pandas as pd

samsung = np.load('./samsung_stock/data/samsung.npy')
kospi200 = np.load('./samsung_stock/data/kospi200.npy')

print(samsung) #(426, 5)
print(samsung.shape) #(426, 5)
print(kospi200) #(426, 5)
print(kospi200.shape) #(426, 5)

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

x, y = split_xy5(samsung, 25, 1)
print(x.shape) #(401, 25, 5)
print(y.shape) #(401, 1)
print(x[0, :], '\n', y[0])

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, test_size = 0.3, shuffle = False)

print(x_train.shape) #(280, 25, 5)
print(x_test.shape)  #(121, 25, 5)

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

x_train = x_train.reshape(280, 25, 5)
x_test = x_test.reshape(121, 25, 5)

# 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(64, input_shape=(25, 5)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train, y_train, epochs=150, batch_size = 1, validation_split = 0.2, callbacks=[early_stopping]) 

loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print('loss:', loss)

y_pred = model.predict(x_test)

for i in range(5):
    print('종가:', y_test[i], 'y예측값:', y_pred[i])
    
# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE : ', RMSE(y_test, y_pred))

# loss: 31006250.549357265
# 종가: [44950] y예측값: [45096.71]
# 종가: [43950] y예측값: [45045.906]
# 종가: [43500] y예측값: [44133.832]
# 종가: [43200] y예측값: [43331.457]
# 종가: [42650] y예측값: [43561.52]
# RMSE :  5568.32563635447