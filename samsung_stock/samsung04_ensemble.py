import numpy as np
import pandas as pd

samsung = np.load('./samsung_stock/data/samsung.npy')
kospi200 = np.load('./samsung_stock/data/kospi200.npy')

# print(samsung)
# print(samsung.shape)

# print(kospi200)
# print(kospi200.shape)


def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps    # 0 + 5 = 5 (x의 끝 숫자)
        y_end_number = x_end_number + y_column  # 5 + 1 =6 (y의 끝 숫자)
        
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number : y_end_number, 3]   # 3번째 열이 '종가'==> y값
        x.append(tmp_x) 
        y.append(tmp_y)
    return np.array(x), np.array(y) 

x1, y1 = split_xy5(samsung,5,1)
x2, y2 = split_xy5(kospi200,5,1)

# print(x.shape)    # (421, 5, 5)
# print(y.shape)    # (421, 1)
# print(x[0,:],'\n', y[0])

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, random_state=1, test_size=0.3, shuffle = False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, random_state=1, test_size=0.3, shuffle = False)

# y2는 kospi200의 '현재가'=>필요 없음


## 3차원 -> 2차원
x1_train = np.reshape(x1_train,(x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,(x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,(x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,(x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))

print(x1_train.shape)  # (294, 25)
print(x1_test.shape)   # (127, 25)


# 데이터 전처리
# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

scaler = StandardScaler()
scaler.fit(x2_train)
x2_train_scaled = scaler.transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)

print(x2_train_scaled[0, :])

# 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(25,))
dense1 = Dense(5)(input1)
dense1 = Dense(2)(dense1)
dense1 = Dense(3)(dense1)
output1 = Dense(1)(dense1)

input2= Input(shape=(25,))
dense2 = Dense(7)(input2)
dense2 = Dense(4)(dense2)
output2 = Dense(1)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([output1,output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1,input2], outputs = output)


# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train_scaled, x2_train_scaled], y1_train, epochs=100, batch_size = 1, validation_split = 0, callbacks=[early_stopping]) 

loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size = 1)
print('loss:', loss)

y_pred = model.predict([x1_test_scaled, x2_test_scaled])

for i in range(5):
    print('종가:', y1_test[i], 'y예측값:', y_pred[i])