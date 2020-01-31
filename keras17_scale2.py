import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000], [30000,40000,50000], [40000,50000,60000],[100,200,300]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

# 데이터 정규화 및 표준화 방법 4가지 사용방법 동일
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import RobustScaler, MaxAbsScaler

scaler = StandardScaler()
scaler.fit(x) 
x1 = scaler.transform(x)

# 실습
# train은 10개, 나머지는 test
# Dense 모델 구현
# R2 지표
# [250,260, 270] 으로 predict
x_train = x1[:10]
x_test = x1[10:14]
y_train = y[:10]
y_test = y[10:14]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape(10, 3, 1)
x_test = x_test.reshape(4, 3, 1)

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(200, activation = 'relu', input_shape = (3,1)))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용
model.fit(x_train,y_train, 
          epochs=100, batch_size = 1) 

# 평가예측
loss, mae = model.evaluate(x_test, y_test, batch_size = 1)
print(loss, mae)

x_input = np.array([250,260,270])
x_input = x_input.reshape(1,3,1)
x_pred = model.predict(x_input, batch_size=1)
print(x_pred)

'''
y_predict = model.predict(x_test,batch_size=1)
# R2 구하기 0<r2<1 결정계수
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("r2:",r2_y_predict)
'''