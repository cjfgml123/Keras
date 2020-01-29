from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape) # 13,3
print(y.shape) # 13

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # 13,3

model = Sequential()
model.add(LSTM(200, activation = 'relu', input_shape = (3,1))) #중요!-input_shape = (x,y_trian) x : 열의 개수 y_trian : 몇개씩 자를지
model.add(Dense(128))
model.add(Dense(200))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용

from keras.callbacks import EarlyStopping # Epoch 을 많이 돌린 후, 특정 시점에서 멈추는 것, 과적합 방지

# patience 는 성능이 증가하지 않는 epoch 을 몇 번이나 허용할 것인가를 정의하고 그 수가 넘으면 스탑
early_stopping = EarlyStopping(monitor='loss', patience= 20, mode = 'auto')

model.fit(x,y, 
          epochs=500, batch_size = 1, verbose= 1, 
          callbacks = [early_stopping]) # verbose = 0 하면 진행과정이 안나옴 verbose = 1 : default로 진행과정이나옴

# 평가예측
loss, mae = model.evaluate(x, y, batch_size = 1)

print(loss, mae)


x_input = array([[6.5,7.5,8.5],[50,60,70],
                 [70,80,90],[100,110,120]])

x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
