from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape)  # (13, 3)
print(y.shape)  # (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)


# 2. 모델구성
# 연습 일뿐 - 통상적으로 LSTM을 이렇게 많이 하면 잘 안됨 보통 2~3개 
model = Sequential()
model.add(LSTM(10, activation = 'relu',input_shape = (3,1), 
               return_sequences = True))   # return_sequences = True : LSTM의 아웃풋을 다음 input layer 차원에 맞춰주는 거(defalut : False)
model.add(LSTM(7, activation = 'relu', return_sequences = True))  # LSTM은 3차원 형식으로 return sequence로 써줘야함  일반적인 Dense는 2차원이므로 안해도 됨
model.add(LSTM(5, activation = 'relu', return_sequences = True)) 
model.add(LSTM(6, activation = 'relu', return_sequences = True)) 
model.add(LSTM(2, activation = 'relu', return_sequences = True)) 
model.add(LSTM(20, activation = 'relu', return_sequences = False))
model.add(Dense(5, activation= 'relu'))
model.add(Dense(1)) 

# model.add(Dense(32))
# model.add(Dense(16))

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])  

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor= 'acc', patience= 20, mode = 'max') 
model.fit(x, y, epochs=200, batch_size=1, verbose=1, callbacks=[early_stopping])  

# 4. 평가예측
loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae) 


x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]])  # (3, ) -> (1,3) -> (1,3,1)
x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
