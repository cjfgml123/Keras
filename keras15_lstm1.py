from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM


x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]])
y = array([4,5,6,7,8])

print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1) # 중요!-input_shape 에맞게 바꿔줘야해서 (5,3,1)로 reshape해준다.
# 만약 1개씩 자른다할때 (x,y,1) 2개씩이면 (x,y,2) ...
# x= x.reshape(5,3,1)


model = Sequential()
model.add(LSTM(200, activation = 'relu', input_shape = (3,1))) #중요!-input_shape = (x,y_trian) x : 열의 개수 y_trian : 몇개씩 자를지
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용
model.fit(x,y, 
          epochs=200, batch_size = 1) 

# 평가예측
loss, mae = model.evaluate(x, y, batch_size = 1)
print(loss, mae)

x_input = array([6,7,8])
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)
print(y_predict)
