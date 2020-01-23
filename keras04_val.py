#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 1 -> 스칼라 , 1,2,3 -> 벡터 ->행렬 ->텐서 
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
x_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 1 -> 스칼라 , 1,2,3 -> 벡터 ->행렬 ->텐서 
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])

# print(x.shape)
# print(y.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential()

#model.add(Dense(5, input_dim =1))
model.add(Dense(5, input_shape = (1,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1,validation_data=(x_val,y_val))

#4. 평가
loss, mae = model.evaluate(x_test, y_test)
print('mae : ', mae)

x_prd = np.array([11,12,13])
aaa = model.predict(x_prd)
print(aaa)

# bbb = model.predict(x)
# print(bbb)

