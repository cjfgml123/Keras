#1. 데이터
import numpy as np
x = np.array([range(1,101),range(101,201),range(301,401)])
y = np.array([range(101,201)])

# print(x.shape) # (3,100)
# print(y.shape) # (1,100)

x = np.transpose(x) # 행과 열을 바꿔줌
y = np.transpose(y)

print(x.shape) 
print(y.shape) 

from sklearn.model_selection import train_test_split

# train, validation , test  6:2:2 분할
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.4, shuffle=False)
x_val1, x_test, y_val2, y_test = train_test_split(x_val,y_val, test_size = 0.5, shuffle=False)


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential()

#model.add(Dense(5, input_dim =1))
model.add(Dense(5, input_shape = (3,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1)) # output 개수

model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,validation_data=(x_val,y_val))

#4. 평가
loss, mse = model.evaluate(x_test, y_test,batch_size=1)
print('mse : ', mse)

x_prd = np.array([[201,202,203],[204,205,206],[207,208,209]])
x_prd = np.transpose(x_prd)
aaa = model.predict(x_prd,batch_size=1)
print(aaa)


y_predict = model.predict(x_test,batch_size=1)
# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

# R2 구하기 0<r2<1 결정계수
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("r2:",r2_y_predict)
