import numpy as np
import pandas as pd



samsung = np.load('./samsung_stock/data/samsung.npy')

kospi200 = np.load('./samsung_stock/data/kospi200.npy')

print(samsung)
print(samsung.shape)
#print(kospi200)

# x 데이터 steps만큼의 행으로 다음 행 y 를 예측하기 위한 데이터를 나누는것 
# 426행에서 x데이터를 1~5행으로 2~6행으로 --- 가면 421,5,5로 된다.
def split_xyS(dataset, time_steps, y_column) :
    x, y = list() , list()
    for i in range(len(dataset)) : # 426번 돈다.
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number,:] # steps = 5 이고 로직이1일때  (5,5) 
        tmp_y = dataset[x_end_number:y_end_number,3]  # 로직 1일때 (5:6) tmp_y에 6일째의 종가값저장 1일차index가 0이므로 헷갈림 주의, 3 <- 종가가 3번째 열
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xyS(samsung,5,1)
print(x.shape) 
print(y.shape)
print(x[0,:],"\n",y[0]) # 처음 5일치의 데이터와 6일째의 종가가 잘 출력되는 것을 확인

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x ,y ,random_state = 1, test_size = 0.3, shuffle = False)

print(x_train.shape)
print(y_test.shape)

# 데이터 전처리
# 3차원 -> 2차원
x_train = np.reshape(x_train,
                     (x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test,
                     (x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
# print(x_train.shape)
# print(x_test.shape)

# standardscaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler() # 3차원안됨 2차원만 가능 reshape해줘야 한다.
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0,:])
print(x_train.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
#model.add(Dense(200, activation = 'relu', input_shape = (25,)))
model.add(Dense(200, input_shape = (25,))) 
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])

model.fit(x_train, y_train ,verbose= 1 ,batch_size=1, epochs=150) # 따로 val 데이터를 안나눴으면 
# validation_split로 훈련데이터에서 0.2 만큼 쓴다.

loss, mse = model.evaluate(x_test, y_test,batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test)

for i in range(5):
    print('종가:', y_test[i], '/ 예측가:',y_pred[i])
