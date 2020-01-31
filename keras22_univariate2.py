from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

dataset = [10,20,30,40,50,60,70,80,90,100]
n_step = 3

x, y = split_sequence(dataset,n_step)

for i in range(len(x)):
    print(x[i],y[i])
    

# 실습 DNN 모델 구성
model = Sequential()
model.add(Dense(200, activation = 'relu', input_shape = (3,)))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(1)) # output 개수


model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용

model.fit(x,y, 
          epochs=150, batch_size = 1) 

# 평가예측
loss, mae = model.evaluate(x, y, batch_size = 1)
print(loss, mae)

x_input = array([[90,100,110]])
y_predict = model.predict(x_input)
print(y_predict)