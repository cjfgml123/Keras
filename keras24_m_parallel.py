from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout

def split_sequence3(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) -1:   
            break
        seq_x, seq_y = sequence[i:end_ix,:], sequence[end_ix, : ]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# print(in_seq1.shape) # (10,)
# print(out_seq.shape) # (10,)

in_seq1 = in_seq1.reshape(len(in_seq1),1)
in_seq2 = in_seq2.reshape(len(in_seq2),1)
out_seq = out_seq.reshape(len(out_seq),1)

# print(in_seq1.shape) # (10,1)
# print(in_seq2.shape) # (10,1)
# print(out_seq.shape) # (10,1)

dataset = hstack((in_seq1, in_seq2, out_seq)) # 합치기 #(10,3)

# print(dataset)
# print(dataset.shape)

n_steps = 3
x , y = split_sequence3(dataset, n_steps)

for i in range(len(x)) : 
    print(x[i], y[i])
    
print(x.shape) # 7,3,3
print(y.shape) # 7,3


# 실습 keras23_multiple1.py
# 1. 함수분석
# 2. DNN 모델 만들기
# 3. 지표는 loss
# DNN은 인풋데이터가 2차원이어야 하므로 3차원데이터를 2차원으로 바꿔줘야 한다.
# 위에 (8,3,2) 이므로 (8,6)으로 바꿔준다. 8 : 행 3*2=6 열 
# 노드수를 맞춰줘야 한다. 8*3*2 = 48 이므로 2차원으로 바꿀 때 행은 그대로 이므로
# 8*x = 48 x = 6
x = x.reshape(7,9)

model = Sequential()
model.add(Dense(200, activation = 'relu', input_shape = (9,)))
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
model.add(Dense(3)) # y 가 (7,3) 이므로 출력 3


model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용

model.fit(x,y, 
          epochs=150, batch_size = 1) 

# 평가예측
loss, mae = model.evaluate(x, y, batch_size = 1)
print(loss, mae)

x_input = array([[90,95,185],[100,105,205],[110,115,225]])
x_input = x_input.reshape(1,9)
y_predict = model.predict(x_input)
print(y_predict)



