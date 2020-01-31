from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense, LSTM

def split_sequence2(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) : #- 1:   x가 다항으로 들어갈때
            break
        seq_x, seq_y = sequence[i:end_ix,:-1], sequence[end_ix-1,-1]
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
x , y = split_sequence2(dataset, n_steps)

for i in range(len(x)) : 
    print(x[i], y[i])
print(x.shape)
print(y.shape)

# 실습
# 1. 함수분석
# 2. DNN 모델 만들기
# 3. 지표는 loss
# DNN은 인풋데이터가 2차원이어야 하므로 3차원데이터를 2차원으로 바꿔줘야 한다.
# 위에 (8,3,2) 이므로 (8,6)으로 바꿔준다. 8 : 행 3*2=6 열 
# 노드수를 맞춰줘야 한다. 8*3*2 = 48 이므로 2차원으로 바꿀 때 행은 그대로 이므로
# 8*x = 48 x = 6
x = x.reshape(8,6)
model = Sequential()
model.add(Dense(200, activation = 'relu', input_shape = (6,)))
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
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용

model.fit(x,y, 
          epochs=150, batch_size = 1) 

# 평가예측
loss, mae = model.evaluate(x, y, batch_size = 1)
print(loss, mae)

x_input = array([[90,95],[100,105],[110,115]])
x_input = x_input.reshape(1,6)
y_predict = model.predict(x_input)
print(y_predict)

print(x)
print(y)
