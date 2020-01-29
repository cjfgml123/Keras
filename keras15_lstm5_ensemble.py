from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 주의 : 앙상블 할때 모델1,2의 각 데이터 shape구조가 같아야 한다. 다르면 error
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],
           [2,3,4],[3,4,5],[4,5,6]])
y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7])


x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

# 모델 1
input1 = Input(shape = (3,1))
model1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(5)(model1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
model1 = Dense(1)(dense3)

# 모델 2
input2 = Input(shape = (3,1))
model2 = LSTM(6, activation='relu')(input1)
dense11 = Dense(4)(model2)
dense21 = Dense(3)(dense11)
dense31 = Dense(2)(dense21)
model2 = Dense(1)(dense31)

# 1번째 모델 합치는 방법
#from keras.layers.merge import concatenate
#merge1 = concatenate([output1,output2]) # input 모델 합치기 # concatenate는 모델 1,2의 아웃풋 노드수가 달라도 된다. 엮는것이므로

# 2번째 방법
from keras.layers.merge import Add
merge1 = Add()([model1,model2]) # 모델1,2 의 아웃풋 노드수가 같아야 한다.

middle1 = Dense(4)(merge1)
middle2 = Dense(3)(middle1)

# 1번째 output
output_1 = Dense(2)(middle2)
output_1 = Dense(1)(output_1) # y의 벡터가 1개이므로 1로 설정

# 2번째 output
output_2 = Dense(3)(middle2)
output_2 = Dense(1)(output_2)

model = Model(inputs = [input1, input2], outputs = [output_1,output_2])
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용

from keras.callbacks import EarlyStopping # Epoch 을 많이 돌린 후, 특정 시점에서 멈추는 것, 과적합 방지

# patience 는 성능이 증가하지 않는 epoch 을 몇 번이나 허용할 것인가를 정의하고 그 수가 넘으면 스탑
early_stopping = EarlyStopping(monitor='loss', patience= 20, mode = 'auto')

model.fit([x1,x2],[y1,y2], 
          epochs=100, batch_size = 1, verbose= 1, 
          callbacks = [early_stopping]) # verbose = 0 하면 진행과정이 안나옴 verbose = 1 : default로 진행과정이나옴

# 평가예측
loss = model.evaluate([x1,x2],[y1,y2], batch_size = 1)

print(loss)


x1_input = array([[6.5,7.5,8.5],[50,60,70], # 중요! : 처음 데이터의 열만 맞춰주면 된다.
                 [70,80,90],[100,110,120]])

x2_input = array([[6.5,7.5,8.5],[50,60,70],
                 [70,80,90],[100,110,120]])

x1_input = x1_input.reshape(4,3,1)
x2_input = x2_input.reshape(4,3,1)

y_predict = model.predict([x1_input,x2_input])
print(y_predict)
