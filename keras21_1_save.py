# 모델 만들기 

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

# model.summary()

model.save('./save/save_test01.h5')
print('save sccess')