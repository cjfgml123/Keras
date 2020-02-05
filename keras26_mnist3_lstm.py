
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense , Conv2D, MaxPooling2D, Flatten, LSTM
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0],28*28,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28*28,1).astype('float32')/255
from keras.utils import np_utils  
y_train = np_utils.to_categorical(y_train)  # 원 핫 인코딩
y_test = np_utils.to_categorical(y_test) 

print(x_train.shape) 
print(y_train.shape) 
#print(type(x_train))

model = Sequential()
model.add(LSTM(7,input_shape=(28*28,1)))  #Dense층 input_shape은 1차원이므로, lstm은 2차원, cnn은 3차원
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(10, activation = 'softmax')) # mnist 0~9 출력 10개이므로 dense 10개를 줘야 한다. activation은 softmax

#model.summary()

model.compile(loss = 'categorical_crossentropy', # activation 을 softmax로 주면 필수로 categorical_crossentropy줘야한다.
              optimizer = 'adam',
              metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience= 20)

model.fit(x_train, y_train,validation_split = 0.2 ,epochs=50, batch_size = 8,verbose = 1, callbacks=[early_stopping]) 

acc = model.evaluate(x_test, y_test)

print(acc) 