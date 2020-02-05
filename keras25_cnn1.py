from keras.models import Sequential
from keras.layers import Dense , Conv2D, MaxPooling2D, Flatten


model = Sequential()
model.add(Conv2D(7, (2,2), strides=2, padding = 'same',
                 input_shape=(5,5,1)))  #Dense층 input_shape은 1차원이므로, lstm은 2차원, cnn은 3차원
model.add(Conv2D(100,(2,2)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())  #Flatten(): Dense 층에 전달하기 위해 필요, 1차원으로 바꿔주는것 이유: Dense층 input_shape은 1차원이므로, lstm은 2차원, cnn은 3차원
model.add(Dense(1))


model.summary()