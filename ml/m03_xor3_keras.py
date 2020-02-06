from sklearn import datasets, preprocessing
from sklearn import model_selection
from keras import datasets, utils
from keras import models, layers, initializers, losses, optimizers, metrics
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD

# #1. 데이터
x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_train = np.array([0, 1, 1, 0])


# #2. 모델
# model = LinearSVC()
model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='sgd',
              metrics=['binary_crossentropy'])
model.fit(x_train, y_train, epochs=1000, batch_size=1)

#4. 평가 예측
x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)
''' # 정확도 값에 따른 이진분류 , 보완필요
for i in len(y_predict) :
    if y_predict[i] > 0.5 :
        y_predict[i] = 1
    else :
        y_predict[i] = 0
'''    
print(x_test, "의 예측결과 : ", y_predict)
# print("acc = ", accuracy_score([0,1, 1, 0], y_predict))
# print("acc = ", accuracy_score(y_train, y_predict))