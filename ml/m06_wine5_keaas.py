import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from keras import models, layers, initializers, losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

#1. 데이터 읽어 들이기
wine = pd.read_csv('C:\Study\keras\ml\data\winequality-white.csv', sep=';', encoding='utf-8')


y = wine["quality"]
x = wine.drop('quality', axis =1)

newlist = []
for v in list(y):
    if v<= 4:
        newlist +=[0]
    elif v <=7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist


label_encoder=LabelEncoder()
label_ids=label_encoder.fit_transform(y)

onehot_encoder=OneHotEncoder(sparse=False)
reshaped=label_ids.reshape(len(label_ids), 1)
onehot=onehot_encoder.fit_transform(reshaped)


x_train, x_test, y_train, y_test = train_test_split(
    x, onehot, test_size= 0.2, train_size = 0.8)

print(onehot.shape)
print(x.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(11, input_shape=(11, ), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

#모델 실행
model.fit(x_train, y_train, epochs=10, batch_size=10)

y_pred = model.predict(x_test)

test_score = accuracy_score(y_test, y_pred.round(), normalize=False)
test_score = test_score / (len(y_test)+1)
print(test_score)