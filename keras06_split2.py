#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split

# train, validation , test  6:2:2 분할
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.4, shuffle=False)
x_val1, x_test, y_val2, y_test = train_test_split(x_val,y_val, test_size = 0.5, shuffle=False)

'''
#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential()

#model.add(Dense(5, input_dim =1))
model.add(Dense(5, input_shape = (1,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1,validation_data=(x_val,y_val))

#4. 평가
loss, mae = model.evaluate(x_test, y_test,batch_size=1)
print('mae : ', mae)

x_prd = np.array([101,102,103])
aaa = model.predict(x_prd,batch_size=1)
print(aaa)

# bbb = model.predict(x)
# print(bbb)
'''
