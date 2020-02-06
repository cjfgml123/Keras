from keras.models import Sequential
from keras.layers import Dense , Dropout
from sklearn.metrics import accuracy_score
import tensorflow as tf 
import numpy as np

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt("./ml/data/pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
y = dataset[:,8]

# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim = 8, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 모델 컴파일
model.compile(loss = "binary_crossentropy",
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 설정
model.fit(X,y, epochs = 200, batch_size = 10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X,y)[1]))