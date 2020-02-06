# LinearSVC, KNeighborsClassifier 로 하기
from keras.models import Sequential
from keras.layers import Dense , Dropout
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf 
import numpy as np

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt("C:\Study\keras\ml\data\pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
y = dataset[:,8]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, shuffle=False)

# 모델의 설정
model = KNeighborsClassifier(n_neighbors= 3)

model.fit(x_train,y_train)

# 평가, 예측
y_pred = model.predict(x_test)
print("정답률 :", accuracy_score(y_test, y_pred)) # 정답률 