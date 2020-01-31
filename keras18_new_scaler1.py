
import numpy as np

x = np.array(range(1,21))
y = np.array(range(1,21))

x = x.reshape(20,1)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size = 0.5
)

print("=============================")
print(x_train)
print("=============================")
print(x_test)
print("=============================")

# 통상적으로 train만 fit하고 나머지는 transform , 처음 데이터 전부를 할때도 있지만 자원낭비가 있을 수 가 있다.
# 표준화할때 x_train만 fit 해줘도 x_test,vaildation 에 적용할 수 있다. 즉 transform 만 해주면된다.
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("=============================")
print(x_train)
print("=============================")
print(x_test)
print("=============================")