from sklearn.datasets import load_boston

boston = load_boston()

x = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size= 0.2, train_size = 0.7, shuffle = True
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 모델완성하기
model1 = LinearRegression()
model2 = Ridge()
model3 = Lasso()
 
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

aaa = model1.score(x_test, y_test) 
print('aaa :', aaa) 

bbb = model2.score(x_test, y_test) 
print('bbb :', bbb)

ccc = model3.score(x_test, y_test) 
print('ccc :', ccc)
