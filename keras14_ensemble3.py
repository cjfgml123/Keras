import numpy as np    
# 데이터
x1 = np.array([range(1, 101), range(101, 201), range(301, 401)])
#x2 = np.array([range(1001, 1101), range(1101, 1201), range(1301, 1401)])
#y1 = np.array([range(101, 201)])


y1 = np.array([range(1, 101), range(101, 201), range(301, 401)])
y2 = np.array([range(1001, 1101), range(1101, 1201), range(1301, 1401)])
y3 = np.array([range(1, 101), range(101, 201), range(301, 401)])

x1 = np.transpose(x1)
#x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, train_size=0.6, random_state=66, 
    shuffle = False) # x, y -> (train(60%), test(40%))

x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, test_size=0.5, random_state=66, 
    shuffle = False) # test(40%) -> (val(50%), test(50%))

y2_train, y2_test, y3_train, y3_test = train_test_split(
    y2, y3, train_size=0.6, random_state=66, 
    shuffle = False) 

y2_val, y2_test, y3_val, y3_test = train_test_split(
    y2_test, y3_test, test_size=0.5, random_state=66, 
    shuffle = False) 

print(y3_train.shape)
print(y3_test.shape)
print(y3_val.shape)


# 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#함수형 모델 1
input1 = Input(shape = (3, ))
dense1 = Dense(50)(input1)
dense3 = Dense(50)(dense1)
output1 = Dense(58)(dense3)

# #함수형 모델 2
# input2 = Input(shape = (3, ))
# dense21 = Dense(7)(input2)
# dense22 = Dense(4)(dense21)
# output2 = Dense(5)(dense22)

# from keras.layers.merge import concatenate
# merge1 = concatenate([output1,output2]) # 모델 1,2 엮기

# middle1 = Dense(4)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(1)(middle2)    # 현재 merge된 마지막 레이어

output_1 = Dense(300)(output1)   # 1번째 아웃풋 모델
output_1 = Dense(3)(output_1)   # 변수명 동일하게 해도 문제 x

output_2 = Dense(300)(output1)  # 2번째 아웃풋 모델
output_2 = Dense(100)(output_2)   
output_2 = Dense(3)(output_2) 

output_3 = Dense(300)(output1)   # 3번째 아웃풋 모델
output_3 = Dense(3)(output_3)   # 변수명 동일하게 해도 문제 x


model = Model(inputs = input1 , 
              outputs = [output_1,output_2,output_3]) # input 여러개면 리스트형식으로 넣어준다.

model.summary()


# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x1_train, [y1_train,y2_train,y3_train], 
          epochs=100, batch_size = 1, validation_data = (x1_val,[y1_val,y2_val,y3_val])) 

# 평가예측
aaa = model.evaluate(x1_test, [y1_test, y2_test, y3_test], batch_size = 1)
print('mse:', aaa)


# # 예측
x1_pred = np.array([[200, 201, 202], [203, 204, 205],[206, 207, 208]])
#x2_pred = np.array([[210, 211, 212], [213, 214, 215],[216, 217, 218]])

x1_pred = np.transpose(x1_pred)
#x2_pred = np.transpose(x2_pred)

predict_bbb = model.predict(x1_pred, batch_size = 1)
print(predict_bbb)

y_predict = model.predict(x1_test, batch_size = 1)
# print(y_predict) (20,3) * 3 리스트

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
val = (RMSE(y1_test,y_predict[0]) + RMSE(y2_test,y_predict[1]) + RMSE(y3_test,y_predict[2]))/3
print('val:',val)
#print('RMSE : ', RMSE(y1_test, y1_predict[0])) # 오차값이기 때문에 값이 적을수록 좋음


# RMSE2
#print('print r_mse :', np.sqrt(mse))

from sklearn.metrics import r2_score
r2_y_predict1 = r2_score(y1_test, y_predict[0])
r2_y_predict2 = r2_score(y2_test, y_predict[1])
r2_y_predict3 = r2_score(y3_test, y_predict[2])
r2_y_predict = (r2_y_predict1 + r2_y_predict2 + r2_y_predict3) / 3
print('R2 : ', r2_y_predict)  #R2는 최대값이 1로 1에 가까울수록 정확함
