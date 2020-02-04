import pandas as pd
import numpy as np

df1 = pd.read_csv('./samsung_stock/samsung.csv',index_col =0, # 첫째줄을 인덱스로 한다.
                  header=0, encoding='cp949', sep = ',') # header = 0 : csv파일에 헤더인덱스를 0으로 설정

print(df1)
print(df1.shape)

df2 = pd.read_csv('./samsung_stock/kospi200.csv',index_col =0, # 첫째줄을 인덱스로 한다.
                  header=0, encoding='cp949', sep = ',')

print(df2)
print(df2.shape)

# kospi200의 모든 데이터
# loc : 인덱스 기준으로 행 데이터 읽기
# iloc : 행 번호를 기준으로 행 데이터 읽기
for i in range(len(df2.index)) :      # 거래량 str -> int로 변경
    df2.iloc[i,4] = int(df2.iloc[i,4].replace(',','')) # replace('찾을값','바꿀값',바꿀횟수)
    
# 삼성전자의 모든 데이터
for i in range(len(df1.index)) :      # 모든 str -> int로 변경
        for j in range(len(df1.iloc[i])) :
                df1.iloc[i,j] = int(df1.iloc[i,j].replace(',',''))

# print(df1)
# print(df1.shape) 

df1 = df1.sort_values(['일자'],ascending = [True]) # 날짜별 오름차순
df2 = df2.sort_values(['일자'],ascending = [True])

print(df1)

df1 = df1.values # pandas 에서 numpy 형식으로 변환
df2 = df2.values 

print(type(df1), type(df2)) # numpy.ndarray 
print(df1.shape, df2.shape) # (426,5)

np.save('./samsung_stock/data/samsung.npy', arr = df1) # numpy 파일로 저장
np.save('./samsung_stock/data/kospi200.npy', arr = df2) 

