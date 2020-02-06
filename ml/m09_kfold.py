import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
# sklearn 0.20.3 에서 31개
# sklearn 0.21.2 에서 40개중 4개만 됨

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./ml/data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,'Name']
x = iris_data.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]



# Classifier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')

kfold_cv = KFold(n_splits=5, shuffle = True)

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기
    model = algorithm()
    
    if hasattr(model, "score") : #
        scores = cross_val_score(model, x, y ,cv=kfold_cv) # 이 함수 자체에 fit()이 포함되있어서 따로 fit을 안해줘도 된다.
        print(name, "의 정답률 =")
        
        #print(scores)
        
        scores = (scores.mean()) # 5개의 정확도를 평균내서 계산
        print(scores)