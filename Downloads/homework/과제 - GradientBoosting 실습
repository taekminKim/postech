#!/usr/bin/env python
# coding: utf-8

# In[4]:


#행렬처리 모듈
import numpy as np
#자료구조인 Series, DataFrame 등을 제공하는 모듈
import pandas as pd
#추세 그래프 및 통계용 차트를 제공하는 시각화 모듈
import seaborn as sns
#그래프 및 시각화 모듈
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from sklearn import linear_model
# sklearn.preprocessing 의 scale : 스케일 조정 패키지
from sklearn.preprocessing import scale, robust_scale, minmax_scale
#sklearn.ensemble의 RandomForestClassifier : 분류랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
#sklearn.ensemble 의 GradientboostingClassifier : 분류그래디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier
#다중공선성 패키지 불러오기
from statsmodels.stats.outliers_influence import variance_inflation_factor
#sklearn 선형회귀 모형 -> 변수 선택법을 위함
from sklearn.linear_model import LinearRegression
#변수 선택법(후진제거법)
from sklearn.feature_selection import RFE
#평가 함수
from sklearn import metrics
#os: 환겨설정 패키지
import os 
from sklearn.model_selection import train_test_split


# In[5]:


#데이터 구성하기
ds_hmeq = pd.read_csv("실습화일/통신고객이탈.csv", engine="python")
ds_hmeq["CHURN"].replace(to_replace="Active", value=0, inplace = True)
ds_hmeq["CHURN"].replace(to_replace="Churned", value=1, inplace = True)
ds_hmeq = ds_hmeq.drop("CUSTOMER_ID", axis = 1, inplace = False)
ds_hmeq.head()


# In[6]:


ds_hmeq.isnull().sum(axis=0)


# In[8]:


#get_dummies : 데이터의 문자형 컬럼에 대한 더미변수 생성
ds_hmeq_dummy = pd.get_dummies(ds_hmeq)
#더미변수 생성된 데이터의 상위 5개 row를 확인
ds_hmeq_dummy.head()


# In[9]:


#dummy 변수로부터 변수 역할 지정
ds_hmeq_y = ds_hmeq_dummy["CHURN"]
ds_hmeq_x = ds_hmeq_dummy.drop("CHURN", axis = 1, inplace = False)
#train_test_split(X: 설명변수 데이터, Y: 목표변수 데이터, test_size = test 데이터비율, random_state : 랜덤)
ds_hmeq_train_x, ds_hmeq_test_x, ds_hmeq_train_y, ds_hmeq_test_y =train_test_split(ds_hmeq_x, ds_hmeq_y, test_size = 0.30, random_state = 1234)

print("train data X size : {}".format(ds_hmeq_train_x.shape))
print("train data Y size : {}".format(ds_hmeq_train_y.shape))
print("test data X size : {}".format(ds_hmeq_test_x.shape))
print("test data Y size : {}".format(ds_hmeq_test_y.shape))


# In[10]:


#그래디언트 부스팅 모델 생성 : GradientBoostingClassifier
gb_uncustomized = GradientBoostingClassifier(random_state = 1234)
gb_uncustomized.fit(ds_hmeq_train_x, ds_hmeq_train_y)
#훈련 데이터 셋 정확도
print("Accucary on training set: {:.3f}".format(gb_uncustomized.score(ds_hmeq_train_x, ds_hmeq_train_y)))
#테스트 데이터 셋 정확도
print("Accucary on test set: {:.3f}".format(gb_uncustomized.score(ds_hmeq_test_x, ds_hmeq_test_y)))


# In[11]:


gb_uncustomized


# In[12]:


#train 데이터셋 스케일 조정
ds_hmeq_train_x_scaled = scale(ds_hmeq_train_x, axis = 0)
# test 데이터셋 스케일 조정
ds_hmeq_test_x_scaled = scale(ds_hmeq_test_x, axis = 0)
#스케일이 변경된 X변수 확인
pd.DataFrame(ds_hmeq_train_x_scaled, columns = ds_hmeq_train_x.columns).head()


# In[13]:


gb_scaled = GradientBoostingClassifier(random_state=1234)
gb_scaled.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
#훈련 데이터 셋 정확도
print("Accucary on training set:{:.3f}".format(gb_scaled.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y)))
#테스트 데이터 셋 정확도
print("Accucary on test set:{:.3f}".format(gb_scaled.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y)))


# In[14]:


#트레인 및 테스트 정확도 결과 저장용
train_accuracy = []
test_accuracy = []
#트리 최대수(1~30)별로 랜덤 포레스트 모델 생성 및 정확도값 저장
for lr in range(1, 100, 5):
    gb1 = GradientBoostingClassifier(learning_rate = lr/100, random_state = 1234)
    gb1.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    train_accuracy.append(gb1.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(gb1.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
#저장된 모델의 트레인/테스트 데이터 분류 정확도 테이블 생성
ds_accuracy1 = pd.DataFrame()
ds_accuracy1["LearningRate"] = [lr/100 for lr in range(1, 100, 5)]
ds_accuracy1["TrainAccuracy"] = train_accuracy
ds_accuracy1["TestAccuracy"] = test_accuracy


# In[15]:


#LearningRate 별 정확도 테이블
ds_accuracy1.round(3)
#LearningRate별 정확도 그래프로 확인
ds_accuracy1.set_index("LearningRate", inplace = False).plot.line()


# In[16]:


#트리 수 : 50 ~ 150까지 10단위로 증가 및 학습율 0.2로 고정
para_estimators = [estimators for estimators in range(50, 150, 10)]
lr = 0.2
#트레인 및 테스트 정확도 결과 저장용
train_accuracy = []
test_accuracy = []
for estimators in para_estimators:
    gb2 = GradientBoostingClassifier(learning_rate = 0.2, n_estimators = estimators, random_state = 1234)
    gb2.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    train_accuracy.append(gb2.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(gb2.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
#저장된 모델의 criterion 및 트레니/테스트 데이터 분류 정확도 테이블 생성
ds_accuracy2 = pd.DataFrame()
ds_accuracy2["Estimators"] = para_estimators
ds_accuracy2["TrainAccuracy"] = train_accuracy
ds_accuracy2["TestAccuracy"] = test_accuracy
ds_accuracy2


# In[17]:


#n_estimators별 정확도 테이블
ds_accuracy2.round(3)
#n_estimators별 정확도 그래프로 확인
ds_accuracy2.set_index("Estimators", inplace= False).plot.line()


# In[18]:


lr = 0.2; n_estimators = 100;
para_max_depth = [i+1 for i in range(5)]
# 트레인 및 테스트 정확도 결과 저장용
train_accuracy = [];
test_accuracy = [];
#criterion:gini , entropy & max_depth : 1~10까지 반복 실행
for depth in para_max_depth:
    gb3 = GradientBoostingClassifier(learning_rate= 0.2, n_estimators = n_estimators, max_depth = depth, random_state = 1234)
    gb3.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    train_accuracy.append(gb3.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(gb3.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
#데이터 테이블로 저장
ds_accuracy3 = pd.DataFrame()
ds_accuracy3["MaxDepth"] = para_max_depth
ds_accuracy3["TrainAccuracy"] = train_accuracy
ds_accuracy3["TestAccuracy"] = test_accuracy


# In[19]:


#max_depth별 정확도 테이블
ds_accuracy3.round(3)
#max_depth별 정확도 그래프로 확인
ds_accuracy3.set_index("MaxDepth", inplace = False).plot.line()


# In[20]:


lr = 0.2; n_estimators = 100; max_depth = 3;
para_min_leaf_size = [i+1 for i in range(10)]
#트레인 및 테스트 정확도 결과 저장용
train_accuracy=[]
test_accuracy=[]
for leafsize in para_min_leaf_size:
    gb4 = GradientBoostingClassifier(learning_rate = lr, n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf =leafsize, random_state=1234)
    gb4.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    train_accuracy.append(gb4.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(gb4.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
    
#데이터 테이블로 저장
ds_accuracy4 = pd.DataFrame()
ds_accuracy4["MinLeafSize"] = para_min_leaf_size
ds_accuracy4["TrainAccuracy"] = train_accuracy
ds_accuracy4["TestAccuracy"] = test_accuracy    
    


# In[21]:


#min_samples_leaf별 정확도 테이블
ds_accuracy4.round(3)
#min_samples_leaf별 정확도 그래프로 확인
ds_accuracy4.set_index("MinLeafSize", inplace = False).plot.line()


# In[22]:


#최종 모델
gb_model = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 4, min_samples_leaf = 30, n_estimators = 5, random_state = 1234)
gb_model.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
#gb_model.feature_importances_로 설명변수 중요도 확인
ds_feature_importance = pd.DataFrame()
ds_feature_importance["feature"] = ds_hmeq_train_x.columns
ds_feature_importance["importance"] = gb_model.feature_importances_
ds_feature_importance.sort_values(by="importance", ascending=False).round(3)

#설명변수 중요도 그리는 함수 정의
def plot_feature_importances(model):
    n_features = ds_hmeq_train_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), ds_hmeq_train_x.columns)
    plt.xlabel("설명변수 중요도")
    plt.ylabel("설명변수")
    plt.ylim(-1, n_features)

#설명변수 중요도 그리는 함수 실행
plot_feature_importances(gb_model)


# In[ ]:





# In[ ]:




