#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
#sklearn.svm의 SVC : & from sklearn.preprocessing의 scale
from sklearn.svm import SVC
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


# In[11]:


#데이터 구성하기
ds_hmeq = pd.read_csv("실습화일/HMEQ.csv")
ds_hmeq.head()


# In[12]:


ds_hmeq.isnull().sum(axis=0)


# In[13]:


#fillna : 결측치를 채우는 함수
#JOB 변수의 결측치는 Other로 입력, inplace: fillna 함수 적용 후 ds_hmeq 데이터에 저장, False면 저장 안함
ds_hmeq["JOB"].fillna("Other", inplace = True)
#숫자형 변수의 결측치는 해당 변수의 평균값 입력 : ds_hmeq.mean() 각 변수별 평균 계산 후 결측치 대체
ds_hmeq.fillna(ds_hmeq.mean(), inplace = True)


# In[14]:


#get_dummies : 데이터의 문자형 컬럼에 대한 더미변수 생성
ds_hmeq_dummy = pd.get_dummies(ds_hmeq)
#더미변수 생성된 데이터의 상위 5개 row를 확인
ds_hmeq_dummy.head()


# In[15]:


#dummy 변수로부터 변수 역할 지정
ds_hmeq_y = ds_hmeq_dummy["BAD"]
ds_hmeq_x = ds_hmeq_dummy.drop("BAD", axis = 1, inplace = False)
#train_test_split(X: 설명변수 데이터, Y: 목표변수 데이터, test_size = test 데이터비율, random_state : 랜덤)
ds_hmeq_train_x, ds_hmeq_test_x, ds_hmeq_train_y, ds_hmeq_test_y =train_test_split(ds_hmeq_x, ds_hmeq_y, test_size = 0.30, random_state = 1234)

print("train data X size : {}".format(ds_hmeq_train_x.shape))
print("train data Y size : {}".format(ds_hmeq_train_y.shape))
print("test data X size : {}".format(ds_hmeq_test_x.shape))
print("test data Y size : {}".format(ds_hmeq_test_y.shape))


# In[16]:


svm_uncustomized = SVC(random_state = 1234)
svm_uncustomized.fit(ds_hmeq_train_x, ds_hmeq_train_y)
#훈련 데이터 셋 정확도
print("Accucary on training set: {:.3f}".format(svm_uncustomized.score(ds_hmeq_train_x, ds_hmeq_train_y)))
#테스트 데이터 셋 정확도
print("Accucary on test set: {:.3f}".format(svm_uncustomized.score(ds_hmeq_test_x, ds_hmeq_test_y)))


# In[8]:


svm_uncustomized


# In[20]:


#트레인 및 테스트 정확도 결과 저장용
train_accuracy = []
test_accuracy = []
para_C = []
#트리 최대수(1~30)별로 랜덤 포레스트 모델 생성 및 정확도값 저장
for C in range(10):
    svm1 = SVC(C= (C+1)/10, random_state = 1234)
    svm1.fit(ds_hmeq_train_x, ds_hmeq_train_y)
    para_C.append((C+1)/10)
    train_accuracy.append(svm1.score(ds_hmeq_train_x, ds_hmeq_train_y))
    test_accuracy.append(svm1.score(ds_hmeq_test_x, ds_hmeq_test_y))
#저장된 모델의 트레인/테스트 데이터 분류 정확도 테이블 생성
ds_accuracy1 = pd.DataFrame()
ds_accuracy1["C"] = para_C
ds_accuracy1["TrainAccuracy"] = train_accuracy
ds_accuracy1["TestAccuracy"] = test_accuracy


# In[21]:


#LearningRate 별 정확도 테이블
ds_accuracy1.round(3)
#LearningRate별 정확도 그래프로 확인
ds_accuracy1.set_index("C", inplace = False).plot.line()


# In[26]:



#트레인 및 테스트 정확도 결과 저장용
train_accuracy = []
test_accuracy = []
para_gamma = []
# Gamma를 0.1부터 1까지 0.1단위로 증가
for gamma in range(1,10):
    svm2 = SVC(gamma = gamma/10, random_state = 1234)
    svm2.fit(ds_hmeq_train_x, ds_hmeq_train_y)
    para_gamma.append(gamma/10)
    train_accuracy.append(svm2.score(ds_hmeq_train_x, ds_hmeq_train_y))
    test_accuracy.append(svm2.score(ds_hmeq_test_x, ds_hmeq_test_y))
#저장된 모델의 criterion 및 트레니/테스트 데이터 분류 정확도 테이블 생성
ds_accuracy2 = pd.DataFrame()
ds_accuracy2["Gamma"] = para_gamma
ds_accuracy2["TrainAccuracy"] = train_accuracy
ds_accuracy2["TestAccuracy"] = test_accuracy
ds_accuracy2


# In[27]:


#n_estimators별 정확도 테이블
ds_accuracy2.round(3)
#n_estimators별 정확도 그래프로 확인
ds_accuracy2.set_index("Gamma", inplace= False).plot.line()


# In[28]:


#train 데이터셋 스케일 조정
ds_hmeq_train_x_scaled = scale(ds_hmeq_train_x, axis = 0)
# test 데이터셋 스케일 조정
ds_hmeq_test_x_scaled = scale(ds_hmeq_test_x, axis = 0)
#스케일이 변경된 X변수 확인
pd.DataFrame(ds_hmeq_train_x_scaled, columns = ds_hmeq_train_x.columns).head()


# In[29]:


#새로운 SVC모형 생성
svm_uncustomized_scaled = SVC(random_state = 1234)
svm_uncustomized_scaled.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
#train 데이터 셋 정확도
print("Accucary on training set:{:.3f}".format(svm_uncustomized_scaled.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y)))
print("Accucary on test set:{:.3f}".format(svm_uncustomized_scaled.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y)))


# In[32]:


# 트레인 및 테스트 정확도 결과 저장용
train_accuracy = [];
test_accuracy = [];
para_C = []
for C in range(15):
    svm1_scaled = SVC(C = C+1, random_state = 1234)
    svm1_scaled.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    para_C.append(C+1)
    train_accuracy.append(svm1_scaled.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(svm1_scaled.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
#데이터 테이블로 저장
ds_accuracy1_scaled = pd.DataFrame()
ds_accuracy1_scaled["C"] = para_C
ds_accuracy1_scaled["TrainAccuracy"] = train_accuracy
ds_accuracy1_scaled["TestAccuracy"] = test_accuracy


# In[33]:


#max_depth별 정확도 테이블
ds_accuracy1.round(3)
#max_depth별 정확도 그래프로 확인
ds_accuracy1_scaled.set_index("C", inplace = False).plot.line()


# In[38]:


# 트레인 및 테스트 정확도 결과 저장용
train_accuracy = [];
test_accuracy = [];
para_gamma = []
for gamma in range(1,10):
    svm2_scaled = SVC(gamma = gamma/10, random_state = 1234)
    svm2_scaled.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    para_gamma.append(gamma/10)
    train_accuracy.append(svm2_scaled.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(svm2_scaled.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
#데이터 테이블로 저장
ds_accuracy2_scaled = pd.DataFrame()
ds_accuracy2_scaled["Gamma"] = para_gamma
ds_accuracy2_scaled["TrainAccuracy"] = train_accuracy
ds_accuracy2_scaled["TestAccuracy"] = test_accuracy  
    


# In[39]:


#min_samples_leaf별 정확도 테이블
ds_accuracy2_scaled.round(3)
#min_samples_leaf별 정확도 그래프로 확인
ds_accuracy2_scaled.set_index("Gamma", inplace = False).plot.line()


# In[ ]:




