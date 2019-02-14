#!/usr/bin/env python
# coding: utf-8

# In[32]:


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
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from sklearn import linear_model
# sklearn.preprocessing 의 scale : 스케일 조정 패키지
from sklearn.preprocessing import scale, robust_scale, minmax_scale
#sklearn.ensemble의 RandomForestClassifier : 분류랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
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


# In[4]:


#데이터 구성하기
ds_hmeq = pd.read_csv("실습화일/HMEQ.csv")
ds_hmeq.head()


# In[5]:


#결측치 확인
ds_hmeq.isnull().sum(axis = 0)


# In[6]:


#fillna : 결측치를 채우는 함수
#JOB 변수의 결측치는 Other로 입력, inplace: fillna 함수 적용 후 ds_hmeq 데이터에 저장, False면 저장 안함
ds_hmeq["JOB"].fillna("Other", inplace = True)
#숫자형 변수의 결측치는 해당 변수의 평균값 입력 : ds_hmeq.mean() 각 변수별 평균 계산 후 결측치 대체
ds_hmeq.fillna(ds_hmeq.mean(), inplace = True)


# In[7]:


#get_dummies : 데이터의 문자형 컬럼에 대한 더미변수 생성
ds_hmeq_dummy = pd.get_dummies(ds_hmeq)
#더미변수 생성된 데이터의 상위 5개 row를 확인
ds_hmeq_dummy.head()


# In[8]:


#random seed 고정 -> 매번 실행시 같은 값을 얻음
np.random.seed(seed= 1234)
# 0.7(70%) 보다 작으면 True
msk = np.random.rand(ds_hmeq_dummy.shape[0]) < 0.7
ds_hmeq_train = ds_hmeq_dummy[msk]
ds_hmeq_test = ds_hmeq_dummy[~msk]
#train/test 데이터의 목표변수 설명변수 지정
ds_hmeq_train_y = ds_hmeq_train["BAD"]
ds_hmeq_train_x = ds_hmeq_train.drop("BAD", axis = 1, inplace = False)
ds_hmeq_test_y = ds_hmeq_test["BAD"]
ds_hmeq_test_x = ds_hmeq_test.drop("BAD", axis = 1, inplace = False)
#train 데이터와 test 데이터 크기
print("train data X size : {}".format(ds_hmeq_train_x.shape))
print("train data Y size : {}".format(ds_hmeq_train_y.shape))
print("test data X size : {}".format(ds_hmeq_test_x.shape))
print("test data Y size : {}".format(ds_hmeq_test_y.shape))




# In[10]:


#랜덤 포레스트 모델 생성 : RandomForestClassifier
rf_uncustomized = RandomForestClassifier(random_state = 1234)
rf_uncustomized.fit(ds_hmeq_train_x, ds_hmeq_train_y)
#훈련 데이터 셋 정확도
print("Accucary on training set: {:.3f}".format(rf_uncustomized.score(ds_hmeq_train_x, ds_hmeq_train_y)))
#테스트 데이터 셋 정확도
print("Accucary on test set: {:.3f}".format(rf_uncustomized.score(ds_hmeq_test_x, ds_hmeq_test_y)))


# In[11]:


rf_uncustomized


# In[23]:


#train 데이터셋 스케일 조정
ds_hmeq_train_x_scaled = scale(ds_hmeq_train_x, axis = 0)
# test 데이터셋 스케일 조정
ds_hmeq_test_x_scaled = scale(ds_hmeq_test_x, axis = 0)
#스케일이 변경된 X변수 확인
pd.DataFrame(ds_hmeq_train_x_scaled, columns = ds_hmeq_train_x.columns).head()


# In[25]:


#새로운 랜덤 포레스트 모델 생성
rf_scaled = RandomForestClassifier(random_state=1234)
rf_scaled.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
#훈련 데이터 셋 정확도
print("Accucary on training set:{:.3f}".format(rf_scaled.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y)))
#테스트 데이터 셋 정확도
print("Accucary on test set:{:.3f}".format(rf_scaled.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y)))


# In[26]:


#트리 수 최대값 : 30;1~30까지 진행
n_iter_tree = 30
#트레인 및 테스트 정확도 결과 저장용
train_accuracy = []
test_accuracy = []
#트리 최대수(1~30)별로 랜덤 포레스트 모델 생성 및 정확도값 저장
for n_tree in range(n_iter_tree):
    rf1 = RandomForestClassifier(n_estimators = n_tree+1, random_state = 1234)
    rf1.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    train_accuracy.append(rf1.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(rf1.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
#저장된 모델의 트레인/테스트 데이터 분류 정확도 테이블 생성
ds_accuracy1 = pd.DataFrame()
ds_accuracy1["NumberofTree"] = [n_tree + 1for n_tree in range(n_iter_tree)]
ds_accuracy1["TrainAccuracy"] = train_accuracy
ds_accuracy1["TestAccuracy"] = test_accuracy


# In[27]:


#테이블 결과
ds_accuracy1
#그래프 생성
ds_accuracy1.set_index("NumberofTree", inplace=False).plot.line()


# In[28]:


#트레인 및 테스트 정확도 결과 저장용
train_accuracy = []
test_accuracy = []
#적용가능한 criterion: gini, entropy
para_criterion = ["gini", "entropy"]
#para_criterion 별로 랜덤 포레스트 모델 생성 및 정확도 값 저장
for criterion in para_criterion:
    rf2 = RandomForestClassifier(criterion = criterion, random_state = 1234)
    rf2.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    train_accuracy.append(rf2.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(rf2.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
#저장된 모델의 criterion 및 트레니/테스트 데이터 분류 정확도 테이블 생성
ds_accuracy2 = pd.DataFrame()
ds_accuracy2["Criterion"] = para_criterion
ds_accuracy2["TrainAccuracy"] = train_accuracy
ds_accuracy2["TestAccuracy"] = test_accuracy
ds_accuracy2


# In[29]:


# 트레인 및 테스트 정확도 결과 저장용
train_accuracy = [];
test_accuracy = [];
#최대 깊이 1~10까지 순차 실행
para_max_depth = [i+1 for i in range(10)]*2
n_iter_depth = 10
#criterion 10개씩 리스트 생성
para_criterion = ["gini"] * n_iter_depth + ["entropy"] * n_iter_depth
#criterion:gini , entropy & max_depth : 1~10까지 반복 실행
for(criterion, depth) in zip(para_criterion, para_max_depth):
    rf3 = RandomForestClassifier(criterion = criterion, max_depth = depth, random_state = 1234)
    rf3.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    train_accuracy.append(rf3.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(rf3.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))
#데이터 테이블로 저장
ds_accuracy3 = pd.DataFrame()
ds_accuracy3["Criterion"] = para_criterion
ds_accuracy3["Depth"] = para_max_depth
ds_accuracy3["TrainAccuracy"] = train_accuracy
ds_accuracy3["TestAccuracy"] = test_accuracy


# In[30]:


#테이블 결과
ds_accuracy3


# In[33]:


#그래프 생성
ds_accuracy3_melt = pd.melt(ds_accuracy3, id_vars = ["Criterion", "Depth"])
ds_accuracy3_melt["Accuracy"] = ds_accuracy3_melt["Criterion"]+"_"+ds_accuracy3_melt["variable"]
sns.lineplot(x="Depth", y="value", hue="Accuracy", data = ds_accuracy3_melt)


# In[34]:


#트레인 및 테스트 정확도 결과 저장용
train_accuracy=[]
test_accuracy=[]
#최대 깊이 4로 고정 실행
n_depth =4
#최소 잎사귀 수 10까지 순차 실행
n_iter_min_leaf_size = 10
para_criterion = ["gini"]* n_iter_min_leaf_size
para_min_leaf_size = [i+1 for i in range(n_iter_min_leaf_size)]
#criterion:gini, entropy & max_depth: 1~10까지 & min_sample_size: 1~10까지 반복 실행
for(criterion, leafsize) in zip(para_criterion, para_min_leaf_size):
    rf4 = RandomForestClassifier(criterion = criterion, max_depth = n_depth, min_samples_leaf = leafsize, random_state=1234)
    rf4.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
    train_accuracy.append(rf4.score(ds_hmeq_train_x_scaled, ds_hmeq_train_y))
    test_accuracy.append(rf4.score(ds_hmeq_test_x_scaled, ds_hmeq_test_y))


# In[36]:


#데이터 테이블로 저장
ds_accuracy4 = pd.DataFrame()
ds_accuracy4["Criterion"] = para_criterion
ds_accuracy4["Depth"] = n_depth
ds_accuracy4["MinLeafSize"] = para_min_leaf_size
ds_accuracy4["TrainAccuracy"] = train_accuracy
ds_accuracy4["TestAccuracy"] = test_accuracy
#테이블 결과
ds_accuracy4
#그래프 생성
ds_accuracy4_melt = pd.melt(ds_accuracy4, id_vars = ["Criterion", "Depth", "MinLeafSize"])
ds_accuracy4_melt["Accuracy"] = ds_accuracy4_melt["Criterion"] + "_" + ds_accuracy4_melt["variable"]
sns.lineplot(x="MinLeafSize", y = "value", hue="Accuracy", data = ds_accuracy4_melt)


# In[37]:


#최종 모델
rf_model = RandomForestClassifier(criterion = "gini", max_depth= 4, min_samples_leaf = 10, n_estimators = 100, random_state = 1234)
rf_model.fit(ds_hmeq_train_x_scaled, ds_hmeq_train_y)
#rf_model.feature_importances_ 로 설명변수 중요도 확인
ds_feature_importance = pd.DataFrame()
ds_feature_importance["feature"] = ds_hmeq_train_x.columns
ds_feature_importance["importance"] = rf_model.feature_importances_
ds_feature_importance.sort_values(by = "importance", ascending=False)

#설명변수 중요도 그리는 함수 정의
def plot_feature_importances(model):
    n_features = ds_hmeq_train_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), ds_hmeq_train_x.columns)
    plt.xlabel("설명변수 중요도")
    plt.ylabel("설명변수")
    plt.ylim(-1, n_features)
#설명변수 중요도 그리는 함수 실행
plot_feature_importances(rf_model)


# In[ ]:




