#!/usr/bin/env python
# coding: utf-8

# In[31]:


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
from sklearn.preprocessing import scale, robust_scale, minmax_scale
#다중공선성 패키지 불러오기
from statsmodels.stats.outliers_influence import variance_inflation_factor
#sklearn 선형회귀 모형 -> 변수 선택법을 위함
from sklearn.linear_model import LinearRegression
#변수 선택법(후진제거법)
from sklearn.feature_selection import RFE


# In[68]:


#ingine: 파일명이 한글일 경우 "Python" 엔진 사용
ds_boston_house = pd.read_csv("실습화일/BOSTON_HOUSING.csv", engine="python")
ds_house  = ds_boston_house.copy()
ds_house


# In[49]:


ds_house.isnull().sum()


# In[33]:


#select_dtypes : 특정 변수 타입을 선택/제외하여 데이터 추출, object: 문자형
ds_house = ds_house.select_dtypes(exclude="object")


# In[41]:


#그래프 분석
ds_health[["CRIM","ZN","INDUS","CHAS"]].hist(figsize = (10, 4))


# In[30]:


#목표변수, 설명변수 역할 지정
ds_house_y = ds_house["MEDV"]
ds_house_x = ds_house.drop("MEDV", axis = 1, inplace = False)

print("목표변수 데이터 크기: {}".format(ds_house_y.shape))
print("설명변수 데이터 크기: {}".format(ds_house_x.shape))


# In[34]:


#절편 추가
ds_house_x_const = sm.add_constant(ds_house_x)
ds_house_x_const.head()


# In[35]:


#산점도 행렬
sns.pairplot(ds_house)


# In[36]:


#상관관계 분석
ds_house.corr().round(3)


# In[37]:


#회귀 모델 생성
reg_multi_model = sm.OLS(ds_house_y, ds_house_x_const)
#적합
reg_multi_results = reg_multi_model.fit()
print(reg_multi_results.summary())


# In[38]:


#데이터 테이블 생성 -> 값 입력
ds_vif = pd.DataFrame()
#변수 명 입력
ds_vif["variable"] = ds_house_x_const.columns
#variance_inflation_factor 다중공선성 함수, exog: 설명변수 데이터, exog_idx: 변수 인덱스
ds_vif["VIF"] = [variance_inflation_factor(ds_house_x_const.values, i) for i in range(ds_house_x_const.shape[1])]
#VIF 낮은 순 정렬
ds_vif.sort_values("VIF", inplace = True)
ds_vif.round(3)


# In[45]:


#목표변수 : MEDV
import statsmodels.formula.api as smf
formula_model = smf.ols(formula = "MEDV ~ ZN + INDUS + CHAS + AGE + DIS + RAD", data = ds_house)
formula_result = formula_model.fit()
print(formula_result.summary())


# In[46]:


ds_house.corr().round(3)


# In[58]:


ds_agg = ds_house["ZN"].value_counts()
ds_agg.sort_index(inplace= True)
ds_agg.plot.bar()


# In[94]:


#탐색적분석 비소매업 비율
ds_house["INDUS"].hist()


# In[98]:


#탐색적분석 비소매업 비율
ds_house[["INDUS", "NOX","AGE","DIS","RAD","MEDV"]].hist(figsize = (10, 8))


# In[72]:


sns.kdeplot(ds_house["MEDV"],ds_house["DIS"], shade = True)


# In[73]:


sns.kdeplot(ds_house["DIS"],ds_house["MEDV"], shade = True)


# In[91]:


ds_sub = ds_house
ds_count = ds_sub.groupby("CHAS",as_index = False).agg("count")
ds_count.plot.pie(y="MEDV", labels = ds_count["CHAS"], autopct = "%.1f%%")


# In[76]:


#회귀 모델 생성
reg_multi_model = sm.OLS(ds_house_y, ds_house_x_const)
reg_multi_results = reg_multi_model.fit()
print(reg_multi_results.summary())


# In[77]:


ds_vif = pd.DataFrame()
ds_vif["variable"] = ds_house_x_const.columns
ds_vif["VIF"] = [variance_inflation_factor(ds_house_x_const.values, i) for i in range(ds_house_x_const.shape[1])]
ds_vif.sort_values("VIF", inplace = True)
ds_vif.round(3)
## RAD, DIS, AGE, CHAS, ZN, INDUS


# In[99]:


#목표변수 : MEDV
import statsmodels.formula.api as smf
formula_model = smf.ols(formula = "MEDV ~ INDUS + NOX + AGE + DIS + RAD", data = ds_house)
formula_result = formula_model.fit()
print(formula_result.summary())


# In[117]:


ds_house['ZN'].value_counts().idxmax()


# In[107]:


ds_house['CRIM'].value_counts().idxmax()


# In[108]:


ds_house['CHAS'].value_counts().idxmax()


# In[109]:


ds_house['NOX'].value_counts().idxmax()


# In[110]:


ds_house['RM'].value_counts().idxmax()


# In[111]:


ds_house['DIS'].value_counts().idxmax()


# In[112]:


ds_house['RAD'].value_counts().idxmax()


# In[113]:


ds_house['TAX'].value_counts().idxmax()


# In[114]:


ds_house['PTRATIO'].value_counts().idxmax()


# In[115]:


ds_house['B'].value_counts().idxmax()


# In[116]:


ds_house['LSTAT'].value_counts().idxmax()


# In[118]:


25.011146716


# In[ ]:




