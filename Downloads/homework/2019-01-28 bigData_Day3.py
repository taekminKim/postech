#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from sklearn import linear_model
from sklearn.preprocessing import scale, robust_scale, minmax_scale


# In[3]:


data = pd.read_csv('실습화일/FITNESS.csv', engine='python', encoding='euc-kr')


# In[94]:


ds_fitness = data.copy()
ds_fitness.head()


# In[95]:


ds_fitness.isnull().sum()


# In[96]:


ds_fitness["GENDER"].fillna("여성", inplace = True)
ds_fitness.head()


# In[97]:


ds_fitness.groupby("GENDER")["WEIGHT"].agg("mean")


# In[98]:


ds_fitness.groupby("GENDER")["WEIGHT"].transform("mean").head()


# In[99]:


fit = ds_fitness


# In[100]:


fit.groupby("GENDER")["WEIGHT"].transform("mean").head()


# In[101]:


fit["WEIGHT"] = fit["WEIGHT"].fillna(fit.groupby("GENDER")["WEIGHT"].transform("mean")).round(3)


# In[102]:


fit.head()


# In[103]:


fit_char = fit.select_dtypes(include="object")


# In[104]:


fit_numeric = fit.select_dtypes(exclude="object")


# In[105]:


#page 252
# scale : 데이터 표준화 함수
ds_scale = scale(fit_numeric)
#scale함수를 사용하면 numpy의 형태로 반환되므로 DataFrame으로 변환
ds_scale = pd.DataFrame(ds_scale, columns = fit_numeric.columns)
ds_scale.head()

#pandas.DataFrame.describe():요약통계량
ds_scale_describe = ds_scale.describe()
ds_scale_describe.round(3)


# In[106]:


#page 253 데이터 Scaling 변환(minmax_scale)

#minmax_scale(): 최소 최대값을 이용하여 데이터 반환
ds_minmax_scale = minmax_scale(fit_numeric)
ds_minmax_scale = pd.DataFrame(ds_minmax_scale, columns = fit_numeric.columns)
ds_minmax_scale.head()

#요약 통계량
ds_minmax_scale_describe = ds_minmax_scale.describe()
ds_minmax_scale_describe.round(3)


# In[107]:


#page 254
#robust_scale():데이터 변환 함수
ds_robust_scale = robust_scale(fit_numeric)
ds_robust_scale = pd.DataFrame(ds_robust_scale, columns = fit_numeric.columns)
ds_robust_scale.head()

#요약통계량
ds_robust_scale_describe = ds_robust_scale.describe()
ds_robust_scale_describe.round(3)


# In[108]:


#page 255
#Scale, Robust, MinMax scale 변환 비교
ds_rstpulse = pd.DataFrame()
ds_rstpulse["Raw"] = ds_fitness["RSTPULSE"]
ds_rstpulse["Scale"] = ds_scale["RSTPULSE"]
ds_rstpulse["Robust"] = ds_robust_scale["RSTPULSE"]
ds_rstpulse["MinMax"] = ds_minmax_scale["RSTPULSE"]
ds_rstpulse.round(3)


# In[109]:


#boxplot : 상자 수염도, figsize: plot의 크기(x축, y축)
#이상치 확인 및 처리 : 상자수염도를 이용한 이상치 확인
ds_fitness.boxplot(figsize = (10,4))


# In[110]:


#RSTPULSE 값 중 100보다 큰 값 확인
ds_fitness["RSTPULSE"]>=100

#RSTPULSE 값 중 100보다 작은 값만 가져오기
ds_fitness = ds_fitness[ds_fitness["RSTPULSE"]<100]
ds_fitness


# In[4]:


data


# In[ ]:




