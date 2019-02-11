#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
#데이터 분할 함수
from sklearn.model_selection import train_test_split
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


# In[2]:


#ingine: 파일명이 한글일 경우 "Python"엔진 사용
ds_data = pd.read_csv("실습화일/유방암.csv", engine="python")
ds_ub = ds_data.copy()
ds_ub


# In[ ]:




