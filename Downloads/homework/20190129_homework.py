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


# In[ ]:


#환율.CSV 데이터 가져오기. 파일 이름에 한글 포함되어 있는 경우
#engine = "python" 지정, parse_dates: 날짜 변수 지정
data2 = pd.read_csv('실습화일/환율.csv', engine='python', parse_dates = ["APPL_DATE"])
ds_currency = data2.copy()
ds_currency.head()

