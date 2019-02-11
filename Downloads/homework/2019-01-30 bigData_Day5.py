#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


#ingine: 파일명이 한글일 경우 "Python" 엔진 사용
ds_health = pd.read_csv("실습화일/체질검사.csv", engine="python")
ds_health.head()


# In[4]:


#목표변수, 설명변수 역할 지정
ds_health_y = ds_health["FAT"]
ds_health_x = ds_health.drop("FAT", axis = 1, inplace = False)

print("목표변수 데이터 크기: {}".format(ds_health_y.shape))
print("설명변수 데이터 크기: {}".format(ds_health_x.shape))

#절편 추가
ds_health_x_const = sm.add_constant(ds_health_x)
ds_health_x_const.head()


# In[5]:


#산점도 행렬
sns.pairplot(ds_health)


# In[6]:


#상관관계 분석
ds_health.corr().round(3)


# In[7]:


#회귀 모델 생성
reg_multi_model = sm.OLS(ds_health_y, ds_health_x_const)
#적합
reg_multi_results = reg_multi_model.fit()
print(reg_multi_results.summary())


# In[8]:


#데이터 테이블 생성 -> 값 입력
ds_vif = pd.DataFrame()
#변수 명 입력
ds_vif["variable"] = ds_health_x_const.columns
#variance_inflation_factor 다중공선성 함수, exog: 설명변수 데이터, exog_idx: 변수 인덱스
ds_vif["VIF"] = [variance_inflation_factor(ds_health_x_const.values, i) for i in range(ds_health_x_const.shape[1])]
#VIF 낮은 순 정렬
ds_vif.sort_values("VIF", inplace = True)
ds_vif.round(3)


# In[9]:


#RFE(recursive feature elimination) 함수 : 입력된 모델을 이용하여 변수중요도가 낮은 순으로 순차적으로 제거
# sklearn의 선형회귀 모델
model = LinearRegression()
#전체의 변수 중 5개의변수가 남을 때까지 변수 제거
rfe = RFE(estimator = model, n_features_to_select = 5).fit(ds_health_x, ds_health_y)
#선택된 변수
selected_cols = ds_health_x.columns[rfe.support_]
#제거된 변수
removed_cols = ds_health_x.columns[~rfe.support_]

print("Selected Variables : {}".format(selected_cols))
print("Removed Variables : {}".format(removed_cols))


# In[10]:


#후진제거법에 의하여 선택된 데이터
ds_health_x_rfe = sm.add_constant(ds_health_x_const[selected_cols])
#회귀 모델 생성
reg_multi_rfe_model = sm.OLS(ds_health_y, ds_health_x_rfe)
#적합
reg_multi_rfe_results = reg_multi_rfe_model.fit()
print(reg_multi_rfe_results.summary())


# In[11]:


#후진제거법에 의하여 선택된 데이터
ds_rfe_vif = pd.DataFrame()
ds_rfe_vif["variable"] = ds_health_x_rfe.columns
ds_rfe_vif["VIF"] = [variance_inflation_factor(ds_health_x_rfe.values, i) for i in range(ds_health_x_rfe.shape[1])]
ds_rfe_vif.sort_values("VIF", inplace = True)
ds_rfe_vif.round(3)


# In[12]:


#설명 변수 중요도 (표준화 적용 이전)
ds_reg_coef = pd.DataFrame({"Coef": reg_multi_rfe_results.params.values[1:]}, index = reg_multi_rfe_model.exog_names[1:])
ds_reg_coef.plot.barh(y="Coef", legend = False)


# In[13]:


#데이터 표준화, 평균 0, 표준편차 1
ds_health_x_scaled = scale(ds_health_x_const[selected_cols])
#후진제거법에 의하여 선택된 데이터에 상수 1 입력
ds_health_x_scaled = sm.add_constant(ds_health_x_scaled)
#회귀 모델 생성
reg_multi_scaled_model = sm.OLS(ds_health_y, ds_health_x_scaled)
#적합
reg_multi_scaled_results = reg_multi_scaled_model.fit()
#설명변수 중요도
ds_reg_scaled_coef = pd.DataFrame({"Coef": reg_multi_scaled_results.params.values[1:]},
                                 index = reg_multi_rfe_model.exog_names[1:])
ds_reg_scaled_coef.plot.barh(y = "Coef", legend = False)


# In[14]:


#설명변수 중요도
import statsmodels.formula.api as smf
#목표변수 : FAT, 설명변수 : NECK, ABDOMEN, FOREARM, WRIST
formula_model = smf.ols(formula = "FAT ~ NECK + ABDOMEN + FOREARM + WRIST+HIP", data= ds_health)
formula_result = formula_model.fit()
print(formula_result.summary())


# In[21]:


######page43
###### 당뇨병 발병 원인과 관련된 데이터 실습
#1. 분석에 필요한 데이터 구성하기
#1-1. 분석에 필요한 데이터 불러오기
ds_dang = pd.read_csv("실습화일/DIABETES.csv", engine = "python")
#ds_dang.GENDER = ds_dang.GENDER.agg(lambda e : 'F' if e==1 else 'M')
ds_dang.rename(columns={"CHOLESTEROL":"TOTAL CHOLESTEROL"}, inplace = True)
ds_dang.head()


# In[22]:


#1-2. 데이터 역할 지정
#목표변수, 설명변수 역할 지정
ds_dang_y = ds_dang["Y"]
ds_dang_x = ds_dang.drop("Y", axis = 1, inplace = False)
print("목표변수 데이터크기: {}".format(ds_health_y.shape))
print("설명변수 데이터크기: {}".format(ds_health_x.shape))


# In[23]:


#절편 추가
ds_dang_x_const = sm.add_constant(ds_dang_x)
ds_dang_x_const.head()
ds_dang


# In[24]:


#2.변수 간의 경향성 파악 - 그래프 분석
sns.pairplot(ds_dang)


# In[25]:


#2-2 변수 간의 경향성 파악 - 상관관계 분석
#상관관계 분석
ds_dang.corr().round(3)


# In[26]:


#3.회귀 모델 생성
reg_multi_model = sm.OLS(ds_dang_y, ds_dang_x_const)
#적합
reg_multi_results = reg_multi_model.fit()
print(reg_multi_results.summary())


# In[27]:


#3. 회귀모델생성 : 전체 설명변수 대상으로 다중 공선성 검토
ds_vif = pd.DataFrame()
#변수명 입력
ds_vif["variable"] = ds_dang_x_const.columns
#variance_inflation_factor 다중공선성 함수, exog: 설명변수 데이터, exog_idx: 변수 인덱스
ds_vif["VIF"] = [variance_inflation_factor(ds_dang_x_const.values, i) for i in range(ds_dang_x_const.shape[1])]
#VIF 낮은 순 정렬
ds_vif.sort_values("VIF", inplace = True)
ds_vif.round(3)


# In[28]:


#3-1 후진제거법을 이용하여 변수 선택
# RFE(recursive feature elimination) 함수: 입력된 모델을 이용하여 변수중요도가 낮은 순으로 순차적으로 제거
# sklearn 의 선형회귀 모델
model = LinearRegression()
# 전체 변수중 5개의 변수가 남을때까지 변수 제거
rfe = RFE(estimator = model, n_features_to_select = 5).fit(ds_dang_x, ds_dang_y)
#선택된 변수
selected_cols = ds_dang_x.columns[rfe.support_]
#제거된 변수
removed_cols = ds_dang_x.columns[~rfe.support_]
print("selected variables : {}".format(selected_cols))
print("removed variables : {}".format(removed_cols))


# In[29]:


#후진제거법을 이용한 변수 선택 - 회귀 모델 선택
ds_dang_x_rfe = sm.add_constant(ds_dang_x_const[selected_cols])
#회귀 모델 생성
reg_multi_rfe_model = sm.OLS(ds_dang_y, ds_dang_x_rfe)
#적합
reg_multi_rfe_results = reg_multi_rfe_model.fit()
print(reg_multi_rfe_results.summary())


# In[30]:


#3-4 후진제거거법을 이용한 변수 선택 - 선택된 설명변수 대상으로 다중 공선성 진단
#후진제거법에 의하여 선택된 데이터
ds_rfe_vif = pd.DataFrame()
ds_rfe_vif["variable"] = ds_dang_x_rfe.columns
ds_rfe_vif["VIF"] = [variance_inflation_factor(ds_dang_x_rfe.values, i) for i in range(ds_dang_x_rfe.shape[1])]
ds_rfe_vif.sort_values("VIF", inplace = True)
ds_rfe_vif.round(3)


# In[32]:



#page 59 시계열 분석 연습
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#자기 상관함수, 부분 자기상관함수
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#ARIMA
from statsmodels.tsa.arima_model import ARIMA
#날짜 데이터 생성 패키지
from datetime import datetime


# In[33]:


#분석에 필요한 데이터 구성하기
ds_currency = pd.read_csv("실습화일/환율.csv", engine = "python", parse_dates = ["APPL_DATE"])
ds_currency.head()


# In[34]:


#변동 추세 확인 - 시간에 따른 환율 추세 그래프 생성
#중국 위엔화, 엔화, 달러의 크기가 다르기 때문에 각각 그래프를 생성 -> 3행 1열, 크기 : (10:8)
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, figsize = (10, 8))
#zip 함수를 사용하면 zip 함수 안의 리스트들을 하나씩 배출
# 1번 loop : ax1, 311, "CNY"
# 2번 loop : ax2, 312, "JPY"
# 3번 loop : ax3, 313, "USD"
for(ax, idx, currency) in zip([ax1, ax2, ax3], [311, 312, 313], ["CNY", "JPY", "USD"]):
    #plot 추가, 311 -> 3행 1열의 1번 그래프
    ax.plot("APPL_DATE", "STD_RATE", data = ds_currency[ds_currency["CURRENCY"] == currency])
    # y축에 통화 표시 -> rotation: label 가로로 생성, labelpad: label과 그래프 사이의 거리
    ax.set_ylabel(currency, rotation = 0, labelpad = 20)


# In[35]:


#시계열 분석 : 데이터 분할 - 통화별 데이터 분할
#CNY 데이터 추출 후 APPL_DATE를 index로 설정
ds_currency_CNY = ds_currency[ds_currency["CURRENCY"] == "CNY"].set_index("APPL_DATE")
#drop method를 이용하여 통화(CURRENCY)와 미달러 환산율(USD_CONV_DATE)변수 제외
ds_currency_CNY.drop(["CURRENCY","USD_CONV_DATE"], axis = 1, inplace = True)
#JPY 데이터 생성
ds_currency_CNY = ds_currency[ds_currency["CURRENCY"] == "JPY"].set_index("APPL_DATE")
ds_currency_CNY.drop(["CURRENCY","USD_CONV_DATE"], axis = 1, inplace = True)
#USD 데이터 생성
ds_currency_CNY = ds_currency[ds_currency["CURRENCY"] == "USD"].set_index("APPL_DATE")
ds_currency_CNY.drop(["CURRENCY","USD_CONV_DATE"], axis = 1, inplace = True)
ds_currency_CNY.head()


# In[36]:


#시계열 분석 : 데이터 분할 - 차분을 통하여 정상성 데이터로 변환
#1번 차분 : z(t) - z(t-1)
#shift(n) : 앞의 n번 째 행의 데이터를 가져옴
ds_cny_diff1 = ds_currency_CNY["STD_RATE"] - ds_currency_CNY["STD_RATE"].shift(1)
ds_cny_diff1.plot()

#2번 차분 : (z(t) - z(t-1)) - (z(t-1) - z(t-1)) = z(t) - 2*(t-1) + z(t-2)
#코드가 길어질 겨웅 가독성을 위하여 연산자 뒤에 "\"표시를 붙임으로써 하나의 문장을 두 문장으로 나눌 수 있음
ds_cny_diff2 = ds_currency_CNY["STD_RATE"] - 2*(ds_currency_CNY["STD_RATE"].shift(1)) +(ds_currency_CNY["STD_RATE"].shift(2))
ds_cny_diff2.plot()


# In[37]:


#시계열 분석 - 자기상관 함수 및 부분 자기상관 함수
lag_size = 30
fig = plt.figure(figsize = (12, 8))
#acf 그래프를 그릴 공간 생성
ax1 = fig.add_subplot(211)
#자기상관 함수 그래프 plot_acf 함수 사용 -> 위에 생성한 공간에 그래프 넣기
fig = plot_acf(ds_currency_CNY["STD_RATE"], lags = lag_size, ax = ax1)
#pacf 그래프를 그릴 공간 생성
ax2 = fig.add_subplot(212)
#부분 자기상관 함수 그래프 plot_pacf 함수 사용 -> 위에 생성한 공간에 그래프 넣기
fig = plot_pacf(ds_currency_CNY["STD_RATE"], lags = lag_size, ax = ax2)


# In[38]:


#시계열 분석 - 시계열 분석
#AR(1), I(2, 차분), MA(0)인 ARIMA 모델
ts_model_cny = ARIMA(ds_currency_CNY, order = (1, 2, 0))
#데이터 적합
# trend: 상수 포함 여부 "nc"이면 상수 미포함, full_output: 모든 출력 결과 표시, disp: 수렴 정보 출력
ts_result_cny = ts_model_cny.fit(trend="c", full_output= True, disp = 1)
print(ts_result_cny.summary())


# In[39]:


#예측 날짜 생성
start_time = datetime.strptime("2016-03-25T00:00:00", "%Y-%m-%dT%H:%M:%S")
end_time = datetime.strptime("2016-04-05T00:00:00", "%Y-%m-%dT%H:%M:%S")
fig, ax = plt.subplots(figsize = (12, 8))
ax = ds_currency_CNY.plot(ax = ax)
fig = ts_result_cny.plot_predict(start = start_time, end= end_time, ax = ax, plot_insample = False)


# In[ ]:




