
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


# In[2]:


#환율.CSV 데이터 가져오기. 파일 이름에 한글 포함되어 있는 경우
#engine = "python" 지정, parse_dates: 날짜 변수 지정
data2 = pd.read_csv('실습화일/환율.csv', engine='python', parse_dates = ["APPL_DATE"])
ds_currency = data2.copy()
ds_currency.head()


# In[3]:


#FITNESS.CSV 데이터 가져오기
data = pd.read_csv('실습화일/FITNESS.csv', engine='python', encoding='euc-kr')
ds_fit = data.copy()
ds_fit.head()


# In[4]:


import matplotlib.font_manager as fm
for f in fm.fontManager.ttflist:
    if 'NanumBarunGothic' in f.fname: plt.rcParams['font.family']='NanumBarunGothic'
    # 폰트 나눔고딕으로

print(plt.rcParams['font.family'])

matplotlib.rcParams['axes.unicode_minus'] = False # '-'기호 제대로 나오게


# In[5]:


#page 273
#나이 그룹별로 수를 카운트
ds_agg = ds_fit["AGEGROUP"].value_counts()
#sort_index(inplace = True 정렬사항을 저장): 인덱스 정렬 함수
ds_agg.sort_values(inplace = True, ascending = True)
#bar chart 생성
ds_agg.plot.bar()


# In[6]:


# hist(변수): OXY에 대한 히스토그램 생성)
ds_fit["OXY"].hist()


# In[7]:


#pandas 의 hist 히스토그램 method 이용, figsize = (x축 크기, y축 크기)
ds_fit[["WEIGHT","AGE","OXY","RUNTIME"]].hist(figsize = (10, 4))


# In[8]:


#seaborn사용. FacetGrid(데이터 지정, col = 컬럼 지정): 지정된 컬럼의 범주별 그리드 영역 생성
grid = sns.FacetGrid(ds_fit, col="GENDER",sharex=True, sharey=True)
# map(함수, 문자형 컬럼명): 범주별 히스토그램 생성
grid.map(plt.hist, "OXY")


# In[9]:


#hist(변수, label=라벨스트링): GENDER=남성에 대한 OXY 히스토그램 생성, alpha = 투명도
plt.hist(ds_fit[ds_fit["GENDER"]=="남성"]["OXY"], label = "남성", alpha = 0.5)
#hist(변수, label=라벨스트링): GENDER=여성에 대한 OXY 히스토그램 생성
plt.hist(ds_fit[ds_fit["GENDER"]=="남성"]["OXY"], label = "여성", alpha = 0.5)
#legend: label을 그래프안에 보여줌
plt.legend()


# In[10]:


#hist(변수, data)
plt.hist(x="OXY", data = ds_fit)
# x축 구간 조정(상세 분포 확인)
plt.hist(x="OXY", data = ds_fit, bins=20)


# In[11]:


#boxplot(column=변수, by = 목표변수)
ds_fit.boxplot(column = "WEIGHT", by = "AGEGROUP", grid = False)
#column=None, by=None, ax=None, fontsize=None, rot=0, grid=True, figsize=None, layout=None, return_type=None, **kwds


# In[12]:


#데이터 선택 : 혈당 산소요구량이 50이상
ds_sub = ds_fit[ds_fit["OXY"]>=50]
#groupby(column):column기준 자료 요약, as_index: groupby 변수를 index로 사용 여부. egg("count"): 자료 count
ds_count = ds_sub.groupby("AGEGROUP", as_index = False).agg("count")
#pie(데이터, labels=레이블 변수 지정, autopct=비율 %로 표시(%.1f%%:소수점 1자리 적용)
temp = ds_count.plot.pie(y="OXY", labels = ds_count["AGEGROUP"], autopct = "%.1f%%")
plt.legend(loc=1)


# In[13]:


#scatterplot(X, Y, hue:  그룹 변수, data)
sns.scatterplot(x="RUNTIME", y = "RUNPULSE", hue ="GENDER", data=ds_fit)


# In[14]:


#filter("column",...):분석 대상 변수 나열
ds_filter = ds_fit.filter(["RUNPULSE","MAXPULSE","OXY","RUNTIME"])
#pairplot(데이터): Scatter Plot 그래프 생성
sns.pairplot(ds_filter)


# In[15]:


#통화를 변수로 변환
ds_curreny_pivot = ds_currency.pivot(index = "APPL_DATE", columns = "CURRENCY", values="STD_RATE")
#중국 위안화 별도 그림(엔화, 달러와 단위 차이)
ds_curreny_pivot["CNY"].plot()
#일본 엔화, 미국 달러
ds_curreny_pivot[["JPY", "USD"]].plot()


# In[16]:


#성별과 나이그룹별 혈당 산소 요구량 평균 계산, groupby:(성별, 연령), agg("mean"): 혈당 산소 요구량 평균값
ds_agg = ds_fit.groupby(["GENDER", "AGEGROUP"], as_index = False).agg("mean")
#pivot(성별, 나이그룹, 혈당 산소 요구량): x축=성별, y축=나이그룹, 값 = 혈당산소요구량으로 pivot
ds_pivot = ds_agg.pivot("GENDER", "AGEGROUP", "OXY")
#heatmap(데이터, cmap: 색상)
sns.heatmap(ds_pivot, cmap="Blues")


# In[17]:


#x축, y축 변수 선택, shade:음영 선택 여부
sns.kdeplot(ds_fit["RUNTIME"], ds_fit["RUNPULSE"], shade=False)


# In[18]:


#그래프에 표현할 변수 선택
ds_filter = ds_fit[["GENDER","OXY","WEIGHT","RSTPULSE"]]
#parallel_coordinates(데이터, 분석 column, color=색 지정)
pd.plotting.parallel_coordinates(ds_filter, "GENDER",color=("RED","GREEN"))


# In[19]:


ds_count = ds_fit["AGEGROUP"].value_counts() #Pie Chart를 위한 집계 데이터
fig, axes = plt.subplots(nrows=2, ncols = 2, figsize = (13, 10)) #(2,2) 4분할, 4개의 그래프 생성
plt.tight_layout(w_pad=5, h_pad=5) #w_pad:열 사이 간격, h_pad: 행 사이 간격
#Histogram
axes[0,0].hist(ds_fit["OXY"]) #[0,0]위치에 histogram 생성
axes[0,0].set_title("Histogram", fontsize = 15) #제목 설정, 글자 크기 15
axes[0,0].set_xlabel("혈당 산소 요구량",fontsize = 12) #x축 label 지정
#Pie Chart
axes[0,1].pie(ds_count, labels = ds_count.index.tolist(), autopct = "%.1f%%") #[0,1]위치에 파이 차트 생성
axes[0,1].set_title("Pie Chart", fontsize = 15)
axes[0,1].set_xlabel("나이그룹", fontsize= 12)
#Trend
axes[1,0].plot("RUNTIME", "RUNPULSE", data = ds_fit, label = "맥박(운동)") #[1,0]에 x축:운동 시간, y축: 맥박(운동), label지정
axes[1,0].plot("RUNTIME", "OXY", data = ds_fit, label="혈당 산소 요구량") #x축: 운동시간, y축 : 산소요구량 라벨 지정
axes[1,0].set_title("Trend", fontsize = 15)
axes[1,0].set_xlabel("운동 시간", fontsize = 12)
axes[1,0].legend() #범례 표시
#Scatter
axes[1,1].scatter("RUNPULSE", "OXY", data = ds_fit) #[1,1]위치에 산점도 표시, x축 : 맥박(운동), y축 : 산소 요구량
axes[1,1].set_title("Scatter", fontsize = 16)
axes[1,1].set_xlabel("맥박(운동)", fontsize = 12)
axes[1,1].set_ylabel("혈당 산소 요구량", fontsize = 12)


# In[22]:


##page17
#데이터를 구성하는 패키지
import pandas as pd
#데이터 행렬 연산 패키지
import numpy as np
#데이터 시각화를 위한 패키지
import matplotlib.pyplot as plt
#시각화 패키지
import seaborn as sns
#통계분석 패키지
import scipy.stats as stats
#선형 모델 패키지(절편 추가 위함)
import statsmodels.api as sm
#회귀분석 패키지
from statsmodels.formula.api import ols
#평가함수 패키지
from statsmodels.tools.eval_measures import rmse
#jupyter notebook 사용 시 그래프 자동 출력 옵션
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


#engine: 파일명에 한글 포함되어 있는 경우 "Python"엔진 지정
ds_repair = pd.read_csv("실습화일/부품수리시간.csv", engine="python")
ds_repair


# In[26]:


#목표변수와 설명변수의 산점도
ds_repair.plot.scatter(x="UNITS", y="MINUTES")


# In[27]:


#목표변수와 설명변수의 산점도
ds_repair.corr(method="pearson").round(3)


# In[34]:





# In[28]:


#X, Y변수 역할지정
ds_repair_x = ds_repair["UNITS"]
ds_repair_y = ds_repair["MINUTES"]
#회귀 모델에 상수 추가
ds_repair_x_const = sm.add_constant(ds_repair_x)
#회귀 분석
reg_model = sm.OLS(ds_repair_y, ds_repair_x_const)
reg_results = reg_model.fit()
print(reg_results.summary())


# In[29]:


#잔차산점도
sns.residplot(ds_repair_x, ds_repair_y)


# In[36]:


#정규화
obs = ds_repair_x.values + ds_repair_y.values
z = (obs- np.mean(obs))/np.std(obs)
#정규확률도
stats.probplot(z, dist="norm", plot = plt)


# In[35]:




