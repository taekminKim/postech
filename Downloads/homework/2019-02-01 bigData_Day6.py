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
#평가 함수
from sklearn import metrics
#os: 환겨설정 패키지
import os 


# In[9]:


ds_hmeq = pd.read_csv("실습화일/HMEQ.csv", engine="python")
ds_hmeq.head()


# In[10]:


#결측치 확인
ds_hmeq.isnull().sum(axis = 0)


# In[11]:


#fillna : 결측치를 채우는 함수
# JOB 변수의 결측치는 Other로 입력, inplace : fillna 함수적용 후 ds_hmeq 데이터에 저장, False 면 저장 안함
ds_hmeq["JOB"].fillna("Other", inplace = True)
#숫자형 변수의 결측치는 해당 변수의 평균값 입력: ds_hmeq.mean() 각 변수별 평균 계산 후 결측치 대체
ds_hmeq.fillna(ds_hmeq.mean(), inplace = True)


# In[12]:


#random seed 고정 -> 매번 실행 시 같은 값을 얻음
np.random.seed(seed = 1234)
# 0.7(70%) 보다 작으면 True -> train 데이터, 아니면 False -> test ㄷ이터
msk = np.random.rand(ds_hmeq.shape[0],) <0.7
ds_hmeq_train  = ds_hmeq[msk]
ds_hmeq_test  = ds_hmeq[~msk]
#train 데이터와 test 데이터 크기
print("train data size : {}".format(ds_hmeq_train.shape))
print("test data size : {}".format(ds_hmeq_test.shape))


# In[13]:


#train 데이터에 상수 추가
ds_hmeq_train = sm.add_constant(ds_hmeq_train)
ds_hmeq_train.head()


# In[15]:


#from_formula 함수를 이용하여 변수 역할 지정
log_model = sm.Logit.from_formula("""BAD ~ LOAN + MORTDUE + VALUE + C(REASON) + C(JOB) + YOJ + DEROG + DELINQ + CLAGE + NINQ + CLNO + DEBTINC + 1""", ds_hmeq_train)
#적합
log_result = log_model.fit()
#결과 출력
print(log_result.summary())


# In[16]:


#로지스틱 모델로 test데이터 예측
y_pred = log_result.predict(ds_hmeq_test)
# 0과 1의 값을 가진 class로 변환
y_pred_class = (y_pred > 0.5).astype(int)
#상위 5건 출력
y_pred_class.head()


# In[18]:


#설명변수 중요도
ds_logistic_coef = pd.DataFrame({"Coef": log_result.params.values[1:]}, index = log_model.exog_names[1:])
ds_logistic_coef.plot.barh(y="Coef")


# In[19]:


#select_dtypes : 특정 변수 타입을 선택/제외하여 데이터 추출
ds_hmeq_char = ds_hmeq.select_dtypes(include = "object")
ds_hmeq_numeric = ds_hmeq.select_dtypes(exclude = "object")
#Data Scale
ds_hmeq_char_train = ds_hmeq_char[msk]
ds_hmeq_numeric_scaled_train = scale(ds_hmeq_numeric[msk])
ds_hmeq_numeric_scaled_train = pd.DataFrame(ds_hmeq_numeric_scaled_train, columns = ds_hmeq_numeric.columns)
ds_hmeq_numeric_scaled_train.head()


# In[21]:


#BAD 데이터를 0과 1로 변환
ds_hmeq_numeric_scaled_train["BAD"] = np.where(ds_hmeq_numeric_scaled_train["BAD"] > 0,1,0)
ds_hmeq_numeric_scaled_train.head()

#scale된 숫자형 데이터와 범주형 데이터 결합
ds_hmeq_scaled_train = pd.concat([ds_hmeq_numeric_scaled_train, ds_hmeq_char_train.reset_index(drop=True)], axis = 1)


# In[22]:


#from_formula 함수를 이용하여 변수 역할 지정
scaled_log_model = sm.Logit.from_formula("""BAD ~ LOAN + MORTDUE + VALUE + C(REASON) + C(JOB) + YOJ + DEROG + DELINQ + CLAGE + NINQ + CLNO + DEBTINC + 1""", ds_hmeq_scaled_train)

#적합
scaled_log_result = scaled_log_model.fit()
#설명변수 중요도
ds_log_scaled_coef = pd.DataFrame({"Coef":scaled_log_result.params.values[1:]}, index = scaled_log_model.exog_names[1:])
ds_log_scaled_coef.plot.barh(y="Coef", legend = False)


# In[108]:


#1. 분석에 필요한 데이터 구성하기
ds_tele = pd.read_csv("실습화일/통신고객이탈.csv", engine="python")
ds_tel = ds_tele.copy()
ds_tel.head()


# In[109]:


#2. 데이터 전처리 - 결측치 처리
ds_tel.isnull().sum(axis = 0)
# 결측치가 없음
ds_tel["CHURN"].replace(to_replace="Active", value=0, inplace = True)
ds_tel["CHURN"].replace(to_replace="Churned", value=1, inplace = True)


# In[110]:


#2. 데이터 전처리 - 데이터 분할
#random seed 고정 -> 매번 실행 시 같은 값을 얻음
np.random.seed(seed = 1234)
#0.7(70%) 보다 작으면 True -> train 데이터, 아니면 False -> test데이터
msk = np.random.rand(ds_tel.shape[0],) < 0.7
ds_tel_train = ds_tel[msk]
ds_tel_test = ds_tel[~msk]
#train 데이터와 test 데이터 크기
print("train data size : {}".format(ds_tel_train.shape))
print("test data size : {}".format(ds_tel_test.shape))


# In[111]:


#train 데이터에 상수 추가
ds_tel_train = sm.add_constant(ds_tel_train)
ds_tel_train.head()
#회귀모델 및 회귀계수 검토


# In[113]:



#from_formula 함수를 이용하여 변수 역할 지정
log_model = sm.Logit.from_formula("""CHURN ~ C(GENDER) + AGE + C(CHARGE_TYPE) + C(HANDSET) + C(USAGE_BAND) + SERVICE_DURATION + DROPPED_CALLS + PEAK_CALLS_NO + PEAK_CALLS_TIME + WEEKEND_CALLS_NO + WEEKEND_CALLS_TIME + TOTAL_CALLS_NO + TOTAL_CALLS_TIME + 1""", ds_tel_train)
#적합
log_result = log_model.fit(method= 'bfgs')
#결과 출력
print(log_result.summary())


# In[114]:


#로지스틱 모델로 test 데이터 예측
y_pred = log_result.predict(ds_tel_test)
#0과 1의 값을 가진 class로 변환
y_pred_class = (y_pred  >0.5).astype(int)
#상위 5건 출력
y_pred_class.head()


# In[115]:


print("Accuracy:{0:.3f}".format(metrics.accuracy_score(ds_tel_test["CHURN"], y_pred_class)))


# In[116]:


#설명변수 중요도
ds_logistic_coef = pd.DataFrame({"Coef":log_result.params.values[1:]}, index = log_model.exog_names[1:])
ds_logistic_coef.plot.barh(y = "Coef")


# In[117]:


#결론 도출
#select_dtypes: 특정 변수 타입을 선택/제외하여 데이터 추출
ds_tel_char = ds_tel.select_dtypes(include = "object")
ds_tel_numeric = ds_tel.select_dtypes(exclude = "object")
#Data Scale
ds_tel_char_train = ds_tel_char[msk]
ds_tel_numeric_scaled_train = scale(ds_tel_numeric[msk])
ds_tel_numeric_scaled_train = pd.DataFrame(ds_tel_numeric_scaled_train, columns = ds_tel_numeric.columns)
ds_tel_numeric_scaled_train.head()


# In[119]:


#CHURN 데이터를 0과 1로 반환
ds_tel_numeric_scaled_train["CHURN"] = np.where(ds_tel_numeric_scaled_train["CHURN"] > 0, 1, 0)
ds_tel_numeric_scaled_train.head()

#scale된 숫자형 데이터와 범주형 데이터 결합
ds_tel_scaled_train = pd.concat([ds_tel_numeric_scaled_train, ds_tel_char_train.reset_index(drop= True)], axis = 1)


# In[122]:


#from_formula 함수를 이용하여 변수 역할 지정
scaled_log_model = sm.Logit.from_formula("""CHURN ~ C(GENDER) + AGE + C(CHARGE_TYPE) + C(HANDSET) + C(USAGE_BAND) + SERVICE_DURATION + DROPPED_CALLS + PEAK_CALLS_NO + PEAK_CALLS_TIME + WEEKEND_CALLS_NO + WEEKEND_CALLS_TIME + TOTAL_CALLS_NO + TOTAL_CALLS_TIME + 1""", ds_tel_scaled_train)
#적합
scaled_log_result = scaled_log_model.fit(method='bfgs')
#설명변수 중요도
ds_log_scaled_coef  =pd.DataFrame({"Coef":scaled_log_result.params.values[1:]}, index = scaled_log_model.exog_names[1:])
ds_log_scaled_coef.plot.barh(y = "Coef", legend = False)


# In[1]:


###의사결정 나무
#sklearn.tree 의 DecisionTreeClassifier : 분류의사결정나무
from sklearn.tree import DecisionTreeClassifier
#sklearn.tree의 export_graphviz : grahviz 패키지가 사용할 수 있는 .dot 확장자 파일로 저장
from sklearn.tree import export_graphviz
#graphviz import : 의사결정 나무 모델 시각화 .dot 확장자 파일 불러오는 패키지
import graphviz


# In[3]:


ds_hmeq = pd.read_csv("실습화일/HMEQ.csv")
ds_hmeq.head()


# In[4]:


#결측치 확인
ds_hmeq.isnull().sum(axis = 0)


# In[5]:


#fillna : 결측치 채우는 함수
ds_hmeq["JOB"].fillna("Other", inplace = True)
ds_hmeq.fillna(ds_hmeq.mean(), inplace = True)


# In[6]:


ds_hmeq_dummy = pd.get_dummies(ds_hmeq)
ds_hmeq_dummy.head()


# In[8]:


#데이터구성하기
np.random.seed(seed=1234)
msk = np.random.rand(ds_hmeq_dummy.shape[0])<0.7
ds_hmeq_train = ds_hmeq_dummy[msk]
ds_hmeq_test = ds_hmeq_dummy[~msk]
#train/test 데이터의 목표변수 설명변ㅅ 지정
ds_hmeq_train_y = ds_hmeq_train["BAD"]
ds_hmeq_train_x = ds_hmeq_train.drop("BAD", axis=1, inplace=False)
ds_hmeq_test_y = ds_hmeq_test["BAD"]
ds_hmeq_test_x = ds_hmeq_test.drop("BAD", axis = 1, inplace = False)

print("train data X size : {}".format(ds_hmeq_train_x.shape))
print("train data Y size : {}".format(ds_hmeq_train_y.shape))
print("test data X size : {}".format(ds_hmeq_test_x.shape))
print("test data Y size : {}".format(ds_hmeq_test_y.shape))


# In[11]:


#데이터 분할 함수
from sklearn.model_selection import train_test_split
#dummy 변수로부터 변수 역할 지정
ds_hmeq_y = ds_hmeq_dummy["BAD"]
ds_hmeq_x = ds_hmeq_dummy.drop("BAD", axis = 1, inplace = False)
#train_test_split
ds_hmeq_train_x2, ds_hmeq_test_x2, ds_hmeq_train_y2, ds_hmeq_test_y2 =train_test_split(ds_hmeq_x, ds_hmeq_y, test_size = 0.30, random_state = 1234)


print("train data X size : {}".format(ds_hmeq_train_x2.shape))
print("test data Y size : {}".format(ds_hmeq_test_y2.shape))
print("train data X size : {}".format(ds_hmeq_train_x2.shape))
print("test data Y size : {}".format(ds_hmeq_test_y2.shape))


# In[12]:


tree_uncustomized = DecisionTreeClassifier(random_state = 1234)
tree_uncustomized.fit(ds_hmeq_train_x, ds_hmeq_train_y)

#훈련 데이터 정확도
print("Accucary on training set : {:.3f}".format(tree_uncustomized.score(ds_hmeq_train_x, ds_hmeq_train_y)))
#test 데이터 정확도
print("Accucary on test set: {:.3f}".format(tree_uncustomized.score(ds_hmeq_test_x, ds_hmeq_test_y)))


# In[13]:


#의사결정나무모델 파라미터 조정
#train 및 test정확도 결과 저장용
train_accuracy= []
test_accuracy = []
#적용가능한 criterion: gini, entropy
para_criterion = ["gini", "entropy"]
#para_criterion 별로 트리 모델 생성 및 정확도값 저장
for criterion in para_criterion:
    tree1 = DecisionTreeClassifier(criterion = criterion, random_state = 1234)
    tree1.fit(ds_hmeq_train_x, ds_hmeq_train_y)
    train_accuracy.append(tree1.score(ds_hmeq_train_x, ds_hmeq_train_y))
    test_accuracy.append(tree1.score(ds_hmeq_test_x, ds_hmeq_test_y))

ds_accuracy1 = pd.DataFrame()
ds_accuracy1["Criterion"] = para_criterion
ds_accuracy1["TrainAccuracy"] = train_accuracy
ds_accuracy1["TestAccuracy"] = test_accuracy
ds_accuracy1.round(3)
    


# In[14]:


#의사결정나무 모델 파라미터 조정
#train 및 test 정확도 결과저장용
train_accuracy = []; test_accuracy = []
#parameter 결과 테이블을 위함
para_criterion = []; para_max_depth = []
#최대 깊이 1~10까지 순차 실행
n_iter_depth = 10
#적용가능한 criterion : gini, entropy
list_criterion = ["gini", "entropy"]

for criterion in list_criterion:
    for depth in range(1, n_iter_depth+1):
        tree2 = DecisionTreeClassifier(criterion = criterion, max_depth = depth, random_state =1234)
        tree2.fit(ds_hmeq_train_x, ds_hmeq_train_y)
        
        train_accuracy.append(tree2.score(ds_hmeq_train_x, ds_hmeq_train_y))
        test_accuracy.append(tree2.score(ds_hmeq_test_x, ds_hmeq_test_y))
        para_criterion.append(criterion)
        para_max_depth.append(depth)


# In[15]:


#데이터 테이블로 저장
ds_accuracy2 = pd.DataFrame()
ds_accuracy2["Criterion"] = para_criterion
ds_accuracy2["Depth"] = para_max_depth
ds_accuracy2["TrainAccuracy"] = train_accuracy
ds_accuracy2["TestAccuracy"] = test_accuracy
ds_accuracy2.round(3)


# In[17]:


ds_accuracy2_melt = pd.melt(ds_accuracy2, id_vars = ["Criterion", "Depth"])
ds_accuracy2_melt["Accuracy"] = ds_accuracy2_melt["Criterion"] + "_" + ds_accuracy2_melt["variable"]
sns.lineplot(x="Depth", y = "value", hue="Accuracy", data = ds_accuracy2_melt)


# In[18]:


#의사결정나무 모델 파라미터 조정
#train 및 test 정확도 결과저장용
train_accuracy = []; test_accuracy = []
#parameter 결과 테이블을 위함
para_criterion = []; para_max_depth = []; para_min_leaf_size = []
#최대 깊이 1~10까지 순차 실행
max_depth = 10
list_min_leaf_size = [i*10 for i in range(1, 6)]

for criterion in list_criterion:
    for leafsize in list_min_leaf_size:
        tree3 = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, min_samples_leaf = leafsize, random_state = 1234)
        tree3.fit(ds_hmeq_train_x, ds_hmeq_train_y)
        train_accuracy.append(tree3.score(ds_hmeq_train_x, ds_hmeq_train_y))
        test_accuracy.append(tree3.score(ds_hmeq_test_x, ds_hmeq_test_y))
        para_criterion.append(criterion)
        para_max_depth.append(max_depth)
        para_min_leaf_size.append(leafsize)


# In[19]:


#데이터테이블로 저장
#데이터 테이블로 저장
ds_accuracy3 = pd.DataFrame()
ds_accuracy3["Criterion"] = para_criterion
ds_accuracy3["Depth"] = para_max_depth
ds_accuracy3["MinLeafSize"] = para_min_leaf_size
ds_accuracy3["TrainAccuracy"] = train_accuracy
ds_accuracy3["TestAccuracy"] = test_accuracy
ds_accuracy3.round(3)


# In[20]:


#정확도를 그래프로 표현
ds_accuracy3_melt = pd.melt(ds_accuracy3, id_vars = ["Criterion", "Depth", "MinLeafSize"])
ds_accuracy3_melt["Accuracy"] = ds_accuracy3_melt["Criterion"]+"_"+ds_accuracy3_melt["variable"]
sns.lineplot(x="MinLeafSize", y="value", hue="Accuracy", data = ds_accuracy3_melt)


# In[21]:


#graphviz 패키지로 트리 모델 시각화
tree = DecisionTreeClassifier(criterion = "gini", max_depth= 4, random_state = 1234)
tree.fit(ds_hmeq_train_x, ds_hmeq_train_y)

export_graphviz(tree, out_file = "tree.dot", class_names=["0","1"], feature_names = ds_hmeq_train_x.columns, impurity=False, filled = True)

os.environ["PATH"]+= os.pathsep + "./.wine/drive_c/Program Files (x86)/Graphviz2.38/bin/"

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[22]:


tree4 = DecisionTreeClassifier(criterion= "gini", max_depth = 4, min_samples_leaf = 50, random_state = 1234)
tree5 = tree4.fit(ds_hmeq_train_x, ds_hmeq_train_y)

export_graphviz(tree5, out_file = "tree2.dot", class_names = ["0","1"], feature_names = ds_hmeq_train_x.columns, impurity=False, filled = True)

with open("tree2.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[23]:


tree4 = DecisionTreeClassifier(criterion = "gini", max_depth = 4, min_samples_leaf = 200, random_state = 1234)
tree5 = tree4.fit(ds_hmeq_train_x, ds_hmeq_train_y)

export_graphviz(tree5, out_file = "tree3.dot", class_names = ["0", "1"], feature_names = ds_hmeq_train_x.columns, impurity = False, filled = True)

with open("tree3.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[24]:


ds_feature_importance = pd.DataFrame()
ds_feature_importance["Feature"] = ds_hmeq_train_x.columns
ds_feature_importance["Importance"] = tree.feature_importances_
ds_feature_importance.sort_values("Importance", ascending = False)
ds_feature_importance.round(3)


# In[26]:


def plot_feature_importances(model):
    n_features = ds_hmeq_train_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), ds_hmeq_train_x.columns)
    plt.xlabel("설명변수 중요도")
    plt.ylabel("설명변수")
    plt.ylim(-1, n_features)
    
plot_feature_importances(tree)


# In[ ]:





# In[ ]:




