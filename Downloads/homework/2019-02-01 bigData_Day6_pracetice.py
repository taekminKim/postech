#!/usr/bin/env python
# coding: utf-8

# In[30]:


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


# In[31]:


#1. 분석에 필요한 데이터 구성하기
ds_tele = pd.read_csv("실습화일/통신고객이탈.csv", engine="python")
ds_tel = ds_tele.copy()
#ds_house_x = ds_house.drop("MEDV", axis = 1, inplace = False)
ds_tel.head()
ds_tel["CHURN"].replace(to_replace="Active", value=0, inplace = True)
ds_tel["CHURN"].replace(to_replace="Churned", value=1, inplace = True)


# In[32]:


###의사결정 나무
#sklearn.tree 의 DecisionTreeClassifier : 분류의사결정나무
from sklearn.tree import DecisionTreeClassifier
#sklearn.tree의 export_graphviz : grahviz 패키지가 사용할 수 있는 .dot 확장자 파일로 저장
from sklearn.tree import export_graphviz
#graphviz import : 의사결정 나무 모델 시각화 .dot 확장자 파일 불러오는 패키지
import graphviz


# In[33]:


#결측치 확인
ds_tel.isnull().sum(axis = 0)


# In[50]:


ds_tel_dummy = pd.get_dummies(ds_tel)
ds_tel_dummy = ds_tel.drop("CUSTOMER_ID", axis = 1, inplace = False)
ds_tel_dummy.head()


# In[51]:


#데이터구성하기
np.random.seed(seed=1234)
msk = np.random.rand(ds_tel_dummy.shape[0])<0.7
ds_tel_train = ds_tel_dummy[msk]
ds_tel_test = ds_tel_dummy[~msk]
#train/test 데이터의 목표변수 설명변ㅅ 지정
ds_tel_train_y = ds_tel_train["CHURN"]
ds_tel_train_x = ds_tel_train.drop("CHURN", axis=1, inplace=False)
ds_tel_test_y = ds_tel_test["CHURN"]
ds_tel_test_x = ds_tel_test.drop("CHURN", axis = 1, inplace = False)

print("train data X size : {}".format(ds_tel_train_x.shape))
print("train data Y size : {}".format(ds_tel_train_y.shape))
print("test data X size : {}".format(ds_tel_test_x.shape))
print("test data Y size : {}".format(ds_tel_test_y.shape))


# In[52]:


#데이터 분할 함수
from sklearn.model_selection import train_test_split
#dummy 변수로부터 변수 역할 지정
ds_tel_y = ds_tel_dummy["CHURN"]
ds_tel_x = ds_tel_dummy.drop("CHURN", axis = 1, inplace = False)
#train_test_split
ds_tel_train_x2, ds_tel_test_x2, ds_tel_train_y2, ds_tel_test_y2 =train_test_split(ds_tel_x, ds_tel_y, test_size = 0.30, random_state = 1234)


print("train data X size : {}".format(ds_tel_train_x2.shape))
print("test data Y size : {}".format(ds_tel_test_y2.shape))
print("train data X size : {}".format(ds_tel_train_x2.shape))
print("test data Y size : {}".format(ds_tel_test_y2.shape))


# In[53]:


tree_uncustomized = DecisionTreeClassifier(random_state = 1234)
tree_uncustomized.fit(ds_tel_train_x, ds_tel_train_y)

#훈련 데이터 정확도
print("Accucary on training set : {:.3f}".format(tree_uncustomized.score(ds_tel_train_x, ds_tel_train_y)))
#test 데이터 정확도
print("Accucary on test set: {:.3f}".format(tree_uncustomized.score(ds_tel_test_x, ds_tel_test_y)))


# In[ ]:


#의사결정나무모델 파라미터 조정
#train 및 test정확도 결과 저장용
train_accuracy= []
test_accuracy = []
#적용가능한 criterion: gini, entropy
para_criterion = ["gini", "entropy"]
#para_criterion 별로 트리 모델 생성 및 정확도값 저장
for criterion in para_criterion:
    tree1 = DecisionTreeClassifier(criterion = criterion, random_state = 1234)
    tree1.fit(ds_tel_train_x, ds_tel_train_y)
    train_accuracy.append(tree1.score(ds_tel_train_x, ds_tel_train_y))
    test_accuracy.append(tree1.score(ds_tel_test_x, ds_tel_test_y))

ds_accuracy1 = pd.DataFrame()
ds_accuracy1["Criterion"] = para_criterion
ds_accuracy1["TrainAccuracy"] = train_accuracy
ds_accuracy1["TestAccuracy"] = test_accuracy
ds_accuracy1.round(3)


# In[54]:


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
        tree2.fit(ds_tel_train_x, ds_tel_train_y)
        
        train_accuracy.append(tree2.score(ds_tel_train_x, ds_tel_train_y))
        test_accuracy.append(tree2.score(ds_tel_test_x, ds_tel_test_y))
        para_criterion.append(criterion)
        para_max_depth.append(depth)


# In[55]:


#데이터 테이블로 저장
ds_accuracy2 = pd.DataFrame()
ds_accuracy2["Criterion"] = para_criterion
ds_accuracy2["Depth"] = para_max_depth
ds_accuracy2["TrainAccuracy"] = train_accuracy
ds_accuracy2["TestAccuracy"] = test_accuracy
ds_accuracy2.round(3)


# In[56]:


ds_accuracy2_melt = pd.melt(ds_accuracy2, id_vars = ["Criterion", "Depth"])
ds_accuracy2_melt["Accuracy"] = ds_accuracy2_melt["Criterion"] + "_" + ds_accuracy2_melt["variable"]
sns.lineplot(x="Depth", y = "value", hue="Accuracy", data = ds_accuracy2_melt)


# In[57]:


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
        tree3.fit(ds_tel_train_x, ds_tel_train_y)
        train_accuracy.append(tree3.score(ds_tel_train_x, ds_tel_train_y))
        test_accuracy.append(tree3.score(ds_tel_test_x, ds_tel_test_y))
        para_criterion.append(criterion)
        para_max_depth.append(max_depth)
        para_min_leaf_size.append(leafsize)


# In[58]:


#데이터테이블로 저장
#데이터 테이블로 저장
ds_accuracy3 = pd.DataFrame()
ds_accuracy3["Criterion"] = para_criterion
ds_accuracy3["Depth"] = para_max_depth
ds_accuracy3["MinLeafSize"] = para_min_leaf_size
ds_accuracy3["TrainAccuracy"] = train_accuracy
ds_accuracy3["TestAccuracy"] = test_accuracy
ds_accuracy3.round(3)


# In[59]:


#정확도를 그래프로 표현
ds_accuracy3_melt = pd.melt(ds_accuracy3, id_vars = ["Criterion", "Depth", "MinLeafSize"])
ds_accuracy3_melt["Accuracy"] = ds_accuracy3_melt["Criterion"]+"_"+ds_accuracy3_melt["variable"]
sns.lineplot(x="MinLeafSize", y="value", hue="Accuracy", data = ds_accuracy3_melt)


# In[60]:


#graphviz 패키지로 트리 모델 시각화
tree = DecisionTreeClassifier(criterion = "gini", max_depth= 4, random_state = 1234)
tree.fit(ds_tel_train_x, ds_tel_train_y)

export_graphviz(tree, out_file = "tree.dot", class_names=["0","1"], feature_names = ds_tel_train_x.columns, impurity=False, filled = True)

os.environ["PATH"]+= os.pathsep + "./.wine/drive_c/Program Files (x86)/Graphviz2.38/bin/"

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[61]:


tree4 = DecisionTreeClassifier(criterion= "gini", max_depth = 4, min_samples_leaf = 50, random_state = 1234)
tree5 = tree4.fit(ds_tel_train_x, ds_tel_train_y)

export_graphviz(tree5, out_file = "tree2.dot", class_names = ["0","1"], feature_names = ds_tel_train_x.columns, impurity=False, filled = True)

with open("tree2.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[62]:


tree4 = DecisionTreeClassifier(criterion = "gini", max_depth = 4, min_samples_leaf = 200, random_state = 1234)
tree5 = tree4.fit(ds_tel_train_x, ds_tel_train_y)

export_graphviz(tree5, out_file = "tree3.dot", class_names = ["0", "1"], feature_names = ds_tel_train_x.columns, impurity = False, filled = True)

with open("tree3.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[48]:


ds_feature_importance = pd.DataFrame()
ds_feature_importance["Feature"] = ds_tel_train_x.columns
ds_feature_importance["Importance"] = tree.feature_importances_
ds_feature_importance.sort_values("Importance", ascending = False)
ds_feature_importance.round(3)


# In[49]:


def plot_feature_importances(model):
    n_features = ds_tel_train_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), ds_tel_train_x.columns)
    plt.xlabel("설명변수 중요도")
    plt.ylabel("설명변수")
    plt.ylim(-1, n_features)
    
plot_feature_importances(tree)


# In[ ]:




