#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from sklearn import linear_model


# In[3]:


np.random.seed(seed=1234)
normal = np.random.normal(20, 2, 100000)
sns.distplot(normal)


# In[6]:


np.random.seed(seed=1234)
normal = np.random.normal(20, 2, 100000)
sns.distplot(normal)
normal = stats.normal(20,2)
xx = np.linspace(10, 30, 400)
plt.plot(xx, normal.pdf(xx))


# In[8]:


normal = stats.norm(20,2)
xx = np.linspace(10, 30, 400)
plt.plot(xx, normal.pdf(xx))


# In[9]:


normal = stats.norm(loc=20, scale=2)
xx = np.linspace(10,30,400)
plt.plot(xx, normal.pdf(xx))


# In[14]:


mu = 175
sigma = 5
x = 180
normal4 = stats.norm(mu,sigma)
xx = np.linspace(160,190,400)
plt.plot(xx,normal4.pdf(xx))
v3 = stats.norm.cdf(x, mu, sigma)
print("{0}cm 이상 경우 상위: {1: .1f}%에 해당함".format(x, (1-v3)*100))


# In[16]:


mu = 12
sigma = 3
x1 = 12
x2 = 15
normal5 = stats.norm(mu,sigma)
xx = np.linspace(3,21,400)
plt.plot(xx,normal5.pdf(xx))
v5_1 = stats.norm.cdf(x1, mu, sigma)
v5_2 = stats.norm.cdf(x2, mu, sigma)
print("{0}km 이상 {1}km 이하 달릴확률은: {2: .1f}%에 해당함".format(x1,x2, (v5_2-v5_1)*100))


# In[17]:


mu = 12
sigma = 3
x1 = 9
x2 = 15
normal5 = stats.norm(mu,sigma)
xx = np.linspace(3,21,400)
plt.plot(xx,normal5.pdf(xx))
v5_1 = stats.norm.cdf(x1, mu, sigma)
v5_2 = stats.norm.cdf(x2, mu, sigma)
print("{0}km 이상 {1}km 이하 달릴확률은: {2: .1f}%에 해당함".format(x1,x2, (v5_2-v5_1)*100))


# In[18]:


mu = 12
sigma = 3
x1 = 15
normal5 = stats.norm(mu,sigma)
xx = np.linspace(3,21,400)
plt.plot(xx,normal5.pdf(xx))
v5_1 = stats.norm.cdf(x1, mu, sigma)
print("{0}km 이상 달릴확률은: {1: .1f}%에 해당함".format(x1, (v5_1)*100))


# In[20]:


mu = 0
sigma = 1
x1 = 15
normal5 = stats.norm(mu,sigma)
xx = np.linspace(-100,100,400)
plt.plot(xx,normal5.pdf(xx))
v5_1 = stats.norm.cdf(x1, mu, sigma)
print("{0}km 이상 달릴확률은: {1: .1f}%에 해당함".format(x1, (v5_1)*100))


# In[22]:


mu = 0
sigma = 1
x1 = 15
normal5 = stats.norm(mu,sigma)
xx = np.linspace(-5,5,100)
plt.plot(xx,normal5.pdf(xx))
v5_1 = stats.norm.cdf(x1, mu, sigma)
print("{0}km 이상 달릴확률은: {1: .1f}%에 해당함".format(x1, (v5_1)*100))


# In[23]:


np.random.seed(seed=1234)
xx = np.linspace(-4,4,100)
for df in [1,5,25]:
    t_df=stats.t(df=df)
    plt.plot(xx,t_df.pdf(xx))
plt.plot(xx,stats,norm.pdf(xx), lw=5)


# In[24]:


np.random.seed(seed=1234)
xx = np.linspace(-4,4,100)
for df in [1,5,25]:
    t_df=stats.t(df=df)
    plt.plot(xx,t_df.pdf(xx))
plt.plot(xx,stats.norm.pdf(xx), lw=5)


# In[26]:


np.random.seed(seed=1234)
xx = np.linspace(-4,4,100)
t = 1.53
df = 5
prop = stats.t.cdf(t,df)
print("P(T<={0}) :{1:.3f}".format(t,prop))


# In[27]:


np.random.seed(seed=1234)
xx = np.linspace(-4,4,100)
t = 2.0
df = 10
prop = stats.t.cdf(t,df)
print("P(T<={0}) :{1:.3f}".format(t,prop))


# In[29]:


np.random.seed(seed=1234)
t_df29 = np.random.standard_t(df=29, size =1000)
sns.distplot(t_df29, fit=stats.norm, kde=False)


# In[30]:


np.random.seed(seed=1234)
t_df100 = np.random.standard_t(df=100, size =1000)
sns.distplot(t_df100, fit=stats.norm, kde=False)


# In[33]:


np.random.seed(seed=1234)
t_df10 = np.random.standard_t(df=10, size =1000)
sns.distplot(t_df10, fit=stats.norm, kde=False)
sns.distplot(t_df10, fit=stats.norm, kde=True)
np.random.seed(seed=1234)
xx = np.linspace(-4,4,100)
df = 10
t_df = stats.t(df=df)
plt.plot(xx,t_df.pdf(xx))


# In[39]:


np.random.seed(seed=1234)
xx = np.linspace(0,100,400)
for df in [1,3,5,10,25,50]:
    chi2_df = stats.chi2(df=df)
    plt.plot(xx,chi2_df.pdf(xx))


# In[40]:


np.random.seed(seed=1234)
chisq_df10 = np.random.chisquare(df=10, size=1000)
sns.distplot(chisq_df10, kde=False)


# In[44]:


np.random.seed(seed=1234)
chisq_df10 = np.random.chisquare(df=10, size=1000)
chisq_df40 = np.random.chisquare(df=40, size=1000)
sns.distplot(chisq_df40, fit = stats.norm, kde=True)


# In[45]:


chisq = 10
df = 30
prop = stats.chi2.cdf(chisq, df)
print("P(X<={0}): {1:.4f}".format(chisq,prop))


# In[51]:


chisq = 10
df = 30
prop = stats.chi2.sf(chisq, df)
print("P(X<={0}): {1:.4f}".format(chisq,prop))


# In[59]:


chisq = 10
df = 30
chip1 = stats.chi2.cdf(chisq, df)
chip2 = stats.chi2.sf(chisq, df)
print("P(X<={0}): {1:.4f}".format(chisq,chip1))
print("P(X<={0}): {1:.4f}".format(chisq,chip2))


# In[61]:


np.random.seed(seed=1234)
F_df10_10 = np.random.f(dfnum=10, dfden=10,size=1000)
sns.distplot(F_df10_10, kde=True)


# In[62]:


np.random.seed(seed=1234)
F_df10_10 = np.random.f(dfnum=100, dfden=100,size=1000)
sns.distplot(F_df10_10, kde=True)


# In[65]:


np.random.seed(seed=1234)
xx=np.linspace(0,10,400)
for df in [1,3,5,25,50]:
    f_df = stats.f(dfn=df, dfd=10)
    plt.plot(xx, f_df.pdf(xx))


# In[75]:


np.random.seed(seed=1234)
F_df10_10 = np.random.f(dfnum=20, dfden=20,size=1000)
sns.distplot(F_df10_10, kde=True)
xx=np.linspace(0,10,400)
f_df = stats.f(dfn=20, dfd=20)
plt.plot(xx, f_df.pdf(xx))


# In[91]:


np.random.seed(seed=1234)
f = 2
dfnum = 15
dfden = 15
xx = np.linspace(0,10,400)
f_df = stats.f(dfn=dfnum, dfd=dfden)
plt.plot(xx, f_df.pdf(xx))
prop = stats.f.cdf(x = f, dfn = dfnum, dfd = dfden)
t_prop = stats.f.sf(x = f, dfn = dfnum, dfd = dfden)
print("P(F<={0}): {1:.4f}".format(f,prop))
print("P(F>{0}): {1:.4f}".format(f,t_prop))


# In[96]:


np.random.seed(seed=1234)
f_df10 = np.random.f(dfnum=10, dfden=10, size =1000)
sns.distplot(f_df10, fit= stats.norm, kde= False)
sns.distplot(f_df10, fit= stats.norm, kde= True)
xx = np.linspace(0,20,400)
f_df = stats.f(dfn=10, dfd=10)
plt.plot(xx, f_df.pdf(xx))


# In[100]:


np.random.seed(seed=1234)
xx = np.linspace(0,3000,400)
for c in [1,3,5,25,50]:
    weibull_df = stats.weibull_min(c=c, scale=1200)
    plt.plot(xx, weibull_df.pdf(xx))
    


# In[101]:


xx = np.linspace(0,3000,400)
c = 2.2
scale=1200
weibull_df = stats.weibull_min(c=c, scale= scale)
plt.plot(xx, weibull_df.pdf(xx))
x = 1500
v4 = weibull_df.cdf(x)
print("P(X<={0}): {1: 4f}".format(x, 1-v4))


# In[103]:


n = 3
for i in range(n+1):
    prop = stats.binom.pmf(k=i, n=n, p= 0.4)
    print("P(X={0}) = {1:.3f}".format(i, prop))


# In[112]:


for n in [3,5,10,25,50,100]:
    bonorm_df = stats.binom(n=n, p=0.4)
    xx = np.linspace(0,n,n+1)
    plt.plot(xx, bonorm_df.pmf(xx))
    


# In[115]:


for mu in [2,4,6,10]:
    poisson_df = stats.poisson(mu=mu)
    xx = np.linspace(0, 3*mu, 3*mu+1)
    plt.plot(xx, poisson_df.pmf(xx))
    
prop = stats.poisson.pmf(3,mu)
cdf_prop= stats.poisson.cdf(2,mu)
print("1분당 {0}번의 전화가 올 확률: {1:.4f}".format(3,prop))
print("1분당 최대 {0}회 이하의 전화가 올 확률: {1:.4f}".format(2,cdf_prop))


# In[122]:


poi_df1 = stats.poisson(mu=2)
v_p1 = poi_df1.pmf(3)
print(v_p1)


# In[3]:


ds_mycars= pd.read_csv('/실습화일/csv', engine='python')


# In[4]:


ds_mycars = pd.read_csv('mycars.csv', engine='python')


# In[5]:


ds_mycars.head()


# In[6]:


ds_mycars = pd.read_csv('실습화일/mycars.csv', engine='python')


# In[7]:


ds_mycars.head()


# In[12]:


ds_mycars['mpg'].var()


# In[17]:


ds_mycars[['mpg','highway_mileage']].mean()


# In[24]:


ds_mycars.groupby('cylinder').mean()


# In[26]:


ds_mycars[['mpg','highway_mileage','cylinder']].groupby('cylinder').mean()


# In[27]:


ds_mycars.mpg.mean()


# In[32]:


ds_mycars.mpg.describe()zz = temp.groupby('AGE')


# In[35]:


ds_EXH_QC=pd.read_csv('실습화일/EXH_QC.csv', engine='python')


# In[36]:


ds_EXH_QC.head()


# In[37]:


ds_EXH_QC.shape


# In[40]:


df = ds_EXH_QC[['Flaws','Period']]
count = df['Flaws'].value_counts().sort_index()
print(count)
count = np.cumsum(count)
print(count)


# In[41]:


percent = count/sum(count)*100
print(percent)


# In[42]:


cumpct = np.cumsum(percent)
cumpct


# In[43]:


data = pd.DataFrame({'count':count, 'cumcnt':cumcnt, 'percent':percent, 'cumpct':cumpct})


# In[44]:


count2 = df['Period'].value_counts().sort_index()
cumcnt2 = np.cumsum(count2)
percent2 = count2/sum(count2)*100
cumpct2 = np.cumsum(percent2)
count_data2 = pd.DataFrame({'count':count2, 'CumCnt':cumcnt2,'Percent':percent2,'CumPct':cumpct2})
count_data2.columns.name = 'Period'
count_data2


# In[47]:


count_data2.T


# In[7]:


ds_example=pd.read_csv('실습화일/EXH_QC.csv',engine='python')
df = ds_example[['Cabbage weight']].copy()
lower,upper = stats.norm.interval(0.95, loc=2.7, scale= 0.397/np.sqrt(40))
# *lower,upper = stats.norm.interval(0.95, loc=df.mean(), scale =stats.sem(df))
print("신뢰구간: ({0},{1})".format(lower.round(2),upper.round(2)))


# In[18]:


df = pd.DataFrame({'sample':[18,18,20,21,20,23,19,18,17,21,22,20,20,21,20,19,19,18,17,19]})
lower,upper = stats.norm.interval(0.95, loc=2.7, scale= 3.8/np.sqrt(20))
# *lower,upper = stats.norm.interval(0.95, loc=df.mean(), scale =stats.sem(df))
print("신뢰구간: ({0},{1})".format(lower.round(2),upper.round(2)))


# In[8]:


df = pd.DataFrame({'sample':[54.1, 53.3, 56.1, 55.7, 54.0, 54.1, 54.5, 57.1, 55.2, 53.8, 54.1, 54.1, 56.1, 55.0, 55.9, 56.0, 54.9, 54.3, 53.9, 55.0]})
lower,upper = stats.t.interval(0.95, len(df)-1, loc=np.mean(df), scale = stats.sem(df))
print("신뢰구간: ({0},{1})".format(lower.round(2), upper.round(2)))


# In[9]:


#page 63 1-Sample t-test 연습
df = pd.DataFrame({'sample':[74.5, 81.2, 73.8, 82.0, 76.3, 75.7, 80.2, 72.6, 77.9, 82.8]})
t_test = stats.ttest_1samp(df, 76.7)
t,p = t_test.statistic.round(3), t_test.pvalue.round(3)
print("1-sample t-test")
print("t:{}".format(t))
print("p:{}".format(p))


# In[10]:


#page 64 1-Sample t-test 실습
df = pd.DataFrame({'sample':[85.0, 79.0, 79.1, 79.9, 81.6, 78.6, 85.4, 83.4, 78.1, 79.2]})
t_test = stats.ttest_1samp(df, 76.7)
t,p = t_test.statistic.round(3), t_test.pvalue.round(3)
print("1-sample t-test")
print("t:{}".format(t))
print("p:{}".format(p))


# In[11]:


#page 65 2-Sample t-test 연습
df1 = pd.DataFrame({'sample':[6,5,5,4,6,7,6,4,5,6,4,5,5,6,4,8,6,5,6,7]})
df2 = pd.DataFrame({'sample':[7,5,7,8,7,8,8,5,7,6,5,5,6,6,5,7,9,7,7,8]})
t_result = stats.ttest_ind(df1, df2)
t, p = t_result.statistic.round(3), t_result.pvalue.round(3)
print("2-Sample t-test")
print("t검정통계량:{}".format(t))
print("p-value:{}".format(p))


# In[14]:


# page67 t-test연습
df1 = pd.DataFrame({'before':[720,589,780,648,720,589,780,648,780,648]})
df2 = pd.DataFrame({'after':[810,670,790,712,810,670,790,712,790,712]})
t_test = stats.ttest_rel(df1, df2)
t, p = t_result.statistic.round(3), t_test.pvalue.round(3)
print("paired t-test")
print("t:{}".format(t))
print("p:{}".format(p))


# In[15]:


# page68 paried t-test 실습
df1 = pd.DataFrame({'before':[720,589,780,648,720,589,780,648,780,648]})
df2 = pd.DataFrame({'after':[710,580,787,712,750,600,782,670,790,680]})
t_test = stats.ttest_rel(df1, df2)
t, p = t_result.statistic.round(3), t_test.pvalue.round(3)
print("paired t-test")
print("t:{}".format(t))
print("p:{}".format(p))


# In[17]:


#page 69 ANOVA연습
df1 = pd.DataFrame({'A':[892,623,721,678,723,790,720,670,690,771]})
df2 = pd.DataFrame({'B':[721,821,910,678,723,790,711,790,745,891]})
df3 = pd.DataFrame({'C':[621,915,888,721,894,834,841,912,845,889]})
f_test = stats.f_oneway(df1, df2,df3)
f, p = f_test.statistic.round(3), f_test.pvalue.round(3)
print("one way ANOVA")
print("F:{}".format(f))
print("p:{}".format(p))


# In[19]:


#70page 연습
count = 15
nobs = 100
value= 0.1
stat, pval = proportions_ztest(count, nobs, value)
print("1 proportion test")
print("z:{0: 0.3f}".format(stat))
print("z:{0: 0.3f}".format(pval))


# In[20]:


#70page 실습
count = 10
nobs = 100
value= 0.2
stat, pval = proportions_ztest(count, nobs, value)
print("1 proportion test")
print("z:{0: 0.3f}".format(stat))
print("z:{0: 0.3f}".format(pval))


# In[21]:


#72 2Propotion test 연습
count = np.array([4,1])
nobs = np.array([1000, 1200])

stat, pval = proportions_ztest(count, nobs)
print("2 Proportion test")
print('p검정통계량:{0:0.3f}'.format(stat))
print('p-value:{0:0.3f}'.format(pval))


# In[22]:


#73 2Proption test 실습
count = np.array([14,5])
nobs = np.array([1200, 1200])
stat, pval = proportions_ztest(count, nobs)

print("2 Proportion test")
print('p 검정통계량 : {0:0.3f}'.format(stat))
print('p-value:{0:0.3f}'.format(pval))


# In[24]:


#74 카이제곱 검정, chi-square test 연습
df = pd.DataFrame({'HSG':[270,260,236,234],'SS':[228,285,225,262],'SPA':[277,284,231,208]})
chi,pval,dof,expected = stats.chi2_contingency(df.T)
print("chi-square test")
print('chisq:{0:0.3f}'.format(chi))
print('p:{0:0.3f}'.format(pval))
print('degree pf freedom:{}'.format(dof))
print('expected value: \n{}'.format(expected.round(3)))


# In[25]:


#75 카이제곱 검정, chi-square test 실습
df = pd.DataFrame({'A/S':[18,8,4,4,3,3],'가격':[1,2,1,1,1,25],'성능':[8,14,3,2,3,8],'확장성':[7,5,4,3,1,10],'디자인':[10,5,9,2,1,2],'안정성':[9,9,5,7,1,1],'기능성':[10,4,4,3,1,7]})
chi,pval,dof,expected = stats.chi2_contingency(df.T)
print("chi-square test")
print('chisq:{0:0.3f}'.format(chi))
print('p:{0:0.3f}'.format(pval))
print('degree pf freedom:{}'.format(dof))
print('expected value: \n{}'.format(expected.round(3)))


# In[ ]:




