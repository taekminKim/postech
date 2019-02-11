#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from sklearn import linear_model


# In[3]:


# page80 가정검정실습
df1 = [37.6, 38.6, 37.2, 36.4, 38.6, 39, 37.2, 36.1, 35.9, 37.1, 36.9, 37.5, 36.3, 38.1, 39, 36.9,  36.8, 37.6, 33, 33.5 ]
df2 = [14, 15, 14, 16, 17, 14, 17, 16, 15,16,14,16,18,13,15,17,14,16,20,21]
plt.scatter(df1, df2)
corr, pval = stats.pearsonr(df1, df2)
print("correlation analysis")
print("corr : {0:0.3f}".format(corr))
print("p : {0:0.3f}".format(pval))


# In[5]:


# page82 단순선형회귀 연습
# 데이터 선언
MINUTES = [1,2,3,4,4,5,6,6,7,8]
UNITS = [23,29,49,64,74,87,96,97,109,119]

#Simple Linear Regression 실행
plt.scatter(MINUTES, UNITS)
MINUTES = sm.add_constant(MINUTES)
model = sm.OLS(UNITS, MINUTES)
result = model.fit()

# Simple Linear Regression 결과 출력
result.summary()
print(result.summary())


# In[6]:


import pandas as pd
import numpy as np
from numpy.random import randn


# In[11]:


np.random.seed(1234)
df = pd.DataFrame(randn(10,2), index = ['a','b','c','d','e','f','g','h','i','j'] , columns = ['A','B'])
df


# In[13]:


df.A


# In[15]:


df[['A','B']]


# In[30]:


df['A'][0]


# In[37]:


df.loc['a':'c',['A']]


# In[39]:


df.loc[:,['A']]


# In[40]:


df.iloc[0:5,[0]]


# In[42]:


df.loc[['d'],['A']]


# In[43]:


df.iloc[[3],[0]]


# In[44]:


df[df['A']>0]


# In[52]:


df[(df.A>0) & (df.B>0)]


# In[53]:


df[(df.A>df.B)]


# In[54]:


df.loc[['a','b','c'],['A','B']]


# In[55]:


df['A+B']=df.A+df.B
df[['A','B','A+B']]


# In[56]:


df['A*2'] = df.A*2
df[:]


# In[57]:


df['A/B'] = df.A/df.B
df


# In[60]:


df.drop('A/B', axis=1)


# In[62]:


df.drop(['A+B','A*2'], axis=1)


# In[64]:


df.sort_values(by='A')


# In[65]:


df.sort_values(by='A', ascending=False)


# In[66]:


np.random.seed(1234)
df1 = pd.DataFrame(randn(10,4), columns=['A','B','C','D'])
df1


# In[69]:


df1.sort_values(by='A', ascending=False)


# In[70]:


df1 = pd.DataFrame(randn(10,4), index = ['a','c','b','d','e','f','g','h','i','j'], columns=['A','B','C','D'])
df1


# In[71]:


df1.columns


# In[72]:


df1.index


# In[74]:


df1.shape


# In[75]:


df1.reindex(columns=['B', 'A', 'C', 'D'])


# In[76]:


df1.rename(columns={'A':'A+'})


# In[95]:


df1 =pd.DataFrame({'A':['A0','A1','A2','A3'],
                  'B':['B0','B1','B2','B3'],
                  'C':['C0','C1','C2','C3'],
                  'D':['D0','D1','D2','D3'],
                 }, index = [0,1,2,3])
df2 =pd.DataFrame({'A':['A4','A5','A6','A7'],
                  'B':['B4','B5','B6','B7'],
                  'C':['C4','C5','C6','C7'],
                  'D':['D4','D5','D6','D7'],
                 }, index = [4,5,6,7])
df3 =pd.DataFrame({'A':['A8','A9','A10','A11'],
                  'B':['B8','B9','B10','B11'],
                  'C':['C8','C9','C10','C11'],
                  'D':['D8','D9','D10','D11'],
                 }, index = [8,9,10,11])


# In[92]:


pd.concat([df1, df2, df3], axis=1)


# In[93]:


pd.concat([df1,df2,df3], axis=1, ignore_index=True)


# In[96]:


df4 =pd.DataFrame({'A':['A12','A13','A14','A15'],
                  'B':['B12','B13','B14','B15'],
                  'C':['C12','C13','C14','C15'],
                  'D':['D12','D13','D14','D15'],
                 }, index = [12,13,14,15])


# In[97]:


pd.concat([df1,df2,df3,df4])


# In[101]:


pd.concat([df1,df2,df3,df4], join='inner')


# In[105]:


#page 106 concat 실습
df1 =pd.DataFrame({'A':['A0','A1','A2','A3'],
                  'B':['B0','B1','B2','B3'],
                  'C':['C0','C1','C2','C3'],
                  'D':['D0','D1','D2','D3'],
                 }, index = [0,1,2,3])
df2 =pd.DataFrame({'E':['E4','E5','E6','E7'],
                  'F':['F4','F5','F6','F7'],
                  'G':['G4','G5','G6','G7'],
                  'H':['H4','H5','H6','H7'],
                 }, index = [0,1,2,3])
df3 =pd.DataFrame({'I':['I8','I9','I10','I11'],
                  'J':['J8','J9','J10','J11'],
                  'K':['K8','K9','K10','K11'],
                  'L':['L8','L9','L10','L11'],
                 }, index = [0,1,2,3])


# In[106]:


pd.concat([df1, df2, df3], axis=1)


# In[111]:


df1 = pd.DataFrame({'data1':[0,1,2,3], 'key':['b','b','a','c']})
df2 = pd.DataFrame({'data2':[0,1,2], 'key':['a','a','b']})
print(df1)
print(df2)


# In[112]:


pd.merge(df1, df2)


# In[113]:


pd.merge(df1, df2, on='key', how='outer')


# In[114]:


pd.merge(df1, df2, on='key', how='left')


# In[116]:


df1.rename(columns={'key':'lkey'})


# In[117]:


pd.merge(df1, df2, left_on='key', right_on='key', how='inner')


# In[133]:


left =pd.DataFrame({'key1':['K0','K1','K2','K3'],
                  'key2':['K0','K1','K0','K1'],
                  'A':['A0','A1','A2','A3'],
                  'B':['B0','B1','B2','B3']})
right =pd.DataFrame({'key1':['K0','K1','K2','K3'],
                  'key2':['K0','K0','K0','K0'],
                  'C':['C0','C1','C2','C3'],
                  'D':['D0','D1','D2','D3']})


# In[134]:


pd.merge(left, right, on=['key1', 'key2'])


# In[135]:


pd.merge(left, right, on=['key1','key2'], how='inner')


# In[136]:


pd.merge(left, right, on=['key1','key2'], how='outer')


# In[145]:


left = pd.read_csv('실습화일/FITNESS_1.csv', engine='python', encoding='euc-kr')
right = pd.read_csv('실습화일/FITNESS_2.csv', engine='python', encoding='euc-kr')


# In[147]:


right


# In[156]:


temp = pd.merge(left, right, on=['NAME','GENDER'], how='inner')
temp


# In[160]:


temp.groupby(['GENDER']).mean()


# In[161]:


temp = pd.merge(left, right, on=['NAME','GENDER','AGE'], how='inner')


# In[162]:


temp.groupby(['GENDER']).mean()


# In[165]:


temp.groupby(['GENDER','AGEGROUP']).mean()


# In[170]:


temp.groupby(['GENDER','AGEGROUP']).mean()['WEIGHT']


# In[174]:


temp.groupby(['GENDER','AGEGROUP'])['WEIGHT'].agg(['mean','std','median','min','max'])


# In[176]:


temp.groupby(['GENDER','AGEGROUP'])['WEIGHT','OXY'].describe()


# In[180]:


temp.groupby(['GENDER','AGEGROUP'])['WEIGHT','OXY','RUNTIME'].agg({'WEIGHT':'mean','OXY':'std','RUNTIME':'max'})


# In[184]:


df = pd.DataFrame({'foo': ['one','one','one','two','two','two'],
                  'bar' : ['A','A','C','A','B','C'],
                  'baz' : [1,2,3,4,5,6],
                  'zoo' : ['x','y','z','q','w','t']})
df


# In[183]:


df.pivot(index='foo', columns='bar', values='baz')


# In[186]:


df.pivot_table(index='foo', columns='bar', values='baz', aggfunc=sum)


# In[187]:


df.melt(id_vars=['foo','bar'])


# In[188]:


df.melt(id_vars=['foo','bar'],value_vars=['zoo'])


# In[ ]:




