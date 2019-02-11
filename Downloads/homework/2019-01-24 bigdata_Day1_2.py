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


# In[1]:


import sklearn
print("sklearn: {}".format(sklearn.__version__))


# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from sklearn import linear_model


# In[2]:


import sklearn
print("sklearn: {}".format(sklearn.__version__))


# In[9]:


ds_example=pd.read_csv('EXH_QC.scv',engine='python')
df = ds_example[['Cabbage weight']].copy()
lower,upper = scipy.stats.norm.interval(0.95, loc=2.7, scale= 0.397/np.sqrt(40))
*lower,upper = scipy.stats.norm.interval(0.95, loc=df.mean(), scale =stats.sem(df))
print("신뢰구간: ({0},{1})".format(lower.round(2),upper.round(2)))


# In[ ]:




