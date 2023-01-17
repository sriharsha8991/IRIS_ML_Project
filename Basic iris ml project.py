#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df = pd.read_csv("iris.csv",index_col=0)


# In[8]:


df.head()


# In[9]:


df.info()


# In[13]:


#Ordinal encoding of Species
df['Species'] = df['Species'].factorize()[0]


# In[12]:


df['Species'].value_counts()


# In[14]:


df.corr()


# ### Petal lenghths and widths are highly correlated with Species

# In[15]:


plt.figure(figsize = (14,9))
sns.heatmap(df.corr(),annot = True)


# In[ ]:





# In[16]:


get_ipython().system('pip install sweetviz')


# In[18]:


import sweetviz as sv


# In[19]:


report  = sv.analyze(df)


# In[21]:


report.show_html()


# In[ ]:





# In[22]:


report.show_notebook()


# In[24]:


get_ipython().system('pip install lazypredict')


# In[25]:


from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
x = df.drop('Species',axis = 1)
y = df.Species


# In[26]:


trainx,testx,trainy,testy = train_test_split(x,y)


# In[27]:


classify = LazyClassifier()


# In[30]:


classify.fit(trainx,testx,trainy,testy)


# ### Support Vectors are producing high Accuracy on the given Data Sets
# 
