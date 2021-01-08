#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


heart = pd.read_csv('heart.csv')


# In[4]:


heart


# In[6]:


heart.isna().sum()


# In[7]:


heart.dtypes


# In[8]:


heart.describe()


# In[9]:


plt.figure(figsize=(20,10))
sns.heatmap(heart.corr(), annot=True, cmap='terrain')


# In[15]:


heart.hist(figsize=(12,12),layout=(5,3))


# In[22]:


heart.plot(kind='box', subplots=True, layout=(5,3), figsize=(12,12))
plt.show()


# In[23]:


sns.catplot(data=heart, x='sex', y='age',  hue='target')


# In[24]:


heart['sex'].value_counts()


# In[25]:


heart['target'].value_counts()


# In[26]:


sns.countplot(x='sex', data=heart,hue='target')


# In[27]:


temp=pd.crosstab(index=heart['sex'],
            columns=[heart['thal']], 
            margins=True)
temp


# In[29]:


temp=pd.crosstab(index=heart['target'],
            columns=[heart['cp']], 
            margins=True)
temp


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()  
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
heart[columns_to_scale] = StandardScaler.fit_transform(heart[columns_to_scale])


# In[32]:


heart.head()


# In[33]:


X= heart.drop(['target'], axis=1)
y= df['target']


# In[38]:


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=40)


# In[39]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)


# In[40]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,prediction1)
cm


# In[41]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction1)


# In[42]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction1))


# In[43]:


sns.heatmap(cm, annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




