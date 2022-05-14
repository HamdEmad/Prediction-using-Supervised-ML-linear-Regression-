#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
df.head()


# In[21]:


X = df[['Hours']]
y = df['Scores']


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)                             


# In[34]:


print("number of test samples :", X_test.shape[0])
print("number of training samples:",X_train.shape[0])


# In[53]:


plt.figure(figsize=(10,8))
sns.regplot(data=df, x= X, y= y)
plt.show();


# X and y have a strong positive linear correlation.

# In[49]:


lr = LinearRegression()


# In[50]:


lr.fit(X_train,y_train)


# In[54]:


lr.intercept_


# In[55]:


lr.coef_


# In[58]:


Y_hat = lr.predict(X_test)


# In[64]:


lr.score(X_train,y_train)


# In[71]:


plt.figure(figsize=(10, 8))
ax1 = sns.distplot(d, hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Scores')
plt.xlabel('Scores')
plt.ylabel('Houres')

plt.show()
plt.close();


# **by using X and y train
# We can see that the fitted values are reasonably close to the actual values since the two distributions overlap a bit. However, there is definitely some room for improvement, so i will use all data to fit the model**

# In[76]:


all_data = LinearRegression()


# In[79]:


all_data.fit(X,y)
y_predict_all_data= all_data.predict(X)


# In[80]:


plt.figure(figsize=(10, 8))
ax1 = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(y_predict_all_data, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Scores')
plt.xlabel('Scores')
plt.ylabel('Houres')

plt.show()
plt.close();


# **By using all data to fit the model, We can see that the fitted values are more reasonably close to the actual values**

# In[81]:


all_data.score(X,y)


# In[90]:


df['target_hour']=9.25


# In[97]:


predicted_Score= all_data.predict(df[['target_hour']])[0]


# In[104]:


print("No of Hours = 9.25")
print("Predicted Score = ",predicted_Score)

