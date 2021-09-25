#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[70]:


df = pd.read_csv('weight-height.csv')


# In[71]:


df.head()


# In[72]:


from sklearn.preprocessing import LabelEncoder


# In[73]:


label = LabelEncoder()


# In[74]:


df.Gender = label.fit_transform(df['Gender'])


# In[75]:


df.head()


# In[76]:


x = df.drop('Weight',axis = 1)
y = df['Weight']


# In[77]:


from sklearn.model_selection import train_test_split


# In[78]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=.70)


# # KNN Regression

# In[79]:


from sklearn.neighbors import KNeighborsRegressor


# In[86]:


reg = KNeighborsRegressor(n_neighbors= 119)


# In[87]:


reg.fit(xtrain,ytrain)


# In[88]:


reg.score(xtest,ytest)


# In[89]:


pred = reg.predict(xtest)


# In[90]:


pred


# # K value calculate

# In[85]:


error_rate = []
for i in range(15,200,10):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(xtrain,ytrain)
    y_predict = knn.predict(xtest)
    error_rate.append(np.mean(y_predict-ytest))

error_rate
# In[18]:


plt.xlabel('K value')
plt.ylabel('error rate')
plt.plot(range(15,200,10),error_rate)


# # Linear Model

# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


lreg = LinearRegression()


# In[21]:


lreg.fit(xtrain,ytrain)


# In[22]:


lreg.score(xtest,ytest)


# In[23]:


pred_value = lreg.predict(xtest)
pred_value


# # Mean Squared Error

# In[39]:


from sklearn.metrics import mean_squared_error


# In[42]:


mse = mean_squared_error(ytest,pred)


# In[43]:


mse


# In[44]:


import math
rmse = math.sqrt(mse)
rmse


# # KNN Classifier

# In[55]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[56]:


data = pd.read_csv('weight-height.csv')


# In[57]:


from sklearn.preprocessing import LabelEncoder
label2 = LabelEncoder()
data.Gender = label.fit_transform(data['Gender'])


# In[58]:


data.head()


# In[59]:


X = data.drop('Gender',axis = 1)
Y = data['Gender']


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=.70)


# In[62]:


from sklearn.neighbors import KNeighborsClassifier


# In[63]:


clf = KNeighborsClassifier(n_neighbors=175)


# In[64]:


clf.fit(X_train,Y_train)


# In[65]:


clf.score(X_test,Y_test)


# In[66]:


error = []
for i in range(15,200,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    Y_predict = knn.predict(X_test)
    error.append(np.mean(Y_predict != Y_test))


# In[67]:


error


# In[68]:


plt.xlabel('K value')
plt.ylabel('error')
plt.plot(range(15,200,10),error)


# In[ ]:





# In[ ]:




