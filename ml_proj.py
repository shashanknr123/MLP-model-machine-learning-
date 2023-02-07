#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import activations




df=pd.read_csv("C:\\Users\\shash\\Downloads\\archive (3)\\ADANIPORTS.csv")
print(df)


# In[63]:


y = df['Volume']
y
df.shape


# In[48]:


df.isnull().sum()


# In[49]:


df.isnull().sum()


# In[50]:


y = np.ravel(y)
print(y)


# In[51]:


print(df.columns)


# In[52]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[[ 'Symbol', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP',
       'Volume', 'Deliverable Volume', '%Deliverble']] = df[['Symbol', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP',
       'Volume', 'Deliverable Volume', '%Deliverble']].apply(le.fit_transform)
df.head()


# In[53]:


X= df.drop(['Close'],axis=1)
X


# In[54]:


scaler=StandardScaler()
X = scaler.fit_transform(X)
print(X)


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[79]:


mlp = MLPRegressor(max_iter=200)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)


# In[80]:


plt.scatter(y_test,y_pred)
plt.plot()


# In[81]:


plt.scatter(y_test, y_pred)
plt.show()


# In[82]:


from sklearn.neural_network import MLPClassifier
mlp1 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=0)
mlp1.fit(X, y)
probabilities = mlp1.predict_proba(X)[:, 1]
plt.scatter(range(len(y)), probabilities)
plt.xlabel("Sample Index")
plt.ylabel("Probability of Positive Class")
plt.show()


# In[83]:


# Plot the difference between the actual and predicted values
difference = y - probabilities
plt.scatter(range(len(y)), difference)
plt.xlabel("Sample Index")
plt.ylabel("Difference")
plt.show()


# In[84]:


from sklearn.metrics import r2_score

y_pred = mlp.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)


# In[ ]:




