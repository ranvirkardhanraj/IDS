#!/usr/bin/env python
# coding: utf-8

# # IDS CODE

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# In[2]:


os.chdir("C:\\Users\\Unique pc\\Downloads\\IDS")


# In[5]:


data = pd.read_csv("Train_data.csv")
data


# In[4]:


obj_encode = LabelEncoder()

protocol_type_d = data["protocol_type"].values
protocol_type_en = obj_encode.fit_transform(protocol_type_d)
data["protocol_type"] = data["protocol_type"].replace(protocol_type_d,protocol_type_en)

service_d = data["service"].values
service_en = obj_encode.fit_transform(service_d)
data["service"] = data["service"].replace(service_d,service_en)

flag_d = data["flag"].values
flag_en = obj_encode.fit_transform(flag_d)
data["flag"] = data["flag"].replace(flag_d,flag_en)

x = data.iloc[:,0:40].values


# In[12]:


class_d = data["class"].values
class_en = obj_encode.fit_transform(class_d)
data["class"] = data["class"].replace(class_d,class_en)
y = data.iloc[:,41].values
y


# In[7]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)


# In[8]:


classifire = RandomForestClassifier(n_estimators=100,criterion="gini")


# In[9]:


classifire.fit(x_train,y_train)


# In[10]:


print(classifire.score(x_test,y_test))


# In[11]:


y_pred=classifire.predict(x_test)


# In[13]:


confusion_matrix(y_test,y_pred)


# In[14]:


test_data = pd.read_csv("Test_data.csv")


# In[15]:


test_data


# In[20]:


protocol_typedd=test_data["protocol_type"].values
protocol_type_en2 = obj_encode.fit_transform(protocol_typedd)
test_data["protocol_type"] = test_data["protocol_type"].replace(protocol_typedd,protocol_type_en2)
test_data["protocol_type"]


# In[21]:


service_dd = test_data["service"].values
service_en2 = obj_encode.fit_transform(service_dd)
test_data["service"] = test_data["service"].replace(service_dd,service_en2)
test_data["service"]


# In[22]:


flag_dd=test_data["flag"].values
flag_en2 = obj_encode.fit_transform(flag_dd)
test_data["flag"]= test_data["flag"].replace(flag_dd,flag_en2)
test_data["flag"]


# In[26]:


test_data


# In[29]:


x = test_data.iloc[:,0:40].values


# In[30]:


predection = classifire.predict(x)


# In[31]:


predection


# In[ ]:




