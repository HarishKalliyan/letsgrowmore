#!/usr/bin/env python
# coding: utf-8

# # Stock Market Prediction and Forecasting using Stacked LSTM

# # Process Step

# 1. Import Dataset
# 2. Data Preparation
# 3. Visualize Dataset 
# 4. Training the Module
# 5. Prediction
# 6. Model Evaluation

# # Import Dataset

# In[1]:


#Importing Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[2]:


#Read DataSet

a=pd.read_csv('D:\Intern Work\LetsGrowMore\Stock  Market\TATAGLOBAL.csv')
a


# In[3]:


a.head(25)


# In[4]:


a.tail(25)


# In[5]:


#Display basic Statistics 
a.describe()


# # Data Preparation

# In[6]:


#Null values in data
Nan = [(c,a[c].isnull().mean()*100) for c in a]
Nan = pd.DataFrame(Nan,columns=['ColName','Percentage'])
Nan


# In[7]:


#Sort by Date
s=a.sort_values(by='Date')
s.head()


# In[8]:


s.reset_index(inplace=True)
s


# # Visualize Dataset

# In[9]:


plt.figure(figsize=(10,7))
plt.plot(s['Date'],s['Close'])


# # Training the Module

# In[ ]:





# In[10]:


cls=s['Close']
cls


# In[11]:


#Implementing MinMax Scaler
sca=MinMaxScaler(feature_range=(0,1))
cls=sca.fit_transform(np.array(cls).reshape(-1,1))
cls


# # Creating a Training and Testing Dataset

# In[12]:


#Splitting of Data to train the Module
train_size=int(len(cls)*0.7)
test_size=len(cls)-train_size
train_data,test_data=cls[0:train_size,:],cls[train_size:len(cls),:1]

#Print the Size of each Data
train_data.shape,test_data.shape


# In[13]:


#Convertion of Array into Data Matrix
def create(data,time=1):
    dax,day=[],[]
    for i in range(len(data)-time-1):
        a=data[i:(i+time),0]
        dax.append(a)
        day.append(data[i+time,0])
    return np.array(dax),np.array(day)


# # Reshaping the Dataset

# In[14]:


#Reshaping of Dataset
time =100
x_train,y_train= create(train_data,time)
x_test,y_test=create(test_data,time)

x_train.shape, y_train.shape


# In[15]:


x_test.shape, y_test.shape


# In[16]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#print the reshaped dataset
x_train,x_test


# # Creating a LSTM Model 

# In[17]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[18]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()


# # Prediction

# In[19]:


model.fit(x_train,y_train,validation_split=0.1,epochs=60,batch_size=64,verbose=1)


# In[20]:


train_predic=model.predict(x_train)
test_predic=model.predict(x_test)


# In[21]:


train_predic=sca.inverse_transform(train_predic)
test_predic=sca.inverse_transform(test_predic)


# # Model Evaluation

# In[22]:


import math
from sklearn.metrics import mean_squared_error


# In[23]:


math.sqrt(mean_squared_error(y_train,train_predic))


# In[24]:


math.sqrt(mean_squared_error(y_test,test_predic))


# In[25]:


look=100

trainPredicPlot = np.empty_like(cls)
trainPredicPlot[:,:]=np.nan
trainPredicPlot[look:len(train_predic)+look,:]=train_predic

testPredicPlot = np.empty_like(cls)
testPredicPlot[:,:]=np.nan
testPredicPlot[len(train_predic)+(look*2)+1:len(cls)-1,:]=test_predic

plt.figure(figsize=(10,7))
plt.plot(sca.inverse_transform(cls))
plt.plot(trainPredicPlot)
plt.plot(testPredicPlot)


# # Prediction for next Month

# In[26]:


len(test_data)


# In[27]:


pred=test_data[511:].reshape(1,-1)
pred.shape


# In[28]:


temp=list(pred)
temp=temp[0].tolist()
temp


# In[29]:


lst=[]
n=100
i=0
while(i<30):
    if(len(temp)>100):
        #print Temp
        pred=np.array(temp[1:])
        print("{} day input {}".format(i,pred))
        pred=pred.reshape(1,-1)
        pred=pred.reshape((1,n,1))
        
        yhat=model.predict(pred,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp.extend(yhat[0].tolist())
        temp=temp[1:]
        
        lst.extend(yhat.tolist())
    
    else:
        pred=pred.reshape((1,n,1))
        yhat=model.predict(pred,verbose=0)
        print(yhat[0])
        temp.extend(yhat[0].tolist())
        print(len(temp))
        lst.extend(yhat.tolist())
    i+=1

print(lst)     


# # Analyzing last 130 days for Closing price

# In[30]:


newday=np.arange(1,101)
predday=np.arange(101,131)
len(cls)


# In[31]:


plt.figure(figsize=(10,7))
plt.plot(newday,sca.inverse_transform(cls[1935:]))
plt.plot(predday,sca.inverse_transform(lst))


# In[32]:


df=cls.tolist()
df.extend(lst)
print(len(df))


# In[33]:


plt.figure(figsize=(10,7))
plt.plot(df[1935:])


# In[34]:


df=sca.inverse_transform(df).tolist()
plt.figure(figsize=(10,7))
plt.plot(df)

