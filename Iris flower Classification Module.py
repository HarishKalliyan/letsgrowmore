#!/usr/bin/env python
# coding: utf-8

# # Iris flower Classification Module

# # Process Step

# 1. Import Dataset
# 2. Visual Dataset
# 3. Data Prepation
# 4. Training the Module
# 5. Prediction
# 6. Model Evaluation

# # Import Dataset

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[2]:


#Importing DataSet from local device
a=pd.read_csv('D:\Intern Work\LetsGrowMore\Iris classification\DataSet\iris.csv')
a


# In[3]:


#Print first 25 dataset
a.head(25)


# In[4]:


#Print detail info of Dataset
a.info()


# In[5]:


a.dtypes


# In[6]:


a.shape


# In[7]:


#Print the Null values in Dataset
a.isnull().sum()


# In[8]:


#Grouping the records by Species
b=a.groupby('Species')
b.head()


# In[9]:


#Finding the Differnt Species Type
a['Species'].unique()


# # Visual Dataset

# In[10]:


#PairPlot using Seaborn
sns.pairplot(a,hue="Species",markers="x")


# In[11]:


#ScatterPlot Diagram
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=a,hue="Species")

plt.subplot(1,2,2)
sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=a,hue="Species")


# In[12]:


#PieChart Diagram
a['Species'].value_counts().plot(kind="pie",autopct="%1.1f%%",shadow=True,figsize=(5,5))
plt.title("Percentage values in Each Species",fontsize=12,c="g")
plt.ylabel("",fontsize=15,c="r")


# In[13]:


#JointPlot Diagram for SepalLengthCm vs SepalWidthCm
sns.jointplot(data=a,x="SepalLengthCm",y="SepalWidthCm",size=7,hue="Species")


# In[14]:


#JointPlot Diagram for PetalLengthCm vs PetalWidthCm
sns.jointplot(data=a,x="PetalLengthCm",y="PetalWidthCm",size=7,hue="Species")


# In[15]:


#Barplot for Species Vs Each DataColumns
plt.figure(figsize=(25,15))
plt.subplot(2,2,1)
sns.barplot(data=a,x="Species",y="SepalLengthCm",palette=("Spectral"))
plt.title("Bar plot Species Vs SepalLengthCm")

plt.subplot(2,2,2)
sns.barplot(data=a,x="Species",y="SepalWidthCm",palette=("Spectral"))
plt.title("Bar plot Species Vs SepalWidthCm")

plt.subplot(2,2,3)
sns.barplot(data=a,x="Species",y="PetalLengthCm",palette=("Spectral"))
plt.title("Bar plot Species Vs PetalLengthCm")

plt.subplot(2,2,4)
sns.barplot(data=a,x="Species",y="PetalWidthCm",palette=("Spectral"))
plt.title("Bar plot Species Vs PetalWidthCm")


# In[16]:


#Boxplot for Species Vs Each DataColumns
plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
sns.boxplot(data=a,x="Species",y="SepalLengthCm",palette=("Spectral"))
plt.title("Box plot Species Vs SepalLengthCm")

plt.subplot(2,2,2)
sns.boxplot(data=a,x="Species",y="SepalWidthCm",palette=("Spectral"))
plt.title("Box plot Species Vs SepalWidthCm")

plt.subplot(2,2,3)
sns.boxplot(data=a,x="Species",y="PetalLengthCm",palette=("Spectral"))
plt.title("Box plot Species Vs PetalLengthCm")

plt.subplot(2,2,4)
sns.boxplot(data=a,x="Species",y="PetalWidthCm",palette=("Spectral"))
plt.title("Box plot Species Vs PetalWidthCm")


# In[17]:


#Distplot for Species Vs Each DataColumns
plt.figure(figsize=(25,15))
plt.subplot(2,2,1)
sns.distplot(a["SepalLengthCm"],color="b").set_title("Sepal Length interval")

plt.subplot(2,2,2)
sns.distplot(a["SepalWidthCm"],color="g").set_title("Sepal Width interval")

plt.subplot(2,2,3)
sns.distplot(a["PetalWidthCm"],color="y").set_title("Petal Width interval")

plt.subplot(2,2,4)
sns.distplot(a["PetalLengthCm"],color="r").set_title("Petal Length interval")


# In[18]:


#HeatMap for the Dataset
sns.heatmap(a.corr())


# # Data Prepation

# In[19]:


#Removing Id from Dataset
a.drop('Id',axis=1,inplace=True)
#Renaming the Species and grouping it
c={'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
a.Species=[c[i] for i in a.Species]
a


# In[20]:


m=a.iloc[:,0:4]
m


# In[21]:


n=a.iloc[:,4]
n


# # Training the Module

# In[22]:


x_train,x_test,y_train,y_test = train_test_split(m,n,test_size=0.33,random_state=42)


# In[23]:


model=LinearRegression()


# In[24]:


model.fit(m,n)


# In[25]:


#Coef of prediction
model.score(m,n)


# In[26]:


model.coef_


# In[27]:


model.intercept_


# # Prediction

# In[28]:


pred= model.predict(x_test)
pred


# # Model Evaluation

# In[29]:


print("Mean Squared Error: %.2f" % np.mean((pred - y_test)**2))

