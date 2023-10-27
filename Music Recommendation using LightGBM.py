#!/usr/bin/env python
# coding: utf-8

# # Music Recommendation using LightGBM

# ## Process Step

# 1.Import Dataset
# 2.Visual Dataset
# 3.Data Cleaning
# 4.Training the Module
# 5.Prediction

# In[1]:


#Importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Reading the dataset

mem=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\members.csv")
song=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\songs.csv")
train=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\train.csv")


# In[3]:


train.head()


# In[4]:


mem.head()


# In[5]:


song.head()


# In[6]:


train.info(),mem.info(),song.info()


# In[7]:


train.describe()


# In[8]:


mem.describe()


# In[9]:


song.describe()


# In[10]:


train.shape


# In[11]:


mem.shape


# In[12]:


song.shape


# #  Visual Dataset

# In[13]:


plt.figure(figsize=(20,10))
sns.countplot(x='source_system_tab',hue='source_system_tab',data=train)


# In[14]:


plt.figure(figsize=(20,10))
sns.countplot(x='source_system_tab',hue="target",data=train)


# In[15]:


plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.countplot(x='source_system_tab',hue="target",data=train)


# In[16]:


plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.countplot(x='source_type',hue="source_type",data=train)


# In[17]:


plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.countplot(x='source_type',hue="target",data=train)


# In[18]:


plt.figure(figsize=(20,10))
sns.countplot(x='language',hue="language",data=song)


# In[19]:


plt.figure(figsize=(20,10))
sns.countplot(x='registered_via',hue="registered_via",data=mem)


# # Data Cleaning

# In[20]:


r = 7000
s = 3000
name=['msno','song_id','source_system_tab','source_screen_name','source_type','target']
test1 = pd.read_csv(r'D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\train.csv',names=name,skiprows=r,nrows=s)


# In[21]:


test=test1.drop(['target'],axis=1)
yts = np.array(test1['target'])


# In[22]:


t_name=['id','msno','song_id','source_system_tab','source_screen_name','source_type']
test['id']=np.arange(s)
test=test[t_name]


# In[23]:


sons_col=['song_id','artist_name','genre_ids','song_length','language']
train=train.merge(song[sons_col],on="song_id",how='left')
test=test.merge(song[sons_col],on="song_id",how='left')


# In[24]:


mem['registration_year']=mem['registration_init_time'].apply(lambda x:int(str(x)[0:4]))
mem['registration_month']=mem['registration_init_time'].apply(lambda x:int(str(x)[4:6]))
mem['registration_date']=mem['registration_init_time'].apply(lambda x:int(str(x)[6:8]))


# In[25]:


mem['expiration_year']=mem['expiration_date'].apply(lambda x:int(str(x)[0:4]))
mem['expiration_month']=mem['expiration_date'].apply(lambda x:int(str(x)[4:6]))
mem['expiration_date']=mem['expiration_date'].apply(lambda x:int(str(x)[6:8]))
mem=mem.drop(['registration_init_time'],axis=1)


# In[26]:


mem_col=mem.columns
train=train.merge(mem[mem_col],on='msno',how='left')
test=test.merge(mem[mem_col],on='msno',how='left')


# In[27]:


train=train.fillna(-1)
test=test.fillna(-1)


# In[28]:


import gc
del mem,song;gc.collect();


# In[29]:


cols=list(train.columns)
cols.remove('target')


# In[30]:


from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
for col in tqdm(cols):
    if train[col].dtype=='object':
        train[col]=train[col].apply(str)
        test[col]=test[col].apply(str)
        
        le=LabelEncoder()
        train_val=list(train[col].unique())
        test_val=list(test[col].unique())
        le.fit(train_val+test_val)
        train[col]=le.transform(train[col])
        test[col]=le.transform(test[col])


# In[31]:


unique_song=range(max(train['song_id'].max(),test['song_id'].max()))
song_pop=pd.DataFrame({'song_id':unique_song,'popularity':0})
train_sort=train.sort_values('song_id')
test_sort=test.sort_values('song_id')
train_sort.reset_index(drop=True,inplace=True)
test_sort.reset_index(drop=True,inplace=True)


# # Training the Module

# In[32]:


pip install lightgbm


# In[33]:


from sklearn.model_selection import train_test_split
import lightgbm as lgb
x=np.array(train.drop(['target'],axis=1))
y=train['target'].values

x_test=np.array(test.drop(['id'],axis=1))
ids=test['id'].values

del train,test;gc.collect();

x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.1,random_state=12)

del x,y;gc.collect();

d_train=lgb.Dataset(x_train,label=y_train)
d_valid=lgb.Dataset(x_valid,label=y_valid)

watchlist=[d_train,d_valid]


# In[34]:


def predict(m1_model):
    model=m1_model.fit(x_train,y_train)
    print('Training Score : {}'.format(model.score(x_train,y_train)))
    y_pred=model.predict(x_valid)
    v_test=model.predict(x_test)
    yhat=(v_test>0.5).astype(int)
    comp=(yhat==yts).astype(int)
    acc=comp.sum()/comp.size*100
    print("Accuracy on test data for the model",acc)


# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


predict(LogisticRegression())


# # Prediction using LightGBM

# In[38]:


par={}
par['learning_rate']=0.4
par['application']='binary'
par['max_depth']=15
par['num_leaves']=2**8
par['verbosity']=0
par['metric']='aus'

model1=lgb.train(par,train_set=d_train,num_boost_round=200,valid_sets=watchlist)


# In[39]:


p_test=model1.predict(x_test)


# In[41]:


yhat=(p_test>0.5).astype(int)
comp=(yhat==yts).astype(int)
acc=comp.sum()/comp.size*100
print('The accuracy of lgbm model on test data is: {0:f}%'.format(acc))

