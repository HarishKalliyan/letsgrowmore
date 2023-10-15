#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


#Reading the dataset

mem=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\members.csv")
subm=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\sample_submission.csv",nrows=20000)
song_info=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\song_extra_info.csv",nrows=20000)
song=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\songs.csv",nrows=20000)
train=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\train.csv",nrows=20000)
test=pd.read_csv(r"D:\Intern Work\LetsGrowMore\Music Recommendation\kkbox-music-recommendation-challenge\test.csv")


# In[3]:


mem.head(25)


# In[4]:


subm.head(25) 


# In[5]:


song_info.head(30)


# In[6]:


song.head(15)


# In[7]:


song['genre_ids'].fillna(' ',inplace=True)
song['composer'].fillna(' ',inplace=True)
song['lyricist'].fillna(' ',inplace=True)


# In[8]:


#Analysing the Null Values in the Datasets

subm.isnull().sum(),mem.isnull().sum(),song_info.isnull().sum(),song.isnull().sum()


# In[9]:


train.head()


# In[10]:


test.tail()


# In[11]:


train.isnull().sum() , test.isnull().sum()


# In[12]:


#Remove the Unnecessary columns

train= train.drop(['source_system_tab','source_screen_name','source_type'],axis=1)
train.head()


# In[13]:


#renaming the Columns 
train.rename(columns={'msno':'user_id'},inplace=True)
train.head()


# In[14]:


train.shape


# # Processing the Dataset

# In[15]:


a=train.merge(song,on='song_id')
a=a.drop(['song_length','language'],axis=1)
a.head()


# In[16]:


a=a.merge(song_info,on='song_id').drop('isrc',axis=1)
a.rename(columns={'name':'song_name'},inplace=True)
a.head(10)


# # Data Cleaning

# In[17]:


a['genre_ids'].value_counts()


# In[18]:


a['genre_ids']=a['genre_ids'].str.replace('|',' ',regex=True)
a['genre_ids'].value_counts()


# In[ ]:




