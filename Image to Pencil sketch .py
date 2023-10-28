#!/usr/bin/env python
# coding: utf-8

# # Image to Pencil sketch
# 

# ## Procedure
1.Import pages
2.Accept the Image
3.Convert into Gray Scale
4.Convert into Inverse GrayScale
5.Display the Pencil and Normal Image
# In[1]:


import cv2


# In[2]:


image=cv2.imread("dog.jpg")
cv2.imshow("Normal Image",image)
cv2.waitKey(0)


# In[3]:


gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image",gray)
cv2.waitKey(0)


# In[4]:


invers=255-gray
cv2.imshow("Inverse GrayScale Image",invers)
cv2.waitKey(0)


# In[5]:


blur=cv2.GaussianBlur(invers,(21,21),0)
inver_blur=255-blur


# In[6]:


pencil=cv2.divide(gray,inver_blur,scale=256.0)
cv2.imshow("Normal Image",image)
cv2.imshow("Pencil sketch",pencil)
cv2.waitKey(0)

