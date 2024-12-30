#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_csv(r"C:\Users\Lenovo\OneDrive - RandomTrees\Desktop\diabetes.csv")


# In[6]:


data.head()


# In[7]:


data.isna().sum()


# In[8]:


data.describe()


# In[9]:


data.duplicated().sum()


# In[10]:


plt.figure(figsize=(12,6))
sns.countplot(x='Outcome',data=data)
plt.show()


# In[11]:


plt.figure(figsize=(12,12))
for i,col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=col,data=data)
plt.show()    
    
    


# In[12]:


sns.pairplot(data, hue='Outcome')
plt.show()


# In[13]:


plt.figure(figsize=(12,12))
for i,col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']):
    plt.subplot(3,3,i+1)
    sns.histplot(x=col,data=data,kde=True)
plt.show()    


# In[14]:


plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),vmin=-1.0,center=0,cmap='RdBu_r',annot=True)
plt.show()


# In[15]:


from sklearn.preprocessing import StandardScaler
x_scale=StandardScaler()
x=pd.DataFrame(x_scale.fit_transform(data.drop(['Outcome'],axis=1),),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
x


# In[16]:


y=data['Outcome']
y


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[18]:


x_train


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
test_score=[]
train_score=[]
for i in range(1,15):
    knn=KNeighborsClassifier(i)
    knn.fit(x_train,y_train)
    train_score.append(knn.score(x_train,y_train))
    test_score.append(knn.score(x_test,y_test))
    


# In[37]:


max_train_score=max(train_score)
train_scores_index=[i for i,v in enumerate(train_score) if v== max_train_score]
print("max train score {},and k={}".format(max_train_score*100,list(map(lambda x:x+1,train_scores_index))))


# In[39]:


max_test_score=max(test_score)
test_scores_index=[i for i,v in enumerate(test_score) if v==max_test_score]
print("max test score{},and k={}".format(max_test_score*100,list(map(lambda x:x+1,test_scores_index))))


# In[49]:


plt.figure(figsize=(12,5))
p=sns.lineplot(x=range(1,15),y=train_score,marker='*',label='train_score')
p=sns.lineplot(x=range(1,15),y=test_score,marker='o',label='test_score')
plt.show()


# In[52]:


knn=KNeighborsClassifier(7)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)


# In[55]:


from sklearn.metrics import confusion_matrix
y_pred=knn.predict(x_test)
confusion_matrix(y_test,y_pred)


# In[56]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




