#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mnist = fetch_openml("mnist_784")


# In[3]:


mnist


# In[4]:


mnist.data[:5]


# In[5]:


mnist.target[:5]


# In[21]:


# converting data to array
import numpy as np
arr1=np.array(mnist.data)
arr2=np.array(mnist.target)
print(arr1[0][679])


# In[7]:


# Visualising the given dataset
c=0
for image in arr1[:5]:
    while c < len(arr2):
        label=arr2[c]
        plt.subplot(1,5, c+1)
        plt.imshow(np.reshape(image, (28,28)), cmap="gray")
        plt.title("Number: %s" % label)
        break
    c=c+1


# In[8]:


# spliotting dataset into training set and test set using train_test_split function
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42) 


# In[9]:


# Fitting data in the LogisticRegression and traioning the alogorithms
mdl = LogisticRegression(solver="lbfgs")
mdl.fit(X_train, y_train)
predictions= mdl.predict(X_test)
score = mdl.score(X_test, y_test)
print(score)


# In[10]:


print(y_test[:5])
mdl.predict(X_test[:5])


# In[11]:


X_test_array=np.array(X_test)
print(len(X_test_array))


# In[12]:


# Converting predictions data into array
predictions_array = np.array(predictions)


# In[13]:



index = 0
plt.imshow(np.reshape(X_test_array[index], (28,28)))
print("Prediction= ", predictions_array[index])
#print("prediction: " + mdl.predict([X_test_array[index]]))[0]


# In[14]:


# Confussion matrix
cm= metrics.confusion_matrix(y_test, predictions)
cm


# In[15]:


import seaborn as sns
fig = plt.figure(figsize=(10,10))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g');
ax.set_xlabel('Predicted', fontsize=20)
ax.set_ylabel('Actual', fontsize=20)


# In[ ]:




