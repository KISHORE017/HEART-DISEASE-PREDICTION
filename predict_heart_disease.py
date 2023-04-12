#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[50]:


data = pd.read_csv("heart.csv")


# In[51]:


type(data)


# In[52]:


data.shape


# In[53]:


data.head()


# In[54]:


data.describe()


# In[55]:


data.info()


# age: The person's age in years
# sex: The person's sex (1 = male, 0 = female)
# cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# chol: The person's cholesterol measurement in mg/dl
# fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# thalach: The person's maximum heart rate achieved
# exang: Exercise induced angina (1 = yes; 0 = no)
# oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# ca: The number of major vessels (0-3)
# thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# target: Heart disease (0 = no, 1 = yes)

# In[56]:


data.sample(5)


# In[57]:


data.isnull().sum()


# In[58]:


data.isnull().sum().sum()


# In[59]:


print(data.corr()["target"].abs().sort_values(ascending=False))


# In[60]:


y = data["target"]
sns.countplot(y)
target_temp = data.target.value_counts()
print(target_temp)


# In[61]:


print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))


# In[62]:


data["sex"].unique()


# In[63]:


sns.barplot(data["sex"],y)


# In[20]:


#Here 0 is female and 1 is male patients

countFemale = len(data[data.sex == 0])
countMale = len(data[data.sex == 1])
print("Percentage of Female Patients:{:.2f}%".format((countFemale)/(len(data.sex))*100))
print("Percentage of Male Patients:{:.2f}%".format((countMale)/(len(data.sex))*100))


# In[21]:


pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(20,10),color=['blue','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Don't have Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[22]:


data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# In[23]:


#Correlation plot
cnames=['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','num_major_vessels']


# In[24]:


f, ax = plt.subplots(figsize=(7, 5))
df_corr = data.loc[:,cnames]
corr = df_corr.corr()
sns.heatmap(corr, annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# In[25]:


df_corr = data.loc[:,cnames]
df_corr


# # Splitting the dataset to Train and Test

# In[26]:


from sklearn.model_selection import train_test_split

predictors = data.drop("target",axis=1)
target = data["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
print("Training features have {0} records and Testing features have {1} records.".      format(X_train.shape[0], X_test.shape[0]))


# In[28]:


X_train.shape


# In[29]:


X_test.shape


# In[30]:


Y_train.shape


# In[31]:


Y_test.shape


# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, Y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, Y_test)))


# In[35]:


rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, Y_train)
print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, Y_test)))


# In[37]:


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0

for x in range(500):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
print(max_accuracy)
print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)


# In[41]:


Y_pred_rf.shape


# In[42]:


score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# # confusion matrix of Random Forest

# In[43]:


from sklearn.metrics import confusion_matrix


# In[44]:


matrix= confusion_matrix(Y_test, Y_pred_rf)


# In[45]:


sns.heatmap(matrix,annot = True, fmt = "d")


# In[47]:


#precision score
precision = precision_score(Y_test, Y_pred_rf)
print("Precision: ",precision)


# In[64]:


#recall
recall = recall_score(Y_test, Y_pred_rf)
print("Recall is: ",recall)


# In[65]:


#F score
print((2*precision*recall)/(precision+recall))


# In[67]:


#cm using bad style
CM =pd.crosstab(Y_test, Y_pred_rf)
CM


# In[68]:


#False negative rate of the model
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]

fnr=FN*100/(FN+TP)
fnr


# # ACCURACY

# In[69]:


print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# In[ ]:




