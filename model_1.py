import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

df=pd.read_csv(r'Fitbit-data.csv')

df.head()

df.info()
data= df[["SO02_level", "resting_heart_rate", "average_weekly_steps", "sleep_average_weekly"]]
data.head()
data_2=data
import random

for x in range(9999):
    if(data_2.at[x,'sleep_average_weekly']>8):
        data_2.at[x,'sleep_average_weekly']=random.randint(5,7)

index_list=[]
for x in range(9999):
    if(data_2.at[x,'sleep_average_weekly']>6):
        index_list.append(x)

for j in range(1000):
       data_2.at[random.choice(index_list),'sleep_average_weekly']=random.randint(9,11)
rows=9999
new_col=[]
for j in range(9999):
    if(6<=data_2.at[j,'sleep_average_weekly']<=8 & 55<=data_2.at[j,'resting_heart_rate']<=85):
        new_col.append("Normal")
    else:
        new_col.append("Alert")
data_2.loc[:, "Class"] = new_col

counts = data_2['Class'].value_counts()

print(counts)


index_list=[]
for x in range(9999):
    if(55<=data_2.at[x,'resting_heart_rate']<=70):
        index_list.append(x)

for j in range(1072):
       data_2.at[random.choice(index_list),'resting_heart_rate']=random.randint(69,86)

rows=9999
new_col=[]
for j in range(9999):
    if( 6<=data_2.at[j,'sleep_average_weekly']<=8 & 60<=data_2.at[j,'resting_heart_rate']<=85 & 90<data_2.at[j, "SO02_level"]<100):
        new_col.append("Normal")
    else:
        new_col.append("Alert")
data_2.loc[:, "Class"] = new_col

counts = data_2['Class'].value_counts()

print(counts)

from sklearn.metrics  import f1_score,accuracy_score
from sklearn.model_selection import train_test_split

data_2['Class'] = data_2['Class'].map({"Normal":0,'Alert':1})
x=data_2.drop(["Class"],axis=1)
y=data_2["Class"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

NB = GaussianNB()
NB.fit(x_train, y_train)
pred_NB = NB.predict(x_test)
print(classification_report(y_test, pred_NB))
accuracy_score(pred_NB,y_test)

import joblib
joblib.dump(NB, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
NB = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
