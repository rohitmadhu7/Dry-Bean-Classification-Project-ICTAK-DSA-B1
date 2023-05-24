import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("project1.csv")
# scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
area = data['Area'].values.reshape(-1, 1)
scaled_area = scaler.fit_transform(area)
data['Area'] = scaled_area

# Label Encoder
le = LabelEncoder()
data['Class'] = le.fit_transform(data['Class'])
data.head()


y=data['Class']
x=data.drop(['Class'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Already split the fata for train and test, next is 
#Importing random forest from sklearn 

from sklearn.ensemble import RandomForestClassifier
#Creating Instance

rf_clf=RandomForestClassifier(random_state=42)

#Fitting the data

rf_clf.fit(x_train.values,y_train.values)
#pickling
pickle.dump(rf_clf,open('model.pkl','wb'))
pickle.dump(le,open('le.pkl','wb'))
pickle.dump(scaler,open('sc.pkl','wb'))

