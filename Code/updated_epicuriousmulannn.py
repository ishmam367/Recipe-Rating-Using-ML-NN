# -*- coding: utf-8 -*-
"""Updated_epicuriousMulAnnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Dn292bGEqV4nA3j6VSgYGc9dJDRYVcs4
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import pandas as pd
import numpy as np

dataset=pd.read_csv('data.csv')

X = dataset.drop(['Index'],axis=1)
#dataset['RatingClass']=dataset['RatingClass'].replace({'Excellent':'2','Good':'1','Average':'0'})

X=dataset[[
       'Servings', 'Ingredients',
       'Instructions']]
y=dataset['RatingClass']
encoder = LabelEncoder()
y1=encoder.fit_transform(y)
Y=pd.get_dummies(y1).values


# convert integers to dummy variables (i.e. one hot encoded)
#dummy_y = np_utils.to_categorical(encoded_y)

from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=123)

model=Sequential()
model.add(Dense(12,input_dim=3,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(10,activation='relu'))
 
model.add(Dense(3,activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20,batch_size=10)

scores1=model.evaluate(X_train,y_train)

print("Training Accuracy",model.metrics_names[1],scores1[1]*100)


y_pred=model.predict(X_test)
y_test_class=np.argmax(y_test,axis=1)
y_pred_class=np.argmax(y_pred,axis=1)
from sklearn.metrics import classification_report,confusion_matrix
print (classification_report(y_test_class,y_pred_class))
print (confusion_matrix(y_test_class,y_pred_class))

print("Testing Accuracy",(7507)/(7507+1032+148))