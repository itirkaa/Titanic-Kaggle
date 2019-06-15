# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 01:13:30 2019

@author: Aakriti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')

# removing id column
train_data.drop(columns="PassengerId", inplace=True)

# conversion to Binary

features = ['Survived','Pclass','Sex','Age', 'Parch', 'Embarked']
train_data = train_data[features]
train_data = train_data.dropna()
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# input
X = train_data.iloc[:,1:]
y = train_data['Survived']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#labelencoder_X = LabelEncoder()
#X.loc[:,"Embarked"] = labelencoder_X.fit_transform(X.loc[:,"Embarked"])
ct = ColumnTransformer([("Any Name", OneHotEncoder(), [4])], remainder="passthrough")
X = np.array(ct.fit_transform(X), dtype=np.float)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)


from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=0)
tree_model.fit(X_train, y_train)
pred = tree_model.predict(X_val)

from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(y_val, pred)
print(error)


from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(random_state=0)
forest_model.fit(X_train, y_train)
pred2 = forest_model.predict(X_val)

error = mean_absolute_error(y_val, pred2)
print(error)






