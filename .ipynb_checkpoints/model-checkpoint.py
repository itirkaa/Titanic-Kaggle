# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 01:13:30 2019

@author: Aakriti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
total_data = [train_data, test_data]

"""
train_data.info()
train_data.describe()
# to check corelation between survival rate to different attributes
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()
train_data[['Sex', 'Survived']].groupby(['Sex']).mean() # females, higher survival
train_data[['Parch', 'Survived']].groupby(['Parch']).mean()
train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean() 
train_data[['Fare', 'Survived']].groupby(['Fare']).mean() # higher fare, higher survival
"""

# correcting by droping data columns
for dataset in total_data:
    dataset.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
    
    # converting Categorical Data
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)
    
    # Completing a numerical continous feature
    dataset["Age"].fillna(dataset["Age"].mode()[0], inplace=True)
    # Binning ages
    dataset['Age'] = pd.cut(dataset['Age'], bins=5, labels=[0,1,2,3,4])
    
    # Creating new features 
    # train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1 # size of each family
    dataset['IsAlone'] = 0 # not alone
    dataset.loc[dataset['SibSp']+dataset['Parch']==0, 'IsAlone'] = 1
    dataset.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    
    # Converting numerical features
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    dataset['Fare'] = pd.qcut(dataset['Fare'], q=4, labels=[0,1,2,3])
    
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Embarked'] = dataset['Embarked'].map({'Q': 0, 'S': 1, 'C': 2}).astype(int)
    
# input
train_data.drop('PassengerId', axis=1, inplace=True)
X = train_data.iloc[:, 1:]
y = train_data.loc[:,'Survived']
X_test = test_data.iloc[:, 1:]

"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
for x in [X, X_test]:
    #labelencoder_X = LabelEncoder()
    #X.loc[:,"Embarked"] = labelencoder_X.fit_transform(X.loc[:,"Embarked"])
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    ct = ColumnTransformer([("Any Name", OneHotEncoder(), [4])], remainder="passthrough")
    dataset = np.array(ct.fit_transform(dataset), dtype=np.float)

"""

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(random_state=0)
forest_model.fit(X_train, y_train)
pred = forest_model.predict(X_val)
score = forest_model.score(X_train, y_train)
error = mean_absolute_error(y_val, pred)
#print("Mean Absolute Error {} \n Score {}".format(error, score))

final_model = RandomForestClassifier(random_state=1)
final_model.fit(X, y)
predictions = forest_model.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
    
submission.to_csv('submission.csv', index=False)