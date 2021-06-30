# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:47:11 2021

@author: 10275
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("train.csv")
strs = df['target'].value_counts()
value_map = dict((v, i) for i,v in enumerate(strs.index))
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
df = df.replace({'target':value_map})
df = df.drop(columns=['id'])
x_train = df.iloc[:, :-1]
y_train = df['target']
df = pd.read_csv("test.csv")
# df = df.drop(columns=['id'])
x_test = df.iloc[:, 1:] # keep the id column for output
df = pd.read_csv("test.csv")

model = RandomForestClassifier(n_jobs=2, n_estimators=500)
model.fit(x_train, y_train)
proba = model.predict_proba(x_test)
# acc = accuracy_score(y_test, y_pred) * 100
# print("\nTesting Accuracy: {:.3f} %".format(acc))
output = pd.DataFrame({'id': df['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3]})
output.to_csv('submission_RF.csv', index=False)