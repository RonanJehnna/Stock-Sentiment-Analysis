# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 01:07:13 2022

@author: anils
"""

import pandas as pd

df = pd.read_csv('Stock Sentiment Analysis/Data.csv', encoding = "ISO-8859-1")
df.head()

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

#Removing punctuations
data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]", " ", regex = True, inplace = True)

# Renaming column names for ease of access
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index
data.head(5)

# Converting headlines to lowercase
for index in new_Index:
    data[index] = data[index].str.lower()
data.head(1)

headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))
headlines[0]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# implement Bag of Words
countVector = CountVectorizer(ngram_range=(2,2))
traindataset = countVector.fit_transform(headlines)

# implement RandomForest Classifier
randomClassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
randomClassifier.fit(traindataset, train['Label'])

# Predict for the Test Dataset
test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = countVector.transform(test_transform)
predictions = randomClassifier.predict(test_dataset)

# import library to check accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

matrix = confusion_matrix(test['Label'], predictions)
print(matrix)
score = accuracy_score(test['Label'], predictions)
print(score)
report = classification_report(test['Label'], predictions)
print(report)