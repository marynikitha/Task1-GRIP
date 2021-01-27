# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



        
import pandas as pd
import numpy as np


from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('http://bit.ly/w-data')
print(df)

plt.scatter(df['Hours'], df['Scores'], label= "stars", color= "green", marker= "*", s=30)


plt.xlabel('Number of hours studied') 

plt.ylabel('Percentage of marks') 

plt.title('Marks vs Hours plot')
plt.show()

sns.regplot(x= df['Hours'], y= df['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


#training data
# Defining X and y from the Data
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
reg = LinearRegression()
reg.fit(train_X, train_y)
print("\n---------Model Trained---------\n")

#prediction model
prediction_y = reg.predict(val_X)
prediction = pd.DataFrame({' Number of Hours studied': [i[0] for i in val_X], 'Predicted Marks': [j for j in prediction_y]})

print(prediction)
print('\n\n')
#comparing actual vs predicted marks
compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': prediction_y})
print(compare_scores)

#accueacy of model
from sklearn.metrics import mean_squared_error
import math

print('\n\nMean square error: ',math.sqrt(mean_squared_error(val_y, prediction_y)))
hours = [9.25]
answer = reg.predict([hours])
print("Score if number of hours studied is 9.25 = {}".format(round(answer[0],3)))
