# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:31:57 2017



import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

file1=pd.ExcelFile("Cereals.xlsx")
df2=file1.parse("Sheet2")

print df2.columns
print df2.shape


print df2["fat"]

rating=df2["rating"]


feature_columns=["fat","fiber","sugars"]
features=df2[feature_columns]#features is the feature matrix

rating=df2["rating"]# label is the label series

print features.shape
print rating.shape

print features[:5]
print rating[:5]


##SEABORN library for data visualization for analyzing correlation between features and response variable
sns.pairplot(df2,x_vars=feature_columns,y_vars="rating")
plt.savefig("3_features_rating.png")
## or figure can be saved in pdf format ;can be seen below
##plt.savefig("fig3.pdf")


#you can look at the correlation between any feature and response variable independently
sns.pairplot(df2,x_vars="fat",y_vars="rating")
plt.savefig("fat_rating.png")

sns.pairplot(df2,x_vars="fiber",y_vars="rating")
plt.savefig("fiber_rating.png")

sns.pairplot(df2,x_vars="sugars",y_vars="rating")
plt.savefig("sugars_rating.png")


#train=df2[:60]
#test=df2[-17:]

#import cross validation from sklearn and split data into train and test
from sklearn.cross_validation import train_test_split
features_train,features_test,rating_train,rating_test=train_test_split(features,rating,random_state=1)

#default split is 75 % for train and 25 % for test
#default split is 75 % for train and 25 % for test

print features_train.shape#57 rows for train
print rating_train.shape#57 instances for train data

print features_test.shape#20 instances for test
print rating_test.shape# 20 rows for test

#import linear regression from sklearn
from sklearn.linear_model import LinearRegression

#instantiate
linreg=LinearRegression()

#fit the model to the training data (learn the coefficents)
linreg.fit(features_train,rating_train)
#model is learning the intercept and coefficents due to the train data;for the line of best fit (57 samples)
print linreg

#print the intercept and coefficents
print linreg.intercept_
print linreg.coef_

#pair the feature names with the coefficents
zip(feature_columns,linreg.coef_)

#make predictions on the testing set
rating_pred=linreg.predict(features_test)

#we need an evaluation metric to compare predicted values with actual ones

rating_predList=[]
for i in rating_pred:
    rating_predList.append(i)
    
    

rating_actual=[]
for i in rating_test:
    rating_actual.append(i)

#accuracy is not supported for continuos values;accuracy is not useful for linear regression    
from sklearn import metrics
print metrics.mean_absolute_error(rating_actual,rating_predList)  
print metrics.mean_squared_error(rating_actual,rating_predList)

#now RMSE is 4.93,but in the model that we take all features, it is 5.95;it improves al lot with those features 
    
#calculate RMSE using sklearn
#calculate RMSE using sklearn
import numpy as np 
print np.sqrt(metrics.mean_squared_error(rating_actual,rating_predList))#calculate RMSE



print np.sqrt(metrics.mean_squared_error(rating_test,rating_pred)) 

from matplotlib import pyplot as plt
       
    