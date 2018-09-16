# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:31:57 2017



import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

file1=pd.ExcelFile("Cereals.xlsx")
df2=file1.parse("Sheet3")

print df2.columns
print df2.shape


print df2["fat"]

rating=df2["rating"]


feature_columns=["calories","protein","fat","sodium","fiber","carbo","sugars","potass","vitamins","shelf","weight","cups"]
features=df2[feature_columns]#features is the feature matrix

rating=df2["rating"]# rating is the label series

##SEABORN library for data visualization for analyzing correlation between features and response variable
sns.pairplot(df2,x_vars=feature_columns,y_vars="rating")
plt.savefig("all_features_rating.png")

#for saving pdf format 
sns.pairplot(df2,x_vars=feature_columns,y_vars="rating")
plt.savefig("all_features_ratingPDF.pdf")

sns.pairplot(df2,x_vars="calories",y_vars="rating")
plt.savefig("calories_rating.png")

sns.pairplot(df2,x_vars="protein",y_vars="rating")
plt.savefig("protein_rating.png")

sns.pairplot(df2,x_vars="sodium",y_vars="rating")
plt.savefig("sodium_rating.png")
#no correlation with sodium and rating

sns.pairplot(df2,x_vars="carbo",y_vars="rating")
plt.savefig("carbo_rating.png")
#no correlation with carbo and rating

sns.pairplot(df2,x_vars="potass",y_vars="rating")
plt.savefig("potass_rating.png")
#no correlation

sns.pairplot(df2,x_vars="vitamins",y_vars="rating")
plt.savefig("vitamins_rating.png")
# NO correlation;it can be easily seen

sns.pairplot(df2,x_vars="shelf",y_vars="rating")
plt.savefig("shelf_rating.png")
# NO correlation;it can be easily seen

sns.pairplot(df2,x_vars="weight",y_vars="rating")
plt.savefig("weight_rating.png")
# NO correlation;it can be easily seen

sns.pairplot(df2,x_vars="cups",y_vars="rating")
plt.savefig("cups_rating.png")
# NO correlation;it can be easily seen

















###############################################
print features.shape
print rating.shape

print features[:5]
print rating[:5]

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
print metrics.mean_absolute_error(rating_predList,rating_actual)  
print metrics.mean_squared_error(rating_predList,rating_actual) 
    
#calculate RMSE using sklearn
#calculate RMSE using sklearn
import numpy as np
print np.sqrt(metrics.mean_squared_error(rating_actual,rating_predList))


print np.sqrt(metrics.mean_squared_error(rating_test,rating_pred)) 

from matplotlib import pyplot as plt
       
    