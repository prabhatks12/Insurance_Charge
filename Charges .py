#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Lets get the data and explore it

data=pd.read_csv("G:\ReStart\Linear_Regression\insurance.csv")
data.head()
data.describe()

# Converting all String values to number using map function,similar thing can be done by using one hot encoder or label encoder

data['sex'] = data.sex.map({'female':0,'male':1})
data['smoker'] = data.smoker.map({'yes':0,'no':1})
data['region'] = data.region.map({'southwest':0,'northwest':1,'southeast':2,'northeast':3})
data.head()

# Now let's check how much each attribute affects the final results

X=data.iloc[:,0:6]
Y=data.iloc[:,6]
print(X.head())
print("\n\n",Y.head())


from sklearn.tree import DecisionTreeRegressor
importance = DecisionTreeRegressor()
importance.fit(X, Y.ravel())
print(importance.feature_importances_)

# As we can see smoker attribute affects the data most followed by bmi and age.Even if you remove remaining attributes our final model will not be affected much.But I am keeping rest of the attributes.
# Now we are checking for colinearity in attributes which means how much one attribute depends on other.This one is the best approach

import seaborn as sb
sb.heatmap(data.corr(),cmap = 'Blues', annot=True)
sb.pairplot(data)

# Charges column is te target and rest are used to predict it.

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,Y_train)
model.predict(X_train)
print(model.score(X,Y))


# As we can see the model's score is not good enough.So we will try to imporve it and also use diferrent models to measure their accuracy.  

from sklearn.preprocessing import PolynomialFeatures

poly_reg  = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_poly,Y,test_size=0.25)

lin_reg = LinearRegression()
lin_reg  = lin_reg.fit(X_train,Y_train)

pred=lin_reg.predict(X_test)
print(lin_reg.score(X_poly,Y))

print(X.shape)

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((1338,1)).astype(float),values=X,axis=1)

x_opt=x[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=x_opt).fit()
regressor_ols.summary()

x_train,x_test,y_train,y_test = train_test_split(x_opt,Y,test_size=0.20)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
pred=regressor.predict(x_test)
print(regressor.score(x_opt,Y))



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, Y_train)

pred = regressor.predict(X_test)
print(regressor.score(X_test,Y_test))
print(pred[0:10])
print(Y_test[0:10])


# So linear regression yields the best result after some preprocessing. 
