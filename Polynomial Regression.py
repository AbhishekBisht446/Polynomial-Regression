# polynomial_Regression
import matplotlib.pyplot as py
import numpy as np
import pandas as pd 

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values #independent variable
Y = dataset.iloc[:,2].values   #dependent variable

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


#fitting polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visualzing the Linear Regression result
py.scatter(X,Y,color="red")
py.plot(X,lin_reg.predict(X),color="blue")
py.title("Truth or Bluff(Linear Regression Graph)")
py.xlabel("Position level")
py.ylabel("Salaries")

#visusalzing of polynomial REgression
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
py.scatter(X,Y,color="red")
py.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
py.title("Truth or Bluff(Linear Regression Graph)")
py.xlabel("Position level")
py.ylabel("Salaries")
py.show()



#Predict a new result with Linear Regression
lin_reg.predict(np.array(6.5).reshape(1,-1))

#Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1,-1)))