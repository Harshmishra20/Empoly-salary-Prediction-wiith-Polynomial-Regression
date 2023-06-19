# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 23:48:23 2023

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'D:\Data Science\Daily Practice\March\16-03-2023\1.POLYNOMIAL REGRESSION\Position_Salaries.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(x, y)
from sklearn.preprocessing import PolynomialFeatures            
poly_reg=PolynomialFeatures(degree=7)            
            
            
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[7]]))
