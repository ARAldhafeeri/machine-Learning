# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:01:15 2020

@author: iahme
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

class Data:
  dataset = pd.read_csv('Salary_Data.csv')
  # Independent variable
  x = dataset.iloc[:, :-1].values
  # Dependent variable
  y = dataset.iloc[:, 1].values
  
  def __init__(self):
    pass


class Moudle:
  x_train, x_test, y_train, y_test = train_test_split(Data.x,
                                                    Data.y,
                                                    test_size = 0.2,
                                                    random_state = 0)
  # Fitting Simple linear regression to the Training set

  regressor = LinearRegression()
  regressor.fit(x_train, y_train)


  # Predicting the test set results
  y_pred = regressor.predict(x_test)

class Plot:
  # Visualising The training set results
  # Our observations for the salary imported from Salary_Data.csv
  plt.scatter(Moudle.x_train, Moudle.y_train, color = 'red')
  # Prediction for the training sets
  plt.plot(Moudle.x_train, Moudle.regressor.predict(Moudle.x_train), color = 'green')
  plt.title('Salary vs Experience(Training set)')
  plt.xlabel('Years of experience')
  plt.ylabel('Salary')
  plt.show()
  
  # Visualising the Test set results
  plt.scatter(Moudle.x_test, Moudle.y_test, color = 'green')
  plt.plot(Moudle.x_train, Moudle.regressor.predict(Moudle.x_train), color = 'blue')
  plt.title('Salary vs Experience(Training set)')
  plt.xlabel('Years of experience')
  plt.ylabel('Salary')
  plt.show()
  
  
if __name__=='__main__':
  Data()
  Moudle()
  Plot()