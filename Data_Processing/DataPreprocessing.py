# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:52:33 2020

@author: iahme
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:50:16 2020

@author: iahme
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd



# Import the data
dataset = pd.read_csv('Data.csv')

# takes all the rows, all the columns except the last column
x = dataset.iloc[:, :-1].values
y =  dataset.iloc[:, 3].values


# handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",
                  strategy="mean",
                  axis = 0)

imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(imputer)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labelencoder_x = LabelEncoder()
# We fited the encoder in the first column
x[: , 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0] )
x = onehotencoder.fit_transform(x).toarray()
x
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y

from sklearn.model_selection import train_test_split 

# Next line means 2 observatin in test set and 8 in the tranning set
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 0)
