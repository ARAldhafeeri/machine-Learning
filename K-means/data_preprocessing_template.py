# Data Preprocessing Template
# The mall hire you to group the customers
# since the mall do not know how they classified the spending 
# This is a clustring problem. 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
y = dataset.iloc[:, 3].values

# Using the albow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
# we need 10 bounds so we use 1,11
for i in range(1,11):
  kmeans = KMeans(n_clusters = i,
                  init = 'k-means++',
                  max_iter = 300,
                  n_init = 10,
                  random_state = 0)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method') 
plt.xlabel("Number of Clusters")
plt.ylabel('wcss')
plt.show() 
  

# Applying k-means to the mall dataset

kmeans = KMeans(n_clusters = 5,
                init = 'k-means++',
                max_iter = 300,
                n_init = 10,
                random_state = 1)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0],
            X[y_kmeans == 0,1],
            s = 100, c= 'red',
            label = 'careful')
plt.scatter(X[y_kmeans == 1, 0],
            X[y_kmeans == 1,1],
            s = 100, c= 'blue',
            label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0],
            X[y_kmeans == 2,1],
            s = 100, c= 'green',
            label = 'Target')
plt.scatter(X[y_kmeans == 3, 0],
            X[y_kmeans == 3,1],
            s = 100, c= 'cyan',
            label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0],
            X[y_kmeans == 4,1],
            s = 100, c= 'magenta',
            label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centeroids')

plt.title('Clusters of clients')
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""