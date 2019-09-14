# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:03:48 2019

@author: Makeen
"""
"""
Video Link 
https://youtu.be/7sJ9hQYpkbU?list=PLYW0LRZ3ePo4TFeouzYff88TStgS-X52R&t=1134
"""
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from matplotlib import style 
style.use("ggplot")


my_input  = np.array([[3,2],[6,6],
         [2.6,3],[7,8] ,[3.5,5],
         [6,11]
         ]
        )

my_model  = KMeans(n_clusters =2)
my_model.fit(my_input)
 
print ("Clusers_Centers :\n ", my_model.cluster_centers_)
print ("labels: " ,my_model.labels_)

colors = ["g.","r.", "c.","y."]

plt.scatter(my_input[:,0], my_input[:,1],c=my_model.labels_)

plt.scatter(my_model.cluster_centers_[:,0], my_model.cluster_centers_[:,1],marker="x", s=250,linewidths=5)

plt.show()

