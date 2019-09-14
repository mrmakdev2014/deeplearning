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
from sklearn import svm 
from matplotlib import style 
style.use("ggplot")


my_input  = np.array([[3,2],[6,6],
         [2.6,3],[7,8] ,[3.5,5],
         [6,11]
         ]
        )

my_output = [0,1,0,1,0,1]

""" svm.SVR  = Support Vector Regression  """
my_model = svm.SVR(kernel='linear' ,C=1.0)
my_model.fit(my_input , my_output)

print ("SVR predicate [0.5,0.8] : " , my_model.predict([[0.5,0.8]]))
print ("SVR predicate [8.5,10] : " , my_model.predict([[8.5,10]]))


plt.scatter(my_input[:,0] , my_input[:,1],c=my_output)


""" c= r mean color = red """
plt.scatter(0.5,0.8,c='r')
plt.scatter(8.5,10,c='r')

plt.show()




