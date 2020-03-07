# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:59:46 2020

@author: Ahadit
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:01:38 2020

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,0].values
y = dataset.iloc[:, 1].values
n=np.size(x)
m_x,m_y=np.mean(x),np.mean(y)
#ss_xy=np.sum(y*x)-n*m_x*m_y
#ss_xx=np.sum(x*x)-n*m_x*m_x
#b0_1=ss_xy/ss_xx  
nr=np.sum((x-m_x)*(y-m_x))
dr=np.sum((x-m_x)**2)
b0_1=nr/dr
b0_0=m_y-b0_1*m_x
y_pred=b0_0+b0_1*x
print(b0_0)
print(b0_1)
plt.scatter(x,y)
plt.plot(x,y_pred, color='r',marker='o')
from sklearn.metrics import r2_score
score=r2_score(y, y_pred)
print(score)
plt.show()
#print("Estimated coefficients:\nb0_0={} \nb0_1={})",.format(b0_0,b0_1))
