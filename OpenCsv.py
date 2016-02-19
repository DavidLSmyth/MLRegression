# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:11:12 2016

@author: Marion
"""

import pandas as pd
import numpy as np
from pandas.stats.api import ols

import matplotlib.pyplot as plt

d = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
file=pd.read_csv('kc_house_train_data.csv',dtype=d)

#increases the separation between few and many bedrooms
#inctreases the variance of bedroom data
b_sq=file['bedrooms']*file['bedrooms']
#interaction variable, large when both large
bed_bath=pd.DataFrame(file['bedrooms']*file['bathrooms'])
#spreads small values, clusters large values
logsqft=np.log(file['sqft_living'])
#doens't do anything really!
lat_long=file['lat']+file['long']
price=file['price']
#file=file+b_sq
print("average values")
print(sum(b_sq)/len(b_sq))
print(sum(bed_bath)/len(bed_bath))
print(sum(logsqft)/len(logsqft))
print(sum(lat_long)/len(lat_long))
x=file[['sqft_living','sqft_living', 'bedrooms', 'bathrooms', 'lat','long']]
#x1=file[['sqft_living','sqft_living', 'bedrooms', 'bathrooms', 'lat','long']]
print(type(x),type(bed_bath))
#res=ols(y=price,x=file[['sqft_living', 'bedrooms', 'bathrooms', 'lat','long']])
res1=ols(y=price,x=file[['sqft_living', 'bedrooms', 'bathrooms', 'lat','long']])
plt.scatter(file['sqft_living'],price/1000)
plt.xlabel('sqft_living')
plt.ylabel('price')
print('data')
#print(res1.beta)