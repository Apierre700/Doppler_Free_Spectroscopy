#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import *
from numpy import *
import matplotlib.pyplot as plt  ##Import Libraries
import pylab
from matplotlib.pyplot import figure 
import math

infile=open('260mvpp_sep.txt', 'r')
data = []
for line in infile:
    data.append(line.strip().split())

infile.close()

infile=open('260_vsweep.txt', 'r')
sat = []
for line in infile:
    sat.append(line.strip().split())

infile.close()

for line in data:
    for i in range(len(line)):
        line[i] = float(line[i])
        
for line in sat:
    for i in range(len(line)):
        line[i] = float(line[i])

x = []
y = []
for line in data:
    x.append(line[0])
    y.append(line[1])
    
y = [i / 3.16 for i in y]
    
saty = []
for line in sat:
    saty.append(line[1])

#ycorr = [y - b for y,b in zip(y,saty)]

plt.figure(figsize=(50,50))
    
    
from scipy.signal import find_peaks_cwt
cb = np.array(y)
indexes = find_peaks_cwt(cb, np.arange(1, 800))
print('Peaks are: %s' % (indexes))
index_to_x = [i * 0.000002 + 0.00876 - 0.010932 for i in indexes]
print('Times  are: %s' % (index_to_x))

for xc in index_to_x:
    plt.axvline(x=xc)

x = [i - 0.010932 for i in x]
print(saty[1086],saty[2903],saty[1086] - saty[2903])

def airy(x_data,fine,lamb):
       return [1.0 / (1 + fine * (math.sin(math.pi * p / lamb)** 2)) for p in x_data]
popt, pcov = curve_fit(airy,x,y,p0=[1500,0.00362])
print("best_vals: {}".format(popt))



#plt.scatter(x,ycorr,label='doppler - sat')
plt.scatter(x,y)
plt.plot(x,airy(x,*popt))
plt.scatter(x,saty)
plt.show()


# In[ ]:




