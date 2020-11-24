#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import *
from numpy import *
import matplotlib.pyplot as plt  ##Import Libraries
import pylab
from matplotlib.pyplot import figure 
import math

infile=open('separate_11_23.txt', 'r')
data = []
for line in infile:
    data.append(line.strip().split())

infile.close()

#infile=open('bettersat.txt', 'r')
#sat = []
#for line in infile:
#    sat.append(line.strip().split())

#infile.close()

for line in data:
    for i in range(len(line)):
        line[i] = float(line[i])
        
#for line in sat:
#    for i in range(len(line)):
#        line[i] = float(line[i])

x = []
y = []
for line in data:
    x.append(line[0])
    y.append(line[1])
    
y = [i / 3.12 for i in y]
    
#satx = [i - 0.000005 for i in x]
#saty = []
#for line in sat:
 #  saty.append(line[1])

#ycorr = [y - b for y,b in zip(y,saty)]


print(x[1130])

x = x[1130:2800]
y = y[1130:2800]

plt.figure(figsize=(50,50))
#print(max(y))    
    
from scipy.signal import find_peaks_cwt
cb = np.array(y)
indexes = find_peaks_cwt(cb, np.arange(1, 800))
print('Peaks are: %s' % (indexes))
index_to_x = [i * 0.000001 + 0.01101 - 0.011218  for i in indexes]
print('Times  are: %s' % (index_to_x))

for xc in index_to_x:
    plt.axvline(x=xc)
    
    
    

x = [i - 0.011218 for i in x]

def airy(x_data,fine,lamb):
       return [1.0 / (1 + fine * (math.sin(math.pi * p / lamb)** 2)) for p in x_data]
popt, pcov = curve_fit(airy,x,y,p0=[150,0.00125])
print("best_vals: {}".format(popt))


print(x[1130])

#plt.scatter(x,ycorr,label='doppler - sat')
plt.scatter(x,y)
plt.plot(x,airy(x,*popt))
#plt.scatter(satx,saty)
plt.show()


# In[ ]:




