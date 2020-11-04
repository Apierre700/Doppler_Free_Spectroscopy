#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import *
from numpy import *
import matplotlib.pyplot as plt  ##Import Libraries
import pylab
from matplotlib.pyplot import figure 


infile=open('betterdop.txt', 'r')
data = []
for line in infile:
    data.append(line.strip().split())

infile.close()

infile=open('bettersat.txt', 'r')
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
    
saty = []
for line in sat:
    saty.append(line[1])

ycorr = [y - b for y,b in zip(y,saty)]

plt.figure(figsize=(50,50))
    
    
from scipy.signal import find_peaks_cwt
cb = np.array(y)
indexes = find_peaks_cwt(cb, np.arange(1, 800))
#print('Peaks are: %s' % (indexes))
index_to_x = [i * 0.0000004 + 0.00934 for i in indexes]
print('Times  are: %s' % (index_to_x))

for xc in index_to_x:
    plt.axvline(x=xc)
    

#plt.scatter(x,ycorr,label='doppler - sat')
plt.scatter(x,y)
plt.scatter(x,saty)
plt.show()


# In[ ]:




