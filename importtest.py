#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import *
from numpy import *
import matplotlib.pyplot as plt  ##Import Libraries
import pylab
from matplotlib.pyplot import figure 


infile=open('doppler2_11-2.txt', 'r')
data = []
for line in infile:
    data.append(line.strip().split())

infile.close()

infile=open('background_11-2.txt', 'r')
back = []
for line in infile:
    back.append(line.strip().split())

infile.close()

for line in data:
    for i in range(len(line)):
        line[i] = float(line[i])
        
for line in back:
    for i in range(len(line)):
        line[i] = float(line[i])

x = []
y = []
for line in data:
    x.append(line[0])
    y.append(line[1])
    
backy = []
for line in back:
    backy.append(line[1])

ycorr = [y - b for y,b in zip(y,backy)]

plt.figure(figsize=(50,50))
    
    
from scipy.signal import find_peaks_cwt
cb = np.array(ycorr)
indexes = find_peaks_cwt(cb, np.arange(1, 350))
#print('Peaks are: %s' % (indexes))
index_to_x = [i * 0.000001 + 0.00827 for i in indexes]
print('Times  are: %s' % (index_to_x))

for xc in index_to_x:
    plt.axvline(x=xc)
    

plt.scatter(x,ycorr,label='doppler - back')
#plt.scatter(x,y)
#plt.scatter(x,backy)
plt.show()


# In[ ]:




