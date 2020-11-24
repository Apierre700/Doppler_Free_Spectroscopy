#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import *
from numpy import *
import matplotlib.pyplot as plt  ##Import Libraries
import pylab
from matplotlib.pyplot import figure 


infile=open('unsat_11-11.txt', 'r')
data = []
for line in infile:
    data.append(line.strip().split())

infile.close()

infile=open('voltageline_11-11.txt', 'r')
volt = []
for line in infile:
    volt.append(line.strip().split())

infile.close()

infile=open('F2_analyzed_11-11.txt', 'r')
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

for line in volt:
    for i in range(len(line)):
        line[i] = float(line[i])
        
x = []
y = []
for line in data:
    x.append(line[0])
    y.append(line[1])
    


ue=[0.0001] * len(y)
le = [i for i in ue]

satx = [i + 0.0001 for i in x]
saty = []
for line in sat:
    saty.append(line[1])

voly = []
for line in volt:
    voly.append(line[1])
    
ycorr = [y - b for y,b in zip(y[800:3100],saty[700:3000])]
ycorr1 = [y - b for y,b in zip(y,saty)]
xcorr = satx[700:3000]

sue=[0.0001 * (2 ** 0.5)] * len(ycorr)
sle = [i for i in sue]

plt.figure(figsize=(50,50))
    
    
from scipy.signal import find_peaks_cwt
cb = np.array(y)
indexes = find_peaks_cwt(cb, np.arange(1, 800))
print('Peaks are: %s' % (indexes))
index_to_x = [i * 0.000001 + 0.00362 for i in indexes]
print('Times  are: %s' % (index_to_x))
  
cb = np.array(ycorr)
indexes = find_peaks_cwt(cb, np.arange(1, 300))
print('Peaks are: %s' % (indexes))
index_to_x = [i * 0.000001 + 0.00432 + 0.0001 for i in indexes]
print('Times  are: %s' % (index_to_x))

for xc in index_to_x[1:7]:
    plt.axvline(x=xc,color='k')

#domain1 = x[1230:1440]    
peakpos1 = 2375 * 0.000001 + 0.00362
peak1 = y[2375]
std1 = sqrt(mean([((i - peakpos1)**2) for i in x]))

def gaus(x,y0,peakpos0,std0):
       return y0*exp(-(x-peakpos0)**2/(2 * std0**2))    
    
pop1,pcov1 = curve_fit(gaus,x,y,p0=[peak1,peakpos1,std1])    

def lin(x,m,b):
    return [m * i + b for i in x]

pop2,pcov2 = curve_fit(lin,x,voly,p0=[0.5,0.04])

print(pop2[0],pop2[1])

#plt.axvline(x=2375 * 0.000001 + 0.00362)

#plt.scatter(x,ycorr,label='doppler - sat')
#plt.scatter(x,y)
#plt.scatter(satx,saty)
plt.plot(xcorr,ycorr,'c')
#plt.errorbar(x,y,yerr=[ue,le],fmt='o')
#plt.errorbar(satx,saty,yerr=[ue,le],fmt='o')
#plt.errorbar(xcorr,ycorr,yerr=[sue,sle],fmt='c')
#plt.scatter(x,voly)
#plt.plot(x,lin(x,*pop2))
#plt.scatter(satx,ycorr1)
#plt.plot(x,gaus(x,*pop1))
plt.show()


# In[ ]:





# In[ ]:




