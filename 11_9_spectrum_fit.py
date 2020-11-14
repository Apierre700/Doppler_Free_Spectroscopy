#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import *
from numpy import *
import matplotlib.pyplot as plt  ##Import Libraries
import pylab
from matplotlib.pyplot import figure 


infile=open('11_9_spectrum.txt', 'r')
data = []
for line in infile:
    data.append(line.strip().split())

infile.close()

for line in data:
    for i in range(len(line)):
        line[i] = float(line[i])
        

x = []
y = []
for line in data:
    x.append(line[0])
    y.append(line[1])
    
x = [(i - 0.012742) * 10 / 0.00457863355 for i in x]

plt.figure(figsize=(50,20))
    
    
from scipy.signal import find_peaks_cwt
cb = np.array(y)
indexes = find_peaks_cwt(cb, np.arange(1, 400))
print('Peaks are: %s' % (indexes))
index_to_x = [(i * 0.000002 + 0.00708  - 0.012742) * 10 / 0.00457863355 for i in indexes]
print('Times  are: %s' % (index_to_x))

for xc in index_to_x:
    plt.axvline(x=xc)
    
domain1 = x[1230:1440]
domain2 = x[1490:1730]
domain3 = x[2150:2370]
domain4 = x[2730:2930]

y1 = y[1230:1440]
y2 = y[1490:1730]
y3 = y[2150:2370]
y4 = y[2730:2930]

peakpos1 = -6.5
peakpos2 = index_to_x[3]
peakpos3 = -2.48
peakpos4 = 0

peak1 = y[1343]
peak2 = y[indexes[3]]
peak3 = y[2263]
peak4 = y[indexes[5]]

std1 = sqrt(mean([((x - peakpos1)**2) for x in domain1]))
std2 = sqrt(mean([((x - peakpos2)**2) for x in domain2]))
std3 = sqrt(mean([((x - peakpos3)**2) for x in domain3]))
std4 = sqrt(mean([((x - peakpos4)**2) for x in domain4]))

def gaus(x,y0,peakpos0,std0):
       return y0*exp(-(x-peakpos0)**2/(std0**2))
    
pop1,pcov1 = curve_fit(gaus,domain1,y1,p0=[peak1,peakpos1,std1])
pop2,pcov2 = curve_fit(gaus,domain2,y2,p0=[peak2,peakpos2,std2])
pop3,pcov3 = curve_fit(gaus,domain3,y3,p0=[peak3,peakpos3,std3])
pop4,pcov4 = curve_fit(gaus,domain4,y4,p0=[peak4,peakpos4,std4])

print(pop1[0],pop2[0],pop3[0],pop4[0])
print(pop1[1],pop2[1],pop3[1],pop4[1])
print(pop3[1] - pop2[1])
print(pop4[1] - pop1[1])

'''
hline = [0.096265789]
fwhm_intersections = set.intersection(set(hline), set(y))
print("first two are the intersections we need:",fwhm_intersections)'''

print(gaus(-6.8317,*pop1))
print(gaus(-6.237,*pop1))

plt.axvline(x=-6.82)
plt.axvline(x=-6.25)

#plt.scatter(x,ycorr,label='doppler - sat')
plt.scatter(x,y)
plt.plot(domain1,gaus(domain1,*pop1))
plt.plot(domain2,gaus(domain2,*pop2))
plt.plot(domain3,gaus(domain3,*pop3))
plt.plot(domain4,gaus(domain4,*pop4))
plt.axhline(y=0.096265789)
plt.show()


# In[ ]:




