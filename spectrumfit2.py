#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import *
from numpy import *
import matplotlib.pyplot as plt  ##Import Libraries
import pylab
from matplotlib.pyplot import figure 


infile=open('doppler_broad_11-18.txt', 'r')
data = []
for line in infile:
    data.append(line.strip().split())

infile.close()

infile=open('background_11-18.txt', 'r')
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
    
satx = [i for i in x]
saty = []
for line in sat:
    saty.append(line[1])

ycorr = [y - b + 0.05 for y,b in zip(y,saty)]



ycorr1 = [y - b for y,b in zip(y,saty)]
xcorr1 = satx[700:3000]


plt.figure(figsize=(50,50))
    
    
from scipy.signal import find_peaks_cwt
cb = np.array(y)
indexes = find_peaks_cwt(cb, np.arange(1, 800))
#print('Peaks are: %s' % (indexes))
index_to_x = [i * 0.0000004 + 0.02415 for i in indexes]
print('Times  are: %s' % (index_to_x))

for xc in index_to_x:
    plt.axvline(x=xc)
    
    
'''Things to remember
   Add 0.05 to ycorr to make positive
   fit gaussians to each peak'''



'''
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
peak4 = y[indexes[6]]

std1 = sqrt(mean([((x - peakpos1)**2) for x in domain1]))
std2 = sqrt(mean([((x - peakpos2)**2) for x in domain2]))
std3 = sqrt(mean([((x - peakpos3)**2) for x in domain3]))
std4 = sqrt(mean([((x - peakpos4)**2) for x in domain4]))

def gaus(x,y0,peakpos0,std0):
       return y0*exp(-(x-peakpos0)**2/(2 * std0**2))
    
pop1,pcov1 = curve_fit(gaus,domain1,y1,p0=[peak1,peakpos1,std1])
pop2,pcov2 = curve_fit(gaus,domain2,y2,p0=[peak2,peakpos2,std2])
pop3,pcov3 = curve_fit(gaus,domain3,y3,p0=[peak3,peakpos3,std3])
pop4,pcov4 = curve_fit(gaus,domain4,y4,p0=[peak4,peakpos4,std4])

print(pop1[0],pop2[0],pop3[0],pop4[0])
print(pop1[1],pop2[1],pop3[1],pop4[1])
print(pop1[2],pop2[2],pop3[2],pop4[2])
print(pop3[1] - pop2[1])
print(pop4[1] - pop1[1])














plt.plot(x[850:3000],y[850:3000],"co",ms=3,label="Doppler Broadened Spectrum")
plt.plot(domain1,gaus(domain1,*pop1),"k-",label="Rb-87 F2 peak")
plt.plot(domain2,gaus(domain2,*pop2),"r-",label="Rb-85 F3 peak")
plt.plot(domain3,gaus(domain3,*pop3),"m-",label="Rb-85 F2 peak")
plt.plot(domain4,gaus(domain4,*pop4),"purple",label="Rb-87 F1 peak")


'''




#plt.scatter(x,ycorr,label='doppler - sat')
plt.scatter(x,y)
plt.scatter(satx,saty)
plt.scatter(x,ycorr)
plt.show()


# In[ ]:




