#!/usr/bin/env python
# coding: utf-8

# In[90]:


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
    
backx = [i for i in x]
backy = []
for line in sat:
    backy.append(line[1])

ycorr = [y - b  for y,b in zip(y,backy)]


def func(x, a, b):
    return a*(np.power(x,1))+b

popt, pcov = curve_fit(func, x[3400:], y[3400:])


print("a = %s, b = %s" % (popt[0],popt[1]))

poly = [func(i, popt[0], popt[1]) for i in x]

ysub = [y - b  for y,b in zip(y,poly)]




plt.figure(figsize=(50,30))
    
   
    
    
from scipy.signal import find_peaks_cwt
cb = np.array(ysub)
indexes = find_peaks_cwt(cb, np.arange(1, 600))
print('Peaks are: %s' % (indexes))
index_to_x = [(i * 0.0000004 + 0.02415 - 0.0245712)* 10 /0.00124974011 + 0.0734241804273828 for i in indexes]
print('Times  are: %s' % (index_to_x))




#for xc in index_to_x:
#    plt.axvline(x=xc)
    


#xs = (3250* 0.0000004 + 0.02415 - 0.0245712)* 10 /0.00124974011 
#plt.axvline(x=xs)




x = [(i - 0.0245712) * 10 /.00124974011 + 0.0734241804273828  for i in x]



'''Things to remember
   Add 0.05 to ycorr to make positive
   fit gaussians to each peak'''




domain1 = x[800:1160]
domain2 = x[1200:1650]
domain3 = x[2075:2550]
domain4 = x[2870:3250]

y1 = ysub[800:1160]
y2 = ysub[1200:1650]
y3 = ysub[2075:2550]
y4 = ysub[2870:3250]

peakpos1 = 0
peakpos2 = index_to_x[2]
peakpos3 = index_to_x[3]
peakpos4 = index_to_x[4]

peak1 = ysub[indexes[1]]
peak2 = ysub[indexes[2]]
peak3 = ysub[indexes[3]]
peak4 = ysub[indexes[4]]


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

print("amplitude values:", pop1[0],pop2[0],pop3[0],pop4[0])
print("peak positions", pop1[1],pop2[1],pop3[1],pop4[1])
print("standard deviations", pop1[2],pop2[2],pop3[2],pop4[2])
print(pop3[1] - pop2[1])
print(pop4[1] - pop1[1])




fwhm1 = 2 * math.sqrt(2 * math.log(2)) * pop1[2]
fwhm2 = 2 * math.sqrt(2 * math.log(2)) * pop2[2]
fwhm3 = 2 * math.sqrt(2 * math.log(2)) * pop3[2]
fwhm4 = 2 * math.sqrt(2 * math.log(2)) * pop4[2]

print("FWHM are: {}, {}, {}, {},".format(fwhm1,fwhm2,fwhm3,fwhm4))






ue=[0.0001 * (2 ** 0.5)] * len(ysub)
le = [i for i in ue]



plt.plot(x,ysub,"co",ms=3,label="Doppler Broadened Spectrum")
plt.plot(domain1,gaus(domain1,*pop1),"k-",label="Rb-87 F2 peak")
plt.plot(domain2,gaus(domain2,*pop2),"r-",label="Rb-85 F3 peak")
plt.plot(domain3,gaus(domain3,*pop3),"m-",label="Rb-85 F2 peak")
plt.plot(domain4,gaus(domain4,*pop4),"purple",label="Rb-87 F1 peak")
plt.xlabel ("Relative Frequency (GHz)",fontsize=30)
plt.ylabel("Amplitude (Arb. Units)",fontsize=30)
plt.legend(fontsize=25)
plt.tick_params(axis='both',labelsize=25)



plt.errorbar(x,ysub,yerr=[ue,le],fmt='o')


#plt.scatter(x,ycorr,label='doppler - sat')
#plt.scatter(x,y)
#plt.scatter(backx,backy)
#plt.scatter(x,ycorr)
#plt.scatter(x, ysub)
#plt.scatter(x,ycorr1)
#plt.plot(x,func(x,*popt)) 
plt.show()


# In[ ]:




