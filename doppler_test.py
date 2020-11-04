#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import *
from numpy import *
import matplotlib.pyplot as plt  ##Import Libraries
import pylab

infile=open('running_list2.txt', 'r')
y_data = []
for line in infile:
    y_data.append(int(line.strip()))

infile.close()

infile=open('back_run_list.txt', 'r')
back = []
for line in infile:
    back.append(int(line.strip()))

infile.close()

def backscatterfunc(x):
    return 0.00000000000000000000020505941248622763*(x+500)**9 + -0.000000000000000000038179264314933404*(x+500)**8 + 0.000000000000000017965810941721175*(x+500)**7 +  30.34554647967693


def comp(x):
    return -3.087514420272611 * 0.00001 * x**3 +  0.055797555505111604* x**2 +-33.47016865039447 * x + 6729.604075708506

y_temp = [y - b for y,b in zip(y_data,back)]

channel=[*range(len(y_data))]
x_data = [i * 0.6457497967358761 - 36.74787316318668 for i in channel]

y_corr = [i for i in y_temp[:200]] + [i - backscatterfunc(x) for i,x in zip(y_temp[200:745],x_data[200:745])] + [y - comp(x) for y,x in zip(y_temp[745:975],x_data[745:975])] + [i for i in y_temp[975:]]
y_corr = [0 if i < 0 else i for i in y_corr]


le=[0.001] * len(y_data)

ue=[0.001] * len(y_data)

print("Max value in running list_2:",max(y_corr))
print(len(x_data),len(y_corr))

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure 
plt.figure(figsize=(50,50))
#plt.scatter(x_data,y_data)
#plt.show()





from scipy import optimize

#def test_func(x,a, b):
 #   return ((-a*np.cos(b*x))**(2)) #an approximation of the ... plot for the best fit with our data

#params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[6.6, 1.2])

#print(params)

#from scipy.stats import chisquare
#chsq = chisquare([100, 117, 97, 183, 393, 959])
#print("X^2:",chsq)

#import scipy.signal
#print('Detect peaks without any filters.')
#indexes = scipy.signal.find_peaks_cwt(y_data, np.arange(1, 4),
#    max_distances=np.arange(1, 4)*2)
#indexes = np.array(indexes) - 1
#print('Peaks are: %s' % (indexes))


#plt.scatter(x_data, y_data, label="Data")
plt.scatter(x_data, y_corr, label="Data 2")
#plt.plot(x_data, test_func(x_data, params[0], params[1]), label="Fitted Function")

from scipy.signal import find_peaks_cwt
cb = np.array(y_corr)
indexes = find_peaks_cwt(cb, np.arange(1, 550))
print('Peaks are: %s' % (indexes))
index_to_x = [i * 0.6457497967358761 - 36.74787316318668 for i in indexes]
print('Energies are: %s' % (index_to_x))

print(sum(y_corr[980:1275]))
#energypeak = 661.6
#count = 

y_gaus = y_corr[1006:1147]
domain = x_data[1006:1147]

energy0 =  658.078908124616
count0 = 681
meanx = mean(domain)

#popt,pcov = curve_fit(gaus,energy,intensity,p0=[45,mean,sigma])
std = sqrt(mean([((x - meanx)**2) for x in domain]))
def gaus(x_data,count0,energy0,std):
       return count0*exp(-(x_data-energy0)**2/(std**2))

popt,pcov = curve_fit(gaus,domain,y_gaus,p0=[681,meanx,std])




plt.legend(loc="best")
plt.errorbar(x_data,y_corr,yerr=[le,ue],fmt='o')#switch between y_data and Y_corr to put your errors bars (use green "g") to switch where your error bars are
plt.xlabel ("energy")
plt.ylabel("counts")
plt.plot(domain,gaus(domain,*popt))
plt.title("energy vs counts")
plt.show()

