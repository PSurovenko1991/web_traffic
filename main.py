import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

data_set = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
x = data_set[:,0]
y = data_set[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

def error(x,y,f):
    return(sp.sum((f(x)-y)**2))


fp1 = sp.polyfit(x,y,1)
f1 = sp.poly1d(fp1)
print('error f1: ',error(x,y,f1))

fp2 = sp.polyfit(x,y,2)
f2 = sp.poly1d(fp2)
print('error f2: ',error(x,y,f2))

fp3 = sp.polyfit(x,y,50)
f3 = sp.poly1d(fp3)
print('error f3: ',error(x,y,f3))

x1 = x[int(3.5*7*24):]

y1 = y[int(3.5*7*24):]

fp4 = sp.polyfit(x1,y1,1)
f4 = sp.poly1d(fp4)
print('error f4: ',error(x1,y1,f4))

fp5 = sp.polyfit(x1,y1,5)  # find min error(min сhange) and moment when line not come down 
a = 5 #!!!
f5 = sp.poly1d(fp5)
print('error f5: ',error(x1,y1,f5))

plt.scatter(x, y,s=9)  
plt.title("webtraffic")
plt.xlabel("hour")
plt.ylabel("hints")
plt.xticks([w*24*7 for w in range(9)], ['week %i' % w for w in range(9) ])
plt.autoscale()
plt.grid(True, linestyle = '-', color = '0.75')
fx = sp.linspace(0,x1[-1]+6,110)
fx1 = sp.linspace(int(3.5*7*24),x1[-1]+6,110)
plt.plot(fx,f1(fx), linewidth=4)
plt.plot(fx,f2(fx), linewidth=4)
plt.plot(fx,f3(fx), linewidth=4)
plt.plot(fx1,f4(fx1), linewidth=4)
plt.plot(fx1,f5(fx1), linewidth=4)

plt.show()
# greenline fill - overtrain
# min error - f5
# train f5 on full data
fp51= sp.polyfit(x,y,5)
f51 = sp.poly1d(fp51)



ans = fsolve(f51-100000,x0=800)/(7*24)

print(ans)

