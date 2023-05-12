import numpy as np
import matplotlib.pyplot as plt
import cmath

# setting the axes projection as polar
plt.axes(projection = 'polar')
sigma = [0.3,0.6,0.9,1.2]
#sigma = [0.25,0.5,0.75,1]

# creating an array containing the radian values
theta = np.arange(0.01, (np.pi), 0.01)

d1 = np.zeros(np.size(theta))
p1 = np.zeros(np.size(theta))
d2 = np.zeros(np.size(theta))
p2 = np.zeros(np.size(theta))
d3 = np.zeros(np.size(theta))
p3 = np.zeros(np.size(theta))
d4 = np.zeros(np.size(theta))
p4 = np.zeros(np.size(theta))
d5 = np.zeros(np.size(theta))
p5 = np.zeros(np.size(theta))

# calculate r values as function of theta
for i in range(0,np.size(sigma)):
    for rad in range(0,np.size(theta)):
        d1[rad] = np.sqrt((1-sigma[i]+sigma[i]*np.cos(theta[rad]))**2+(sigma[i]*np.sin(theta[rad]))**2)
        p1[rad] = cmath.phase(complex(1-sigma[i]+sigma[i]*np.cos(theta[rad]),sigma[i]*np.sin(theta[rad])))/(sigma[i]*theta[rad])
        d2[rad] = np.sqrt(np.cos(theta[rad])**2 + sigma[i]**2*np.sin(theta[rad])**2)
        p2[rad] = cmath.phase(complex(np.cos(theta[rad]),sigma[i]*np.sin(theta[rad])))/(sigma[i]*theta[rad])
        
        d3[rad] = np.sqrt((1-1.5*sigma[i] + 0.5*sigma[i]**2 + np.cos(theta[rad])*(2*sigma[i] - sigma[i]**2) \
                    + np.cos(2*theta[rad])*0.5*(sigma[i]**2-sigma[i]))**2+(np.sin(theta[rad])*(sigma[i]**2 - 2*sigma[i])\
                    + np.sin(2*theta[rad])*(0.5*sigma[i]*(1-sigma[i])))**2)
        p3[rad] = cmath.phase(complex(1-1.5*sigma[i] + 0.5*sigma[i]**2 + np.cos(theta[rad])*(2*sigma[i] - sigma[i]**2)\
                    + np.cos(2*theta[rad])*0.5*(sigma[i]**2-sigma[i]),np.sin(theta[rad])*(sigma[i]**2 - 2*sigma[i])\
                    + np.sin(2*theta[rad])*0.5*sigma[i]*(1-sigma[i])))/(sigma[i]*theta[rad])
        
        d4[rad] = np.sqrt((1+sigma[i]**2*(np.cos(theta[rad])-1))**2+sigma[i]**2*np.sin(theta[rad])**2)
        p4[rad] = cmath.phase(complex(1+sigma[i]**2*(np.cos(theta[rad])-1),sigma[i]*np.sin(theta[rad])))/(sigma[i]*theta[rad])    
        d5[rad] = np.sqrt((1-0.5*sigma[i]+0.5*sigma[i]**3-sigma[i]**2+np.cos(theta[rad])*(sigma[i]**2+(2/3)*(sigma[i]-sigma[i]**3))\
                  -np.cos(2*theta[rad])*((1/6)*sigma[i]*(1-sigma[i]**2)))**2 \
                  + ((np.sin(theta[rad])/3)*(sigma[i]**3-4*sigma[i]) + np.sin(2*theta[rad])*((1/6)*sigma[i] - sigma[i]**3))**2)
        p5[rad] = cmath.phase(complex( 1-0.5*sigma[i]+0.5*sigma[i]**3-sigma[i]**2+np.cos(theta[rad])*(sigma[i]**2+(2/3)*(sigma[i]-sigma[i]**3))\
                  -np.cos(2*theta[rad])*((1/6)*sigma[i]*(1-sigma[i]**2)),(np.sin(theta[rad])/3)*(sigma[i]**3-4*sigma[i]) \
                  + np.sin(2*theta[rad])*((1/6)*sigma[i] - sigma[i]**3)))/(sigma[i]*theta[rad])
    
       
    plt.figure(1)
    plt.polar(theta,d1)
    plt.title(r'First Order Upwind $\epsilon_D$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(2)
    plt.polar(theta,p1)
    plt.title(r'First Order Upwind $\epsilon_{\phi}$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(3)
    plt.polar(theta,d2)
    plt.title(r'Lax Fridrichs $\epsilon_D$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(4)
    plt.polar(theta,p2)
    plt.title(r'Lax Fridrichs $\epsilon_{\phi}$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(5)
    plt.polar(theta,d3)
    plt.title(r'Second Order Upwind $\epsilon_D$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(6)
    plt.polar(theta,p3)
    plt.title(r'Second Order Upwind $\epsilon_{\phi}$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(7)
    plt.polar(theta,d4)
    plt.title(r'Lax Wendroff $\epsilon_D$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(8)
    plt.polar(theta,p4)
    plt.title(r'Lax Wendroff $\epsilon_{\phi}$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(9)
    plt.polar(theta,d5)
    plt.title(r'WKL $\epsilon_D$');
    plt.legend(['0.3','0.6','0.9','1.2'])
    plt.figure(10)
    plt.polar(theta,p5)
    plt.title(r'WKL $\epsilon_{\phi}$');
    plt.legend(['0.3','0.6','0.9','1.2'])

# display the polar plot
plt.show()
  

  




