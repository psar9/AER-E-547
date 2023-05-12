import numpy as np
import matplotlib.pylab as plt

L = 4
dx = 0.04
Nx = int(L/dx)+1
x = np.linspace(0,L,Nx)
a = 1
sigma = 0.5
dt = (sigma*dx)/a
t = 2
nt = int(t/dt)
u1 = np.zeros(Nx)
u2 = np.zeros(Nx)
u3 = np.zeros(Nx)
u4 = np.zeros(Nx)
u5 = np.zeros(Nx)
new_u1 = np.zeros(Nx)
new_u2 = np.zeros(Nx)
new_u3 = np.zeros(Nx)
new_u4 = np.zeros(Nx)
new_u5 = np.zeros(Nx)
u = np.zeros(Nx)

u1[0] = 1
u2[0] = 1
u3[0] = 1
u4[0] = 1
u5[0] = 1

for i in range(0,int((a*t)/dx) + 1):
    u[i] = 1   

'''
l = 1
#l=0.25
nx = int(l/dx)  
for i in range(0,nx+1):
    u1[i] = np.sin((2*np.pi*i)/nx)
    u2[i] = np.sin((2*np.pi*i)/nx)
    u3[i] = np.sin((2*np.pi*i)/nx)
    u4[i] = np.sin((2*np.pi*i)/nx)
    u5[i] = np.sin((2*np.pi*i)/nx)
    u[i+int((a*t)/dx) + 1] = np.sin((2*np.pi*i)/nx)'''


for n in range(0,nt):
    
    for i in range(1,Nx-1):
        new_u1[i] = u1[i] - sigma*(u1[i] - u1[i-1])
        
    for i in range(1,Nx-1):
        new_u2[i] = 0.5*(u2[i+1] + u2[i-1]) - 0.5*sigma*(u2[i+1] - u2[i-1])
    
    new_u3[1] = u3[1] - sigma*(u3[1] - u3[0])
    for i in range(2,Nx-1):
        new_u3[i] = u3[i] - sigma*(2-sigma)*(u3[i] - u3[i-1]) + 0.5*sigma*(1-sigma)*(u3[i] - u3[i-2])
        
    for i in range(1,Nx-1):
        new_u4[i] = u4[i] - 0.5*sigma*(u4[i+1] - u4[i-1]) + 0.5*sigma**2*(u4[i+1] - 2*u4[i] + u4[i-1])
    
    new_u5[1] = u5[1] - 0.5*sigma*(u5[2] - u5[0]) + 0.5*sigma**2*(u5[2] - 2*u5[1] + u5[0])
    for i in range(2,Nx-1):
        new_u5[i] = u5[i] - 0.5*sigma*(u5[i+1] - u5[i-1]) + 0.5*sigma**2*(u5[i+1] - 2*u5[i] + u5[i-1])\
            + (1/6)*sigma*(1-sigma**2)*(-u5[i-2] + 3*u5[i-1] - 3*u5[i] + u5[i+1])
    
    for i in range(1,Nx-1):
        u1[i] = new_u1[i]
        u2[i] = new_u2[i]
        u3[i] = new_u3[i]
        u4[i] = new_u4[i]
        u5[i] = new_u5[i]
        
plt.figure(1)
plt.clf()
plt.grid()
plt.plot(x,new_u1,x,u,'r');
plt.legend(['First Order Upwind','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show(); 

plt.figure(2)
plt.clf()
plt.grid()
plt.plot(x,new_u2,x,u,'r');
plt.legend(['Lax -Friedrichs','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show(); 

plt.figure(3)
plt.clf()
plt.grid()
plt.plot(x,new_u3,x,u,'r');
plt.legend(['Second-order Upwind','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show(); 

plt.figure(4)
plt.clf()
plt.grid()
plt.plot(x,new_u4,x,u,'r');
plt.legend(['Lax-Wendroff','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show(); 

plt.figure(5)
plt.clf()
plt.grid()
plt.plot(x,new_u5,x,u,'r');
plt.legend(['Third order WKL','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show(); 

