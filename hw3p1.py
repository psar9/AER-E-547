import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
L = 4
dx = 0.04
Nx = int(L/dx)
x = np.linspace(dx/2,L-dx/2,Nx)
a = 1
sigma = 0.5
dt = (sigma*dx)/a
t = 1
nt = int(t/dt)
u1 = np.zeros(Nx)
u1e = np.zeros(Nx)
u2 = np.zeros(Nx)
u2e = np.zeros(Nx)
ua = np.zeros(Nx)
new_u1 = np.zeros(Nx)
new_u2 = np.zeros(Nx)

'''#Heavyside Function
u1[0] = 1
u2[0] = 1
u1e[0] = 1
u2e[0] = 1
# Analytical Solution for the Heavyside Function
for i in range(0,int((a*t)/dx) + 1):
    ua[i] = 1  
'''  
'''   
#Sinusoidal Function
l = 1
#l=0.25
nx = int(l/dx)  
for i in range(0,nx+1):
    u1[i] = np.sin((2*np.pi*i)/nx)
    u1e[i] = np.sin((2*np.pi*i)/nx)
    u2[i] = np.sin((2*np.pi*i)/nx)
    u2e[i] = np.sin((2*np.pi*i)/nx)
    # Analytical Solution for the Sinusoidal Function
    ua[i+int((a*t)/dx) + 1] = np.sin((2*np.pi*i)/nx)
'''

#Pulse Function
l = int(Nx/4)
r = int(Nx/2)

u1[l:r] = 1
u1e[l:r] = 1
u2[l:r] = 1
u2e[l:r] = 1


#Transmissive Analytical Solution for Pulse Function
ua[l+int((a*t)/dx):r+int((a*t)/dx)] = 1

'''    
#Periodic Analytical Solution for Pulse Function
pl = int(a*t/4)%4+l+int((a*t)/dx)
pr = int(a*t/4)%4+r+int((a*t)/dx)
if(pl<=Nx and pr>Nx):
    ua[0:pr-Nx] = 1
    ua[pl:] = 1  
elif(pl>Nx):
    ua[pl-Nx:pr-Nx] = 1 
else:
    ua[pl:pr] = 1'''

def FOU(t,u): 
   
    df = np.zeros(Nx) 
    f = np.zeros(Nx+1)
    
    for i in range(1,Nx+1):
        f[i] = a * u[i-1]
    
    
    for i in range(0,Nx):
        df[i] = f[i+1] - f[i]
        
    df[0] = 0
    df[-1] = 0
    
    return -df/dx

def SOU(t,u): 
    
    df = np.zeros(Nx)
    f = np.zeros(Nx+1)
    
    f[1] = a * u[0]
  
    for i in range(2,Nx+1):
        #f[i] = (-a/dx) * ((2-sigma)*y[i] - 0.5*(1-sigma)*(y[i] + y[i-1]) - (2-sigma)*y[i-1] + 0.5*(1-sigma)*(y[i-1] + y[i-2]))
        f[i] = a * (u[i-1] + 0.5*(1-0)*(u[i-1] - u[i-2]))
    
    
    for i in range(0,Nx):    
        df[i] = f[i+1] - f[i]
        
    df[0] = 0
    df[-1] = 0

    return -df/dx
'''   
u1sol = solve_ivp(FOU, [0, t], u1)   
u2sol = solve_ivp(SOU, [0, t], u2)

plt.figure(1)
plt.clf()
plt.grid()
plt.plot(x,u1sol.y[:,-1],x,ua,'r');
#plt.plot(x,u1sol,x,ua,'r');
plt.legend(['First Order Upwind','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show(); 

#plt.figure(2)
plt.clf()
plt.grid()
plt.plot(x,u2sol.y[:,-1],x,ua,'r');
#plt.plot(x,u2sol,x,ua,'r');
plt.legend(['Second Order Upwind','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show();

'''

'''
def rungeKutta(u, fun):
    # Count number of iterations using step size or
    # step height h
    t0 = 0
    nt = int(t/dt)
    # Iterate for number of iterations
    for i in range(0, nt):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = fun(t0, u)
        k2 = fun(t0 + 0.5 * dt, u + 0.5 * dt * k1)
        k3 = fun(t0 + 0.5 * dt, u + 0.5 * dt * k2)
        k4 = fun(t0 + dt, u + dt * k3)
 
        # Update next value of y
        u = u + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)*dt
 
        # Update next value of x
        t0 = t0 + dt
    return u

u1sol = rungeKutta(u1, FOU)
u2sol = rungeKutta(u2, SOU)
'''

for it in range(0, nt):
    
    u1sol = solve_ivp(FOU, [it*dt, (it*dt)+dt], u1)   
    u2sol = solve_ivp(SOU, [it*dt, (it*dt)+dt], u2)
    ua = np.zeros(Nx)
    ua[l+int((a*(it*dt))/dx):r+int((a*(it*dt))/dx)] = 1

    if (it%5==0):
        plt.figure(1)
        plt.clf()
        plt.grid()
        plt.plot(x,u1sol.y[:,-1],x,ua,'r');
        #plt.plot(x,u1sol,x,ua,'r');
        plt.legend(['First Order Upwind','Analytical Solution']);
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.pause(0.1)
        
        plt.figure(2)
        plt.clf()
        plt.grid()
        plt.plot(x,u2sol.y[:,-1],x,ua,'r');
        #plt.plot(x,u2sol,x,ua,'r');
        plt.legend(['Second Order Upwind','Analytical Solution']);
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.pause(0.1)
        
   
    u1 = u1sol.y[:,-1]
    u2 = u2sol.y[:,-1]

plt.show()
'''
for n in range(0,nt):   
    new_u1 = u1e + dt*FOU(t,u1e)
    new_u2 = u2e + dt*SOU(t,u2e)
        
    for i in range(1,Nx-1):
        u1e[i] = new_u1[i]
        u2e[i] = new_u2[i]

plt.figure(3)
plt.clf()
plt.grid()
plt.plot(x,new_u1,x,ua,'r');
plt.legend(['First Order Upwind','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show(); 

plt.figure(4)
plt.clf()
plt.grid()
plt.plot(x,new_u2,x,ua,'r');
plt.legend(['Second Order Upwind','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.show(); 
'''

