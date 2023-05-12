import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp

L = 4
dx = 0.04
Nx = int(L/dx)
x = np.linspace(dx/2,L-dx/2,Nx)
a = 1
sigma = 0.5
dt = (sigma*dx)/a
t = 2
nt = int(t/dt)
u2 = np.zeros(Nx)
u2v = np.zeros(Nx)
u2m = np.zeros(Nx)
u2s = np.zeros(Nx)
u2r = np.zeros(Nx)
u2vr = np.zeros(Nx)
u2mr = np.zeros(Nx)
u2sr = np.zeros(Nx)
ua = np.zeros(Nx)
new_u2 = np.zeros(Nx)
new_u2v = np.zeros(Nx)
new_u2m = np.zeros(Nx)
new_u2s = np.zeros(Nx)

'''
# Heavyside Function
u2[0] = 1
u2v[0] = 1
u2m[0] = 1
u2s[0] = 1
u2r[0] = 1
u2vr[0] = 1
u2mr[0] = 1
u2sr[0] = 1
# Analytical Solution for the Heavyside Function
ua[0:int((a*t)/dx)] = 1
'''

# Sinusoidal Function
l = 1
#l = 0.25
nx = int(l/dx)  
for i in range(0,nx+1):
    u2[i] = np.sin((2*np.pi*i)/nx)
    u2v[i] = np.sin((2*np.pi*i)/nx)
    u2m[i] = np.sin((2*np.pi*i)/nx)
    u2s[i] = np.sin((2*np.pi*i)/nx)
    u2r[i] = np.sin((2*np.pi*i)/nx)
    u2vr[i] = np.sin((2*np.pi*i)/nx)
    u2mr[i] = np.sin((2*np.pi*i)/nx)
    u2sr[i] = np.sin((2*np.pi*i)/nx)
    #Analytical Solution for the Sinusoidal Function
    ua[i+int((a*t)/dx) + 1] = np.sin((2*np.pi*i)/nx)


def lim(t,u,psi):
    
    r = np.ones(Nx)
    f = np.zeros(Nx+1)
    df = np.zeros(Nx)
    
    f[1] = a * u[0]

    for i in range(2,Nx-1):
        '''with np.errstate(divide = 'ignore', invalid = 'ignore'):
            r[i] = (y[i+1]-y[i])/(y[i]-y[i-1])'''
        if((u[i] - u[i-1]) != 0):
            r[i] = (u[i+1]-u[i])/(u[i]-u[i-1])

        f[i] = a * (u[i-1] + 0.5*psi(r[i-1])*(u[i-1]-u[i-2]))
    
    f[-2] = a * (u[-3] + 0.5*(u[-3]-u[-4])) 
    f[-1] = a * (u[-2] + 0.5*(u[-2]-u[-3])) 
        
    for i in range(0,Nx):
        df[i] = f[i+1] - f[i]
    
    df[0] = 0
    df[-1] = 0
        
    return -df/dx

def SOU(r):  
    return 1

def vanleer(r):
    psi = (r+abs(r))/(1+abs(r))
    return psi

def minmod(r):
    psi = max(0,min(1,r))
    return psi

def superbee(r):
    psi = max(0,min(2*r,1),min(r,2))
    return psi
'''      
u2sol = solve_ivp(lim, [0, t], u2r, args = [SOU])
u2vsol = solve_ivp(lim, [0, t], u2vr, args = [vanleer])
u2msol = solve_ivp(lim, [0, t], u2mr, args = [minmod])
u2ssol = solve_ivp(lim, [0, t], u2sr, args = [superbee])

plt.figure(1)
plt.clf()
plt.grid()
plt.plot(x,u2vsol.y[:,-1],x,u2msol.y[:,-1],x,u2ssol.y[:,-1],x,u2sol.y[:,-1],'k',x,ua,'r');
plt.legend(['VanLeer','Minmod','Superbee','Second Order Upwind','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
#plt.title('Heavyside Function')
plt.title('Sinusoidal Function with $\lambda$ = 1')
plt.show(); 
'''

for it in range(0, nt):
    
    u2sol = solve_ivp(lim, [it*dt, (it*dt)+dt], u2r, args = [SOU])
    u2vsol = solve_ivp(lim, [it*dt, (it*dt)+dt], u2vr, args = [vanleer])
    u2msol = solve_ivp(lim, [it*dt, (it*dt)+dt], u2mr, args = [minmod])
    u2ssol = solve_ivp(lim, [it*dt, (it*dt)+dt], u2sr, args = [superbee])
    ua = np.zeros(Nx)
    ua[int((a*it*dt)/dx) + 1:nx+2+int((a*it*dt)/dx)] = np.sin((np.linspace(0,2*np.pi*nx, nx+1))/nx)


    if (it%5==0):
        plt.figure(1)
        plt.clf()
        plt.grid()
        plt.plot(x,u2vsol.y[:,-1],x,u2msol.y[:,-1],x,u2ssol.y[:,-1],x,u2sol.y[:,-1],'k',x,ua,'r');
        plt.legend(['VanLeer','Minmod','Superbee','Second Order Upwind','Analytical Solution']);
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        #plt.title('Heavyside Function')
        plt.title('Sinusoidal Function with $\lambda$ = 1')
        plt.pause(0.1)
   
    u2r = u2sol.y[:,-1]
    u2vr = u2vsol.y[:,-1]
    u2mr = u2msol.y[:,-1]
    u2sr = u2ssol.y[:,-1]


plt.show()

'''
for n in range(0,nt):   
    new_u2 = u2 + dt*lim(t,u2,SOU)
    new_u2v = u2v + dt*lim(t,u2v,vanleer)
    new_u2m = u2m + dt*lim(t,u2m, minmod)
    new_u2s = u2s + dt*lim(t,u2s,superbee)
        
    for i in range(1,Nx-1):
        u2[i] = new_u2[i]
        u2v[i] = new_u2v[i]
        u2m[i] = new_u2m[i]
        u2s[i] = new_u2s[i]  

plt.figure(2)
plt.clf()
plt.grid()
plt.plot(x,new_u2v,x,new_u2m,x,new_u2s,x,new_u2,'k',x,ua,'r');
plt.legend(['VanLeer','Minmod','Superbee','Second Order Upwind','Analytical Solution']);
plt.xlabel(r'x');
plt.ylabel(r'u');
plt.title('Heavyside Function')
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