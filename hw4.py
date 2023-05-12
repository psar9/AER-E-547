import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

L = 1
dx = 0.01
Nx = int(L/dx)
x = np.linspace(dx/2,L-dx/2,Nx)
a = 1
sigma = 0.5
dt = (sigma*dx)/a
t = 2
nt = int(t/dt)
tt = np.linspace(0,t,nt)
xp,tp = np.meshgrid(x,tt)
N = 50
ua = np.ones(Nx)
'''
# Heavyside Function
ua[0:int(Nx/2)] = -1
'''
'''
# Train of Sine Waves with irregular Amplitude
for i in range(0,Nx):
    ua[i] = 2 + 0.5*np.sin((2*np.pi*i*5)/Nx) + 0.5*np.sin((2*np.pi*i*1)/Nx)
'''

#Sinusoidal Function
l = 1
#l = 2
for i in range(0,Nx):
    ua[i] = np.sin((2*np.pi*i*l)/Nx)


def CS(t,u):
    
    df = np.zeros(Nx) 
    f = np.zeros(Nx+1)
   
    for i in range(1,Nx):
        f[i] = 0.25*(u[i-1]**2+u[i]**2)
        
    f[-1] = 0.5*u[-1]**2
    '''
    #Transmissive boundary conditions
    f[0] = 0.5*u[0]**2
    '''
    
    #Periodic Boundary Conditions
    f[0] = 0.25*(u[-2]**2+u[0]**2)
    
  
    for i in range(0,Nx):
        df[i] = f[i+1] - f[i]
    
    return -df/dx

def UP(t,u): 
   
    df = np.zeros(Nx) 
    f = np.zeros(Nx+1)
    
    for i in range(1,Nx):
        uS = 0.5*(u[i] + u[i-1])
        if(uS>0):   
            f[i] = 0.5*u[i-1]**2
        else:
            f[i] = 0.5*u[i]**2
    f[-1] = 0.5*u[-1]**2
    '''
    #Transmissive boundary conditions
    f[0] = 0.5*u[0]**2
    '''
    
    #Periodic Boundary Conditions
    uS = 0.5*(u[0] + u[-2])
    if(uS>0):   
        f[0] = 0.5*u[-2]**2
    else:
        f[0] = 0.5*u[0]**2
    
    
    for i in range(0,Nx):
        df[i] = f[i+1] - f[i]
    
    return -df/dx

def FOG(t,u):
    df = np.zeros(Nx) 
    f = np.zeros(Nx+1)
    
    for i in range(1,Nx):
        uL = u[i-1]
        uR = u[i]
        if(uL>=uR):
            f[i] = 0.5*max(uL**2,uR**2)
        elif(uL<0 and uR>0):
            f[i] = 0
        else:
            f[i] = 0.5*min(uL**2,uR**2)
    f[-1] = 0.5*u[-1]**2
    '''
    #Transmissive boundary conditions
    f[0] = 0.5*u[0]**2
    '''
    
    #Periodic Boundary Conditions
    uL = u[-2]
    uR = u[0]
    if(uL>=uR):
        f[0] = 0.5*max(uL**2,uR**2)
    elif(uL<0 and uR>0):
        f[0] = 0
    else:
        f[0] = 0.5*min(uL**2,uR**2)
    
    for i in range(0,Nx):
        df[i] = f[i+1] - f[i]
    
    return -df/dx
   
def SOG(t,u):
    df = np.zeros(Nx)
    f = np.zeros(Nx+1)
 
    for i in range(2,Nx-1):
        uL = u[i-1] + 0.5*(u[i-1] - u[i-2])
        uR = u[i] - 0.5*(u[i+1] - u[i])
        if(uL>=uR):
            f[i] = 0.5*max(uL**2,uR**2)
        elif(uL<0 and uR>0):
            f[i] = 0
        else:
            f[i] = 0.5*min(uL**2,uR**2)   
    
    uL = u[0]
    uR = u[1]
    if(uL>=uR):
        f[1] = 0.5*max(uL**2,uR**2)
    elif(uL<0 and uR>0):
        f[1] = 0
    else:
        f[1] = 0.5*min(uL**2,uR**2)
       
    uL = u[-2]
    uR = u[-1]
    if(uL>=uR):
        f[-2] = 0.5*max(uL**2,uR**2)
    elif(uL<0 and uR>0):
        f[-2] = 0
    else:
        f[-2] = 0.5*min(uL**2,uR**2)
        
    f[-1] = 0.5*u[-1]**2
    '''
    #Transmissive boundary conditions
    f[0] = 0.5*u[0]**2   
    '''
    
    #Periodic Boundary Conditions
    uL = u[-2] + 0.5*(u[-2] - u[-3])
    uR = u[0] - 0.5*(u[1] - u[0])
    if(uL>=uR):
        f[0] = 0.5*max(uL**2,uR**2)
    elif(uL<0 and uR>0):
        f[0] = 0
    else:
        f[0] = 0.5*min(uL**2,uR**2) 
    

    for i in range(0,Nx):    
        df[i] = f[i+1] - f[i]

    return -df/dx
  
def SOGL(t,u,psi):
    rL = np.ones(Nx)
    rR = np.ones(Nx)
    df = np.zeros(Nx)
    f = np.zeros(Nx+1) 

    for i in range(1,Nx-1):
        if((u[i] - u[i-1]) != 0):
            rL[i] = (u[i+1]-u[i])/(u[i]-u[i-1])
        if((u[i] - u[i+1]) != 0):   
            rR[i] = (u[i-1]-u[i])/(u[i]-u[i+1])
            
    for i in range(2,Nx-1):
        uL = u[i-1] + psi(rL[i-1])*0.5*(u[i-1] - u[i-2])
        uR = u[i] + psi(rR[i])*0.5*(u[i] - u[i+1])
        if(uL>=uR):
            f[i] = 0.5*max(uL**2,uR**2)
        elif(uL<0 and uR>0):
            f[i] = 0
        else:
            f[i] = 0.5*min(uL**2,uR**2)   
    
    uL = u[0]
    uR = u[1]
    if(uL>=uR):
        f[1] = 0.5*max(uL**2,uR**2)
    elif(uL<0 and uR>0):
        f[1] = 0
    else:
        f[1] = 0.5*min(uL**2,uR**2)
       
    uL = u[-2]
    uR = u[-1]
    if(uL>=uR):
        f[-2] = 0.5*max(uL**2,uR**2)
    elif(uL<0 and uR>0):
        f[-2] = 0
    else:
        f[-2] = 0.5*min(uL**2,uR**2)
        
    f[-1] = 0.5*u[-1]**2
    '''
    #Transmissive boundary conditions
    f[0] = 0.5*u[0]**2   
    '''
    
    #Periodic Boundary Conditions
    uL = u[-2] + 0.5*(u[-2] - u[-3])
    uR = u[0] - 0.5*(u[1] - u[0])
    if(uL>=uR):
        f[0] = 0.5*max(uL**2,uR**2)
    elif(uL<0 and uR>0):
        f[0] = 0
    else:
        f[0] = 0.5*min(uL**2,uR**2) 
    
    
    for i in range(0,Nx):    
        df[i] = f[i+1] - f[i]

    return -df/dx

def vanleer(r):
    psi = (r+abs(r))/(1+abs(r))
    return psi

def minmod(r):
    psi = max(0,min(1,r))
    return psi

def superbee(r):
    psi = max(0,min(2*r,1),min(r,2))
    return psi

def ROE1(t,u):
    df = np.zeros(Nx) 
    f = np.zeros(Nx+1)
      
    for i in range(1,Nx):
        uL = u[i-1]
        uR = u[i]
        dfdu = 0.5*(uL + uR)
        eps = max(0,0.5*(uR - uL))
        if(abs(dfdu) >= eps):
            dfdu = dfdu
        else:
            dfdu = eps
        f[i] = 0.25*(uL**2+uR**2) - 0.5*abs(dfdu)*(uR - uL)
    f[-1] = 0.5*u[-1]**2
    '''
    #Transmissive boundary conditions
    f[0] = 0.5*u[0]**2
    '''
    
    #Periodic Boundary Conditions
    uL = u[-2]
    uR = u[0]
    dfdu = 0.5*(uL + uR)
    eps = max(0,0.5*(uR - uL))
    if(abs(dfdu) >= eps):
        dfdu = dfdu
    else:
        dfdu = eps
    f[0] = 0.25*(uL**2+uR**2) - 0.5*abs(dfdu)*(uR - uL)
    
    for i in range(0,Nx):    
        df[i] = f[i+1] - f[i]

    return -df/dx    

def ROE2(t,u):
    df = np.zeros(Nx) 
    f = np.zeros(Nx+1)
    
    for i in range(2,Nx-1):
        uL = u[i-1] + 0.5*(u[i-1] - u[i-2])
        uR = u[i] - 0.5*(u[i+1] - u[i])
        dfdu = 0.5*(uL + uR)
        eps = max(0,0.5*(uR - uL))
        if(abs(dfdu) >= eps):
            dfdu = dfdu
        else:
            dfdu = eps
        f[i] = 0.25*(uL**2+uR**2) - 0.5*abs(dfdu)*(uR - uL)
    
    uL = u[0]
    uR = u[1]
    dfdu = 0.5*(uL + uR)
    eps = max(0,0.5*(uR - uL))
    if(abs(dfdu) >= eps):
        dfdu = dfdu
    else:
        dfdu = eps
    f[1] = 0.25*(uL**2+uR**2) - 0.5*abs(dfdu)*(uR - uL)
    
    uL = u[-2]
    uR = u[-1]
    dfdu = 0.5*(uL + uR)
    eps = max(0,0.5*(uR - uL))
    if(abs(dfdu) >= eps):
        dfdu = dfdu
    else:
        dfdu = eps
    f[-2] = 0.25*(uL**2+uR**2) - 0.5*abs(dfdu)*(uR - uL)
    
    f[-1] = 0.5*u[-1]**2
    '''
    #Transmissive boundary conditions
    f[0] = 0.5*u[0]**2
    '''
    
    #Periodic Boundary Conditions
    uL = u[-2] + 0.5*(u[-2] - u[-3])
    uR = u[0] - 0.5*(u[1] - u[0])
    dfdu = 0.5*(uL + uR)
    eps = max(0,0.5*(uR - uL))
    if(abs(dfdu) >= eps):
        dfdu = dfdu
    else:
        dfdu = eps
    f[0] = 0.25*(uL**2+uR**2) - 0.5*abs(dfdu)*(uR - uL)
    
    for i in range(0,Nx):    
        df[i] = f[i+1] - f[i]

    return -df/dx   


u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):
    
    usol = solve_ivp(CS, [it*dt, (it*dt)+dt], u) 
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):
        plt.figure(1)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Central Scheme for time = %1.3f' %t)
        #plt.pause(0.2)
   
    u = usol.y[:,-1]
plt.show()

plt.figure(2)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
plt.title('Central  Scheme for time = %1.3f' %t)

u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):
      
    usol = solve_ivp(UP, [it*dt, (it*dt)+dt], u)
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):

        plt.figure(3)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Upwind Scheme for time = %1.3f' %t)
        #plt.pause(0.2)

    u = usol.y[:,-1]
plt.show()

plt.figure(4)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r't');
plt.ylabel(r'x');
plt.title('Upwind Scheme for time = %1.3f' %t)

u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):

    usol = solve_ivp(FOG, [it*dt, (it*dt)+dt], u)
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):

        plt.figure(5)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('First Order Godunov for time = %1.3f' %t)
        #plt.pause(0.2)

    u = usol.y[:,-1]
plt.show()

plt.figure(6)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r't');
plt.ylabel(r'x');
plt.title('First Order Godunov for time = %1.3f' %t)

u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):

    usol = solve_ivp(SOG, [it*dt, (it*dt)+dt], u)
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):

        plt.figure(7)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Second Order Godunov for time = %1.3f' %t)
        #plt.pause(0.2)

    u = usol.y[:,-1]
plt.show()

plt.figure(8)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r't');
plt.ylabel(r'x');
plt.title('Second Order Godunov for time = %1.3f' %t)

u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):

    usol = solve_ivp(SOGL, [it*dt, (it*dt)+dt], u, args = [vanleer])
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):

        plt.figure(9)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Second Order Godunov with van Leer Limiter for time = %1.3f' %t)
        #plt.pause(0.2)

    u = usol.y[:,-1]
plt.show()

plt.figure(10)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r't');
plt.ylabel(r'x');
plt.title('Second Order Godunov with van Leer Limiter for time = %1.3f' %t)

u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):

    usol = solve_ivp(SOGL, [it*dt, (it*dt)+dt], u, args = [minmod])
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):

        plt.figure(11)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Second Order Godunov with Minmod Limiter for time = %1.3f' %t)
        #plt.pause(0.2)

    u = usol.y[:,-1]
plt.show()

plt.figure(12)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r't');
plt.ylabel(r'x');
plt.title('Second Order Godunov with Minmod Limiter for time = %1.3f' %t)

u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):

    usol = solve_ivp(SOGL, [it*dt, (it*dt)+dt], u, args = [superbee])
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):

        plt.figure(13)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Second Order Godunov with Superbee Limiter for time = %1.3f' %t)
        #plt.pause(0.2)

    u = usol.y[:,-1]
plt.show()

plt.figure(14)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r't');
plt.ylabel(r'x');
plt.title('Second Order Godunov with Superbee Limiter for time = %1.3f' %t)

u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):

    usol = solve_ivp(ROE1, [it*dt, (it*dt)+dt], u)
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):

        plt.figure(15)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Roe First Order Scheme time = %1.3f' %t)
        #plt.pause(0.2)

    u = usol.y[:,-1]
plt.show()

plt.figure(16)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r't');
plt.ylabel(r'x');
plt.title('Roe First Order Scheme for time = %1.3f' %t)

u = ua
uc = np.zeros((Nx, nt))

for it in range(0, nt):

    usol = solve_ivp(ROE2, [it*dt, (it*dt)+dt], u)
    uc[:,it] = usol.y[:,-1]
    if (it%5==0):

        plt.figure(17)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Roe Second Order Scheme time = %1.3f' %t)
        #plt.pause(0.2)

    u = usol.y[:,-1]
plt.show()

plt.figure(18)
plt.contourf(xp, tp, np.transpose(uc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r't');
plt.ylabel(r'x');
plt.title('Roe Second Order Scheme for time = %1.3f' %t)

'''
import matplotlib.animation as ani
u = ua
frames=nt-2
fig = plt.figure()
#ax = plt.gca()
z = 0
def scatter_ani(i=int):
    global z
    global u
    usol = solve_ivp(ROE1, [z*dt, (z*dt)+dt], u)   
    #u1c[:,z] = u1sol.y[:,-1]
    if (z%5==0):
        plt.figure(1)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,usol.y[:,-1],x,ua,'r');
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('First Order Roe Scheme for time = %1.3f' %t)
        #plt.pause(0.2)
       
    u = usol.y[:,-1]
    z = z+1

anim = ani.FuncAnimation(fig, scatter_ani, frames=frames, interval=200)
anim.save("ROEtest.gif")
'''