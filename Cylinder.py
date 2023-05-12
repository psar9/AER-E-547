import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

L = 2
Nx = 100
Ny = 100
dx = L/Nx
dy = L/Ny
x = np.linspace(dx/2,L-dx/2,Nx)
y = np.linspace(dy/2,L-dy/2,Ny)
a = 1
g = 1.4
R = 0.4
sigma = 0.5
dt = (sigma*dx)/a
t = 0.4
nt = int(t/dt)
velx = np.zeros((Nx,Ny))
vely = np.zeros((Nx,Ny))
rho = np.zeros((Nx,Ny))
pr = np.zeros((Nx,Ny))
Et = np.zeros((Nx,Ny)) 
QC = np.zeros(4*Nx*Ny)
QI = np.zeros(4*Nx*Ny)

for i in range(0,Nx):
    for j in range(0,Ny):
        if(np.sqrt((x[i]-0.5*L)**2 + (y[j]-0.5*L)**2) <= R):
            rho[i,j] = 1.0
            velx[i,j] = 0.0
            vely[i,j] = 0.0
            pr[i,j] = 1.0    
        else:
            rho[i,j] = 0.125
            velx[i,j] = 0.0
            vely[i,j] = 0.0
            pr[i,j] = 0.1
    
        QI[(Ny*i)+j] = rho[i,j]
        QI[(Nx*Ny)+(Ny*i)+j] = rho[i,j]*velx[i,j]
        QI[(2*Nx*Ny)+(Ny*i)+j] = rho[i,j]*vely[i,j]
        QI[(3*Nx*Ny)+(Ny*i)+j] = rho[i,j]*(pr[i,j]/(rho[i,j]*(g-1)) + 0.5*velx[i,j]**2 + 0.5*vely[i,j]**2)

def Et(rho, u, v, p):
    return ((p/(rho*(g-1))) + 0.5*u**2 + 0.5*v**2)

def Ht(rho, u, v, p):
    return (((g*p)/(rho*(g-1))) + 0.5*u**2 + 0.5*v**2)

def pres(q1, q2, q3, q4):
    return (g-1)*(q4 - ((q2**2+q3**2)/(2*q1)))

def QtoU(Q):
    [q1, q2, q3, q4] = Q
    return np.array([q1, q2/q1, q3/q1, (q4 - ((q2**2+q3**2)/(2*q1)))*(g-1)])

def UtoQ(U):
    [r,u,v,p] = U
    return np.array([r, r*u, r*v, r*((p/(r*(g-1))) + 0.5*u**2 + 0.5*v**2)])

def FluxF(U):
    [r,u,v,p] = U
    return np.array([r*u, r*u**2 + p, r*u*v, (r*Et(r,u,v,p) + p)*u])

def FluxG(U):
    [r,u,v,p] = U
    return np.array([r*u, r*u*v, r*v**2 + p, (r*Et(r,u,v,p) + p)*v])

def SOU(t,QC,schx,schy,lim):
  
    f1 = np.zeros((Nx+1,Ny))
    f2 = np.zeros((Nx+1,Ny))
    f3 = np.zeros((Nx+1,Ny))
    f4 = np.zeros((Nx+1,Ny))
    g1 = np.zeros((Nx,Ny+1))
    g2 = np.zeros((Nx,Ny+1))
    g3 = np.zeros((Nx,Ny+1))
    g4 = np.zeros((Nx,Ny+1))
    U1 = np.zeros((Nx,Ny))
    U2 = np.zeros((Nx,Ny))
    U3 = np.zeros((Nx,Ny))
    U4 = np.zeros((Nx,Ny))
    df = np.zeros(4*Nx*Ny)
    dg = np.zeros(4*Nx*Ny)
    rL1 = np.ones((Nx+1,Ny+1))
    rR1 = np.ones((Nx+1,Ny+1))
    rL2 = np.ones((Nx+1,Ny+1))
    rR2 = np.ones((Nx+1,Ny+1))
    rL3 = np.ones((Nx+1,Ny+1))
    rR3 = np.ones((Nx+1,Ny+1))
    rL4 = np.ones((Nx+1,Ny+1))
    rR4 = np.ones((Nx+1,Ny+1))
    rb1 = np.ones((Nx+1,Ny+1))
    rt1 = np.ones((Nx+1,Ny+1))
    rb2 = np.ones((Nx+1,Ny+1))
    rt2 = np.ones((Nx+1,Ny+1))
    rb3 = np.ones((Nx+1,Ny+1))
    rt3 = np.ones((Nx+1,Ny+1))
    rb4 = np.ones((Nx+1,Ny+1))
    rt4 = np.ones((Nx+1,Ny+1))
            
    for i in range(0,Nx):
        for j in range(0,Ny):
 
            U1[i,j] = QtoU([QC[(Ny*i)+j],QC[(Nx*Ny)+(Ny*i)+j],QC[(2*Nx*Ny)+(Ny*i)+j],QC[(3*Nx*Ny)+(Ny*i)+j]])[0]
            U2[i,j] = QtoU([QC[(Ny*i)+j],QC[(Nx*Ny)+(Ny*i)+j],QC[(2*Nx*Ny)+(Ny*i)+j],QC[(3*Nx*Ny)+(Ny*i)+j]])[1]
            U3[i,j] = QtoU([QC[(Ny*i)+j],QC[(Nx*Ny)+(Ny*i)+j],QC[(2*Nx*Ny)+(Ny*i)+j],QC[(3*Nx*Ny)+(Ny*i)+j]])[2]
            U4[i,j] = QtoU([QC[(Ny*i)+j],QC[(Nx*Ny)+(Ny*i)+j],QC[(2*Nx*Ny)+(Ny*i)+j],QC[(3*Nx*Ny)+(Ny*i)+j]])[3]
           
    for i in range(2,Nx-1):
        for j in range(0,Ny):
        
            if((U1[i-1,j] - U1[i-2,j]) != 0):
                rL1[i,j] = (U1[i,j]-U1[i-1,j])/(U1[i-1,j]-U1[i-2,j])
            if((U1[i,j] - U1[i+1,j]) != 0):   
                rR1[i,j] = (U1[i-1,j]-U1[i,j])/(U1[i,j]-U1[i+1,j])
            if((U2[i-1,j] - U2[i-2,j]) != 0):
                rL2[i,j] = (U2[i,j]-U2[i-1,j])/(U2[i-1,j]-U2[i-2,j])
            if((U2[i,j] - U2[i+1,j]) != 0):   
                rR2[i,j] = (U2[i-1,j]-U2[i,j])/(U2[i,j]-U2[i+1,j])
            if((U3[i-1,j] - U3[i-2,j]) != 0):
                rL3[i,j] = (U3[i,j]-U3[i-1,j])/(U3[i-1,j]-U3[i-2,j])
            if((U3[i,j] - U3[i+1,j]) != 0):   
                rR3[i,j] = (U3[i-1,j]-U3[i,j])/(U3[i,j]-U3[i+1,j]) 
            if((U4[i-1,j] - U4[i-2,j]) != 0):
                rL4[i,j] = (U4[i,j]-U4[i-1,j])/(U4[i-1,j]-U4[i-2,j])
            if((U4[i,j] - U4[i+1,j]) != 0):   
                rR4[i,j] = (U4[i-1,j]-U4[i,j])/(U4[i,j]-U4[i+1,j]) 
        
    for i in range(2,Nx-1):
        for j in range(0,Ny):
            
            Ul1 = U1[i-1,j] + lim(rL1[i,j])*0.5*(U1[i-1,j] - U1[i-2,j])
            Ur1 = U1[i,j] + lim(rR1[i,j])*0.5*(U1[i,j] - U1[i+1,j])
            Ul2 = U2[i-1,j] + lim(rL2[i,j])*0.5*(U2[i-1,j] - U2[i-2,j])
            Ur2 = U2[i,j] + lim(rR2[i,j])*0.5*(U2[i,j] - U2[i+1,j])
            Ul3 = U3[i-1,j] + lim(rL3[i,j])*0.5*(U3[i-1,j] - U3[i-2,j])
            Ur3 = U3[i,j] + lim(rR3[i,j])*0.5*(U3[i,j] - U3[i+1,j])
            Ul4 = U4[i-1,j] + lim(rL4[i,j])*0.5*(U4[i-1,j] - U4[i-2,j])
            Ur4 = U4[i,j] + lim(rR4[i,j])*0.5*(U4[i,j] - U4[i+1,j])
            
            f1[i,j] = FSx(np.array([Ul1,Ul2,Ul3,Ul4]),np.array([Ur1,Ur2,Ur3,Ur4]),schx)[0]
            f2[i,j] = FSx(np.array([Ul1,Ul2,Ul3,Ul4]),np.array([Ur1,Ur2,Ur3,Ur4]),schx)[1]
            f3[i,j] = FSx(np.array([Ul1,Ul2,Ul3,Ul4]),np.array([Ur1,Ur2,Ur3,Ur4]),schx)[2]
            f4[i,j] = FSx(np.array([Ul1,Ul2,Ul3,Ul4]),np.array([Ur1,Ur2,Ur3,Ur4]),schx)[3]
  
    for j in range(0,Ny):
        
        f1[1,j] = FSx(np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),np.array([U1[1,j],U2[1,j],U3[1,j],U4[1,j]]),schx)[0]
        f2[1,j] = FSx(np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),np.array([U1[1,j],U2[1,j],U3[1,j],U4[1,j]]),schx)[1]
        f3[1,j] = FSx(np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),np.array([U1[1,j],U2[1,j],U3[1,j],U4[1,j]]),schx)[2]
        f4[1,j] = FSx(np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),np.array([U1[1,j],U2[1,j],U3[1,j],U4[1,j]]),schx)[3]
        
        f1[-2,j] = FSx(np.array([U1[-2,j],U2[-2,j],U3[-2,j],U4[-2,j]]),np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),schx)[0]
        f2[-2,j] = FSx(np.array([U1[-2,j],U2[-2,j],U3[-2,j],U4[-2,j]]),np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),schx)[1]
        f3[-2,j] = FSx(np.array([U1[-2,j],U2[-2,j],U3[-2,j],U4[-2,j]]),np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),schx)[2]
        f4[-2,j] = FSx(np.array([U1[-2,j],U2[-2,j],U3[-2,j],U4[-2,j]]),np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),schx)[3]
    '''     
    # Extrapolation Boundary Conditions
    f1[0,:] = f1[1,:]
    f1[-1,:] = f1[-2,:]
    f2[0,:] = f2[1,:]
    f2[-1,:] = f2[-2,:]
    f3[0,:] = f3[1,:]
    f3[-1,:] = f3[-2,:]
    f4[0,:] = f4[1,:]
    f4[-1,:] = f4[-2,:]
    '''
    #Transmissive Boundary Conditions
    for j in range(0,Ny):
        
        f1[0,j] = FSx(np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),schx)[0]
        f2[0,j] = FSx(np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),schx)[1]
        f3[0,j] = FSx(np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),schx)[2]
        f4[0,j] = FSx(np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),np.array([U1[0,j],U2[0,j],U3[0,j],U4[0,j]]),schx)[3]
        
        f1[-1,j] = FSx(np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),schx)[0]
        f2[-1,j] = FSx(np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),schx)[1]
        f3[-1,j] = FSx(np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),schx)[2]
        f4[-1,j] = FSx(np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),np.array([U1[-1,j],U2[-1,j],U3[-1,j],U4[-1,j]]),schx)[3]
    
    '''
    # Wall Boundary Conditions
    f1[0,:] = 0
    f1[-1,:] = 0
    f2[0,:] = U4[0,:]
    f2[-1,:] = U4[-1,:]
    f3[0,:] = 0
    f3[-1,:] = 0
    f4[0,:] = 0
    f4[-1,:] = 0
    '''
    
    for i in range(0,Nx):
        for j in range(2,Ny-1):
            
            if((U1[i,j-1] - U1[i,j-2]) != 0):
                rb1[i,j] = (U1[i,j]-U1[i,j-1])/(U1[i,j-1]-U1[i,j-2])
            if((U1[i,j] - U1[i,j+1]) != 0):   
                rt1[i,j] = (U1[i,j-1]-U1[i,j])/(U1[i,j]-U1[i,j+1])
            if((U2[i,j-1] - U2[i,j-2]) != 0):
                rb2[i,j] = (U2[i,j]-U2[i,j-1])/(U2[i,j-1]-U2[i,j-2])
            if((U2[i,j] - U2[i,j+1]) != 0):   
                rt2[i,j] = (U2[i,j-1]-U2[i,j])/(U2[i,j]-U2[i,j+1])
            if((U3[i,j-1] - U3[i,j-2]) != 0):
                rb3[i,j] = (U3[i,j]-U3[i,j-1])/(U3[i,j-1]-U3[i,j-2])
            if((U3[i,j] - U3[i,j+1]) != 0):   
                rt3[i,j] = (U3[i,j-1]-U3[i,j])/(U3[i,j]-U3[i,j+1]) 
            if((U4[i,j-1] - U4[i,j-2]) != 0):
                rb4[i,j] = (U4[i,j]-U4[i,j-1])/(U4[i,j-1]-U4[i,j-2])
            if((U4[i,j] - U4[i,j+1]) != 0):   
                rt4[i,j] = (U4[i,j-1]-U4[i,j])/(U4[i,j]-U4[i,j+1]) 
    
    for i in range(0,Nx):
        for j in range(2,Ny-1):    
            Ub1 = U1[i,j-1] + lim(rb1[i,j])*0.5*(U1[i,j-1] - U1[i,j-2])
            Ut1 = U1[i,j] + lim(rt1[i,j])*0.5*(U1[i,j] - U1[i,j+1])
            Ub2 = U2[i,j-1] + lim(rb2[i,j])*0.5*(U2[i,j-1] - U2[i,j-2])
            Ut2 = U2[i,j] + lim(rt2[i,j])*0.5*(U2[i,j] - U2[i,j+1])
            Ub3 = U3[i,j-1] + lim(rb3[i,j])*0.5*(U3[i,j-1] - U3[i,j-2])
            Ut3 = U3[i,j] + lim(rt3[i,j])*0.5*(U3[i,j] - U3[i,j+1])
            Ub4 = U4[i,j-1] + lim(rb4[i,j])*0.5*(U4[i,j-1] - U4[i,j-2])
            Ut4 = U4[i,j] + lim(rt4[i,j])*0.5*(U4[i,j] - U4[i,j+1])
            
            g1[i,j] = FSy(np.array([Ub1,Ub2,Ub3,Ub4]),np.array([Ut1,Ut2,Ut3,Ut4]),schy)[0]
            g2[i,j] = FSy(np.array([Ub1,Ub2,Ub3,Ub4]),np.array([Ut1,Ut2,Ut3,Ut4]),schy)[1]
            g3[i,j] = FSy(np.array([Ub1,Ub2,Ub3,Ub4]),np.array([Ut1,Ut2,Ut3,Ut4]),schy)[2]
            g4[i,j] = FSy(np.array([Ub1,Ub2,Ub3,Ub4]),np.array([Ut1,Ut2,Ut3,Ut4]),schy)[3]
    
    for i in range(0,Nx):
        
        g1[i,1] = FSy(np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),np.array([U1[i,1],U2[i,1],U3[i,1],U4[i,1]]),schy)[0]
        g2[i,1] = FSy(np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),np.array([U1[i,1],U2[i,1],U3[i,1],U4[i,1]]),schy)[1]
        g3[i,1] = FSy(np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),np.array([U1[i,1],U2[i,1],U3[i,1],U4[i,1]]),schy)[2]
        g4[i,1] = FSy(np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),np.array([U1[i,1],U2[i,1],U3[i,1],U4[i,1]]),schy)[3]
        
        g1[i,-2] = FSy(np.array([U1[i,-2],U2[i,-2],U3[i,-2],U4[i,-2]]),np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),schy)[0]
        g2[i,-2] = FSy(np.array([U1[i,-2],U2[i,-2],U3[i,-2],U4[i,-2]]),np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),schy)[1]
        g3[i,-2] = FSy(np.array([U1[i,-2],U2[i,-2],U3[i,-2],U4[i,-2]]),np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),schy)[2]
        g4[i,-2] = FSy(np.array([U1[i,-2],U2[i,-2],U3[i,-2],U4[i,-2]]),np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),schy)[3]
    '''
    # Extrapolation Boundary Conditions
    g1[:,0] = g1[:,1]
    g1[:,-1] = g1[:,-2]
    g2[:,0] = g2[:,1]
    g2[:,-1] = g2[:,-2]
    g3[:,0] = g3[:,1]
    g3[:,-1] = g3[:,-2]
    g4[:,0] = g4[:,1]
    g4[:,-1] = g4[:,-2]
    
    '''
    #Transmissive Boundary Conditions
    for i in range(0,Nx):
        
        g1[i,0] = FSy(np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),schy)[0]
        g2[i,0] = FSy(np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),schy)[1]
        g3[i,0] = FSy(np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),schy)[2]
        g4[i,0] = FSy(np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),np.array([U1[i,0],U2[i,0],U3[i,0],U4[i,0]]),schy)[3]
        
        g1[i,-1] = FSy(np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),schy)[0]
        g2[i,-1] = FSy(np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),schy)[1]
        g3[i,-1] = FSy(np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),schy)[2]
        g4[i,-1] = FSy(np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),np.array([U1[i,-1],U2[i,-1],U3[i,-1],U4[i,-1]]),schy)[3]
       
    '''
    # Wall Boundary Conditions
    g1[:,0] = 0
    g1[:,-1] = 0
    g2[:,0] = 0
    g2[:,-1] = 0
    g3[:,0] = U4[:,0]
    g3[:,-1] = U4[:,-1]
    g4[:,0] = 0
    g4[:,-1] = 0
    '''
    
    for i in range(1,Nx-1):
        for i in range(0,Ny):
  
            df[(Ny*i)+j]= f1[i+1,j] - f1[i,j]
            df[(Nx*Ny)+(Ny*i)+j] = f2[i+1,j] - f2[i,j]
            df[(2*Nx*Ny)+(Ny*i)+j] = f3[i+1,j] - f3[i,j]
            df[(3*Nx*Ny)+(Ny*i)+j] = f4[i+1,j] - f4[i,j]
        
    for i in range(0,Nx):
        for j in range(1,Ny-1):
            
            dg[(Ny*i)+j] = g1[i,j+1] - g1[i,j]
            dg[(Nx*Ny)+(Ny*i)+j] = g2[i,j+1] - g2[i,j]
            dg[(2*Nx*Ny)+(Ny*i)+j] = g3[i,j+1] - g3[i,j]
            dg[(3*Nx*Ny)+(Ny*i)+j] = g4[i,j+1] - g4[i,j]
        
    return -(df/dx) -(dg/dy)

def vanleer(r):
    psi = (r+abs(r))/(1+abs(r))
    return psi

def FSx(UL,UR,schx):
    
    if(schx == SWx or schx == VLx):
        [fLm, fLp] = schx(UL)
        [fRm, fRp] = schx(UR)
        inFlux = fLp + fRm
    
    elif(schx == FDS):
        inFlux = FDS(UL,UR)
    
    return inFlux

def FSy(Ub,Ut,schy):
    
    if(schy == SWy or schy == VLy):
        [gbm, gbp] = schy(Ub)
        [gtm, gtp] = schy(Ut)
        inFlux = gbp + gtm
    
    elif(schy == FDS):
        inFlux = FDS(Ub,Ut)
    
    return inFlux

def SWx(U):
    
    [r,u,v,p] = U
    c = np.sqrt((g*p/r))
    Mu = u/c
    if(Mu<-1):
        Fm = FluxF(U)
        Fp = np.zeros(4)
    elif(Mu>1):
        Fp = FluxF(U)
        Fm = np.zeros(4)
    elif(Mu>= -1 and Mu<0):
        Fp = (r*(u+c)/(2*g))*np.array([1, u+c, v+c,  0.5*(u+c)**2 + (((3-g)/(g-1))*0.5*c**2)])
        Fm = FluxF(U) - Fp
    else: # 0 <= M <= 1
        Fm = (r*(u-c)/(2*g))*np.array([1, u-c, v-c, 0.5*(u-c)**2 + (((3-g)/(g-1))*0.5*c**2)])
        Fp = FluxF(U) - Fm
    
    return np.array([Fm, Fp])

def SWy(U):
    
    [r,u,v,p] = U
    c = np.sqrt((g*p/r))
    Mv = v/c
    if(Mv<-1):
        Gm = FluxG(U)
        Gp = np.zeros(4)
    elif(Mv>1):
        Gp = FluxG(U)
        Gm = np.zeros(4)
    elif(Mv>= -1 and Mv<0):
        Gp = (r*(v+c)/(2*g))*np.array([1, v+c, u+c, 0.5*(v+c)**2 + (((3-g)/(g-1))*0.5*c**2)])
        Gm = FluxG(U) - Gp
    else: # 0 <= M <= 1
        Gm = (r*(v-c)/(2*g))*np.array([1, v-c, u-c, 0.5*(v-c)**2 + (((3-g)/(g-1))*0.5*c**2)])
        Gp = FluxG(U) - Gm
    
    return np.array([Gm, Gp])

def VLx(U):
    
    [r,u,v,p] = U
    c = np.sqrt(g*p/r)
    Mu = u/c
    if(Mu<-1):
        Fm = FluxF(U)
        Fp = np.zeros(4)
    elif(Mu>1):
        Fp = FluxF(U)
        Fm = np.zeros(4)
    else: # 0 <= |M| <= 1
        t1f = (r*c/4.0)*(Mu+1)**2
        t2f = 0.5*(g-1)*Mu + 1
        Fp = t1f*np.array([1, 2*c*t2f/g, v, (2*(c*t2f)**2/(g**2 - 1.0)) + 0.5*v**2 ])
        Fm = FluxF(U) - Fp
    
    return np.array([Fm, Fp])

def VLy(U):
    
    [r,u,v,p] = U
    if(r<0):
        print(r)
    c = np.sqrt(g*p/r)
    Mv = v/c
    if(Mv<-1):
        Gm = FluxG(U)
        Gp = np.zeros(4)
    elif(Mv>1):
        Gp = FluxG(U)
        Gm = np.zeros(4)
    else: # 0 <= |M| <= 1
        t1g = (r*c/4.0)*(Mv+1)**2
        t2g = 0.5*(g-1)*Mv + 1
        Gp = t1g*np.array([1, u, 2*c*t2g/g, (2*(c*t2g)**2/(g**2 - 1.0)) + 0.5*u**2])
        Gm = FluxG(U) - Gp
    
    return np.array([Gm, Gp])


def FDS(UL, UR):
    
    [rL,uL,vL,pL] = UL
    [rR,uR,vR,pR] = UR
    cL = np.sqrt(g*pL/rL)
    cR = np.sqrt(g*pR/rR)
 
    # Stagnation Enthalpy
    HtL = Ht(rL,uL,vL,pL)
    HtR = Ht(rR,uR,vR,pR)
    
    # Evaluate Roe average quantities
    alpha = np.sqrt(rR/rL)
    r_a  = np.sqrt(rL*rR)
    u_a = (uL + alpha*uR)/(1+alpha)
    v_a = (vL + alpha*uR)/(1+alpha)
    Ht_a = (HtL + alpha*HtR)/(1+alpha)
    c_a = np.sqrt((g-1)*(Ht_a - 0.5*u_a**2))
    
    # eigenvalues of the Roe averaged matrix, and left and right states
    #eig_a = np.array([u_a, u_a-c_a, u_a+c_a])
    #eigL = np.array([uL, uL-cL, uL+cL])
    #eigR = np.array([uR, uR-cR, uR+cR])
    eig_a = np.array([u_a, u_a+c_a, u_a-c_a])
    eigL = np.array([uL, uL+cL, uL-cL])
    eigR = np.array([uR, uR+cR, uR-cR])
    
    # entropy correction
    eps = np.zeros(3)
    for i in range(1,3):
        eps[i] = 4*max(0, eigR[i] - eigL[i])
        #eps[i] = max(0, eig_a[i] - eigR[i], eig_a[i] - eigL[i])
        
    eig_corr = np.abs(eig_a)
    for i in range(1,3):
        if(abs(eig_a[i]) < eps[i]):
            eig_corr[i] = 0.5*((eig_a[i]**2/eps[i]) + eps[i])
            
    # eigenvectors corresponding tp conservative variables
    e1 = np.array([1, u_a, 0.5*u_a**2])
    e2 = (0.5*r_a/c_a)*np.array([1, u_a + c_a, Ht_a + (u_a*c_a)])
    e3 = (0.5*r_a/c_a)*np.array([1, u_a - c_a, Ht_a - (u_a*c_a)])
    
    evec = np.array([e1, e2, e3])
    
    # jump in primitive variables across the interface: left to right
    [dr, du, dp] = UR - UL

    # jump in characterictic variables
    delW = np.array([dr - (dp/c_a**2), du + (dp/(r_a*c_a)), -du + (dp/(r_a*c_a))])
    
    # interface flux: symmetric form
    Roeflux = 0.5*(FluxF(UL) + FluxF(UR) - np.dot(np.multiply(eig_corr,delW),evec))
    
    return Roeflux

def rungeKutta(u, fun):
    # Count number of iterations using step size or step height h
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

QC = QI
rho = np.zeros((Nx,Ny))
vel = np.zeros((Nx,Ny))
pr = np.zeros((Nx,Ny))

for it in range(0, nt):

    usol = solve_ivp(SOU, [it*dt, (it*dt)+dt], QC, args = [VLx,VLy,vanleer]) 
    for i in range(0,Nx):
        for j in range(0,Ny):
            rho[i,j] = QtoU([usol.y[(Ny*i)+j,-1],usol.y[(Nx*Ny)+(Ny*i)+j,-1],\
                             usol.y[(2*Nx*Ny)+(Ny*i)+j,-1],usol.y[(3*Nx*Ny)+(Ny*i)+j,-1]])[0]
            velx[i,j] = QtoU([usol.y[(Ny*i)+j,-1],usol.y[(Nx*Ny)+(Ny*i)+j,-1],\
                             usol.y[(2*Nx*Ny)+(Ny*i)+j,-1],usol.y[(3*Nx*Ny)+(Ny*i)+j,-1]])[1]
            vely[i,j] = QtoU([usol.y[(Ny*i)+j,-1],usol.y[(Nx*Ny)+(Ny*i)+j,-1],\
                             usol.y[(2*Nx*Ny)+(Ny*i)+j,-1],usol.y[(3*Nx*Ny)+(Ny*i)+j,-1]])[2]
            pr[i,j]  = QtoU([usol.y[(Ny*i)+j,-1],usol.y[(Nx*Ny)+(Ny*i)+j,-1],\
                             usol.y[(2*Nx*Ny)+(Ny*i)+j,-1],usol.y[(3*Nx*Ny)+(Ny*i)+j,-1]])[3]
    
    if (it%5==0):
        plt.figure(1)
        #plt.grid('true')
        plt.pcolor(x,y,rho);
        plt.xlabel(r'x');
        plt.ylabel(r'$\rho$');
        plt.title('Density - 1st order Van Leer Scheme for time = %1.3f' %t)
        
        plt.figure(2)
        #plt.grid('true')
        plt.pcolor(x,y,velx);
        plt.xlabel(r'x');
        plt.ylabel(r'$u_x$');
        plt.title('Velocity - 1st order Van Leer Scheme for time = %1.3f' %t)
        
        plt.figure(3)
        #plt.grid('true')
        plt.pcolor(x,y,vely);
        plt.xlabel(r'x');
        plt.ylabel(r'$u_y$');
        plt.title('Velocity - 1st order Van Leer Scheme for time = %1.3f' %t)
        
        plt.figure(4)
        #plt.grid('true')
        plt.pcolor(x,y,pr);
        plt.xlabel(r'x');
        plt.ylabel(r'p');
        plt.title('Pressure - 1st order Van Leer Scheme for time = %1.3f' %t)
    
    QC = usol.y[:,-1]
    print(it)
     
plt.show()
'''
np.savetxt('output/rho.txt',rho)
np.savetxt('output/velx.txt',velx)
np.savetxt('output/vely.txt',vely)
np.savetxt('output/pr.txt',pr)
'''
'''
import matplotlib.animation as ani
QC = QI
rho = np.zeros(Nx)
frames=nt-2
fig = plt.figure()
z = 0
def scatter_ani(i=int):
    global z
    global QC
    usol = solve_ivp(FOU, [z*dt, (z*dt)+dt], QC, args = [SW]) 
    for i in range(0,Nx):
        rho[i] = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[0]
    
    if (z%5==0):
        plt.figure(1)
        plt.grid('true')
        plt.plot(x,rho);
        plt.xlabel(r'x');
        plt.ylabel(r'$\rho$');
        plt.title('Density - 1st order Steger Warming Scheme for time = %1.3f' %t)
       
    QC = usol.y[:,-1]
    z = z+1

anim = ani.FuncAnimation(fig, scatter_ani, frames=frames, interval=200)
anim.save("1storderSW.gif")
'''
