import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

L = 1
Nx = 100
dx = L/Nx
x = np.linspace(dx/2,L-dx/2,Nx)
a = 1
g = 1.4
R = 1#8.314
sigma = 0.5
dt = (sigma*dx)/a
t = 0.4
nt = int(t/dt)
tt = np.linspace(0,t,nt)
xp,tp = np.meshgrid(x,tt)
N = 30
vel = np.zeros(Nx)
rho = np.zeros(Nx)
pr = np.zeros(Nx) 
Et = np.zeros(Nx) 
QC = np.zeros(3*Nx)
QI = np.zeros(3*Nx)
IC = 1
UL1 = [1.0, 0.75, 1.0]
UR1 = [0.125, 0.0, 0.1]
UL2 = [1.0, -2.0, 0.4]
UR2 = [1.0, 2.0, 0.4]
UL3 = [1.0, 0.0, 1000]
UR3 = [1.0, 0.0, 0.01]
ULini = UL1 + UL2 + UL3
URini = UR1 + UR2 + UR3

for i in range(0,Nx):
    if(x[i] <= 0.3):
        rho[i] = ULini[(IC-1)*3 + 0]
        vel[i] = ULini[(IC-1)*3 + 1]
        pr[i] = ULini[(IC-1)*3 + 2]      
    else:
        rho[i] = URini[(IC-1)*3 + 0]
        vel[i] = URini[(IC-1)*3 + 1]
        pr[i] = URini[(IC-1)*3 + 2]
    
    QI[i] = rho[i]
    QI[Nx+i] = rho[i]*vel[i]
    QI[(2*Nx) + i] = rho[i]*(pr[i]/(rho[i]*(g-1)) + 0.5*vel[i]**2)

def Et(rho, u, p):
    return ((p/(rho*(g-1))) + 0.5*u**2)

def Ht(rho, u, p):
    return (((g*p)/(rho*(g-1))) + 0.5*u**2)

def pres(q1,q2,q3):
    return (g-1)*(q3 - (q2**2/(2*q1)))

def QtoU(Q):
    [q1,q2,q3] = Q
    return np.array([q1, q2/q1, (q3 - (q2**2/(2*q1)))*(g-1)])

def UtoQ(U):
    [r,u,p] = U
    return np.array([r, r*u, r*(p/(r*(g-1)) + 0.5*u**2)])

def Flux(U):
    [r,u,p] = U
    return np.array([r*u, r*u**2 + p, (r*Et(r,u,p) + p)*u])
    
def FTCS(t,QC): 
   
    Q1 = np.zeros(Nx)
    Q2 = np.zeros(Nx)
    Q3 = np.zeros(Nx)
    F1 = np.zeros(Nx+1)
    F2 = np.zeros(Nx+1)
    F3 = np.zeros(Nx+1)
    dfc = np.zeros(3*Nx)
    
    for i in range(0,Nx):
        Q1[i] = QC[i]
        Q2[i] = QC[Nx+i]
        Q3[i] = QC[(2*Nx)+i]
    
    for i in range(1,Nx+1):
        
        F1[i] = Q2[i-1]
        F2[i] = Q2[i-1]**2/Q1[i-1] + pres(Q1[i-1],Q2[i-1],Q3[i-1])
        F3[i] = ((Q3[i-1]*Q2[i-1])/Q1[i-1]) + (pres(Q1[i-1],Q2[i-1],Q3[i-1])*Q2[i-1]/Q1[i-1])      
    
    #Transmissive boundary conditions
    F1[0] = Q2[0]
    F2[0] = Q2[0]**2/Q1[0] + pres(Q1[0],Q2[0],Q3[0])
    F3[0] = ((Q3[0]*Q2[0])/Q1[0]) + (pres(Q1[0],Q2[0],Q3[0])*Q2[0]/Q1[0])    
    
    for i in range(1,Nx-1):
        
        c = np.sqrt(g*R*pres(Q1[i],Q2[i],Q3[i])/Q1[i])
        mu = max(abs(Q2[i]/Q1[i]) , abs((Q2[i]/Q1[i]) + c), abs((Q2[i]/Q1[i]) - c))
        dfc[i] = F1[i+1] - F1[i-1] - mu*(Q1[i+1] - 2*Q1[i] + Q1[i-1])
        dfc[Nx+i] = F2[i+1] - F2[i-1] - mu*(Q2[i+1] - 2*Q2[i] + Q2[i-1])
        dfc[(2*Nx)+i] = F3[i+1] - F3[i-1] - mu*(Q3[i+1] - 2*Q3[i] + Q3[i-1])
    
    return -0.5*dfc/dx
 
def FOU(t,QC,sch):
    
    f1 = np.zeros(Nx+1)
    f2 = np.zeros(Nx+1)
    f3 = np.zeros(Nx+1)
    U1 = np.zeros(Nx)
    U2 = np.zeros(Nx)
    U3 = np.zeros(Nx)
    dfc = np.zeros(3*Nx)
    
    for i in range(0,Nx):
 
        U1[i] = QtoU([QC[i],QC[Nx+i],QC[(2*Nx) + i]])[0]
        U2[i] = QtoU([QC[i],QC[Nx+i],QC[(2*Nx) + i]])[1]
        U3[i] = QtoU([QC[i],QC[Nx+i],QC[(2*Nx) + i]])[2]
        
    for i in range(1,Nx):
        f1[i] = FS(np.array([U1[i-1],U2[i-1],U3[i-1]]),np.array([U1[i],U2[i],U3[i]]),sch)[0]
        f2[i] = FS(np.array([U1[i-1],U2[i-1],U3[i-1]]),np.array([U1[i],U2[i],U3[i]]),sch)[1]
        f3[i] = FS(np.array([U1[i-1],U2[i-1],U3[i-1]]),np.array([U1[i],U2[i],U3[i]]),sch)[2]
    
    # Extrapolation Boundary Conditions
    f1[0] = f1[1]
    f1[-1] = f1[-2]
    f2[0] = f2[1]
    f2[-1] = f2[-2]
    f3[0] = f3[1]
    f3[-1] = f3[-2]
    
    '''
    # Wall Boundary Conditions
    f1[0] = 0
    f1[-1] = 0
    f2[0] = 0
    f2[-1] = 0
    f3[0] = 0
    f3[-1] = 0
    '''
        
    for i in range(1,Nx-1):
  
        dfc[i] = f1[i+1] - f1[i]
        dfc[Nx+i] = f2[i+1] - f2[i]
        dfc[(2*Nx)+i] = f3[i+1] - f3[i]
        
    return -dfc/dx

def SOU(t,QC,sch,lim):
  
    f1 = np.zeros(Nx+1)
    f2 = np.zeros(Nx+1)
    f3 = np.zeros(Nx+1)
    U1 = np.zeros(Nx)
    U2 = np.zeros(Nx)
    U3 = np.zeros(Nx)
    dfc = np.zeros(3*Nx)
    rL1 = np.ones(Nx+1)
    rR1 = np.ones(Nx+1)
    rL2 = np.ones(Nx+1)
    rR2 = np.ones(Nx+1)
    rL3 = np.ones(Nx+1)
    rR3 = np.ones(Nx+1)
            
    for i in range(0,Nx):
 
        U1[i] = QtoU([QC[i],QC[Nx+i],QC[(2*Nx) + i]])[0]
        U2[i] = QtoU([QC[i],QC[Nx+i],QC[(2*Nx) + i]])[1]
        U3[i] = QtoU([QC[i],QC[Nx+i],QC[(2*Nx) + i]])[2]
           
    for i in range(2,Nx-1):
        
        if((U1[i-1] - U1[i-2]) != 0):
            rL1[i] = (U1[i]-U1[i-1])/(U1[i-1]-U1[i-2])
        if((U1[i] - U1[i+1]) != 0):   
            rR1[i] = (U1[i-1]-U1[i])/(U1[i]-U1[i+1])
        if((U2[i-1] - U2[i-2]) != 0):
            rL2[i] = (U2[i]-U2[i-1])/(U2[i-1]-U2[i-2])
        if((U2[i] - U2[i+1]) != 0):   
            rR2[i] = (U2[i-1]-U2[i])/(U2[i]-U2[i+1])
        if((U3[i-1] - U3[i-2]) != 0):
            rL3[i] = (U3[i]-U3[i-1])/(U3[i-1]-U3[i-2])
        if((U3[i] - U3[i+1]) != 0):   
            rR3[i] = (U3[i-1]-U3[i])/(U3[i]-U3[i+1])    
            
    for i in range(2,Nx-1):
        
        Ul1 = U1[i-1] + lim(rL1[i])*0.5*(U1[i-1] - U1[i-2])
        Ur1 = U1[i] + lim(rR1[i])*0.5*(U1[i] - U1[i+1])
        Ul2 = U2[i-1] + lim(rL2[i])*0.5*(U2[i-1] - U2[i-2])
        Ur2 = U2[i] + lim(rR2[i])*0.5*(U2[i] - U2[i+1])
        Ul3 = U3[i-1] + lim(rL3[i])*0.5*(U3[i-1] - U3[i-2])
        Ur3 = U3[i] + lim(rR3[i])*0.5*(U3[i] - U3[i+1])
        f1[i] = FS(np.array([Ul1,Ul2,Ul3]),np.array([Ur1,Ur2,Ur3]),sch)[0]
        f2[i] = FS(np.array([Ul1,Ul2,Ul3]),np.array([Ur1,Ur2,Ur3]),sch)[1]
        f3[i] = FS(np.array([Ul1,Ul2,Ul3]),np.array([Ur1,Ur2,Ur3]),sch)[2]
        
    f1[1] = FS(np.array([U1[0],U2[0],U3[0]]),np.array([U1[1],U2[1],U3[1]]),sch)[0]
    f2[1] = FS(np.array([U1[0],U2[0],U3[0]]),np.array([U1[1],U2[1],U3[1]]),sch)[1]
    f3[1] = FS(np.array([U1[0],U2[0],U3[0]]),np.array([U1[1],U2[1],U3[1]]),sch)[2]
    
    f1[-2] = FS(np.array([U1[-2],U2[-2],U3[-2]]),np.array([U1[-1],U2[-1],U3[-1]]),sch)[0]
    f2[-2] = FS(np.array([U1[-2],U2[-2],U3[-2]]),np.array([U1[-1],U2[-1],U3[-1]]),sch)[1]
    f3[-2] = FS(np.array([U1[-2],U2[-2],U3[-2]]),np.array([U1[-1],U2[-1],U3[-1]]),sch)[2]
    
    
    # Extrapolation Boundary Conditions
    f1[0] = f1[1]
    f1[-1] = f1[-2]
    f2[0] = f2[1]
    f2[-1] = f2[-2]
    f3[0] = f3[1]
    f3[-1] = f3[-2]
    
    '''
    # Wall Boundary Conditions
    f1[0] = 0
    f1[-1] = 0
    f2[0] = 0
    f2[-1] = 0
    f3[0] = 0
    f3[-1] = 0
    '''
    
    for i in range(1,Nx-1):
  
        dfc[i] = f1[i+1] - f1[i]
        dfc[Nx+i] = f2[i+1] - f2[i]
        dfc[(2*Nx)+i] = f3[i+1] - f3[i]
        
    return -dfc/dx

def vanleer(r):
    psi = (r+abs(r))/(1+abs(r))
    return psi

def minmod(r):
    psi = max(0,min(1,r))
    return psi

def superbee(r):
    psi = max(0,min(2*r,1),min(r,2))
    return psi

def FS(UL,UR,sch):
    
    if(sch == SW or sch == VL):
        [fLm, fLp] = sch(UL)
        [fRm, fRp] = sch(UR)
        inFlux = fLp + fRm
    
    elif(sch == FDS):
        inFlux = FDS(UL,UR)
    
    return inFlux

def SW(U):
    
    [r,u,p] = U
    c = np.sqrt((g*p/r))
    M = u/c
    if(M<-1):
        Fm = Flux(U)
        Fp = np.zeros(3)
    elif(M>1):
        Fp = Flux(U)
        Fm = np.zeros(3)
    elif(M>= -1 and M<0):
        Fp = (r*(u+c)/(2*g))*np.array([1, u+c, 0.5*(u+c)**2 + (((3-g)/(g-1))*0.5*c**2)])
        Fm = Flux(U) - Fp
    else: # 0 <= M <= 1
        Fm = (r*(u-c)/(2*g))*np.array([1, u-c, 0.5*(u-c)**2 + (((3-g)/(g-1))*0.5*c**2)])
        Fp = Flux(U) - Fm
    
    return np.array([Fm, Fp])

def VL(U):
    
    [r,u,p] = U
    c = np.sqrt(g*p/r)
    M = u/c
    if(M<-1):
        Fm = Flux(U)
        Fp = np.zeros(3)
    elif(M>1):
        Fp = Flux(U)
        Fm = np.zeros(3)
    else: # 0 <= |M| <= 1
        t1p = (r*c/4.0)*(M+1)**2
        t2p = 0.5*(g-1)*M + 1
        Fp = t1p*np.array([1, 2*c*t2p/g, 2*(c*t2p)**2/(g**2 - 1.0)])
        Fm = Flux(U) - Fp
    
    return np.array([Fm, Fp])


def FDS(UL, UR):
    
    [rL,uL,pL] = UL
    [rR,uR,pR] = UR
    cL = np.sqrt(g*pL/rL)
    cR = np.sqrt(g*pR/rR)
    '''
    ML = uL/cL
    MR = uR/cR
    if(ML > 1):
        Roeflux = Flux(UL)
    elif(MR < -1):
        Roeflux = Flux(UR)
    else:
    '''
    # Stagnation Enthalpy
    HtL = Ht(rL,uL,pL)
    HtR = Ht(rR,uR,pR)
    
    # Evaluate Roe average quantities
    alpha = np.sqrt(rR/rL)
    r_a  = np.sqrt(rL*rR)
    u_a = (uL + alpha*uR)/(1+alpha)
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
    Roeflux = 0.5*(Flux(UL) + Flux(UR) - np.dot(np.multiply(eig_corr,delW),evec))
    
    return Roeflux

'''
QC = QI
rhoc = np.zeros((Nx, nt))
velc = np.zeros((Nx, nt))
prc = np.zeros((Nx, nt))
rho = np.zeros(Nx)
vel = np.zeros(Nx)
pr = np.zeros(Nx)
for it in range(0, nt):

    usol = solve_ivp(FTCS, [it*dt, (it*dt)+dt], QC) 
    for i in range(0,Nx):
        rho[i] = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[0]
        vel[i] = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[1]
        pr[i]  = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[2]
    rhoc[:,it] = rho
    velc[:,it] = vel
    prc[:,it] = pr
    if (it%5==0):
        plt.figure(1)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,rho);
        plt.xlabel(r'x');
        plt.ylabel(r'$\rho$');
        plt.title('Density - Central Scheme for Dissipation for time = %1.3f' %t)
        #plt.pause(0.2)
        
        plt.figure(2)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,vel);
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Velocity - Central Scheme for Dissipation for time = %1.3f' %t)
        #plt.pause(0.2)
        
        plt.figure(3)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,pr);
        plt.xlabel(r'x');
        plt.ylabel(r'p');
        plt.title('Pressure - Central Scheme for Dissipation for time = %1.3f' %t)
        #plt.pause(0.2)
    
    QC = usol.y[:,-1]
     
plt.show()

plt.figure(4)
plt.contourf(xp, tp, np.transpose(rhoc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
plt.title('Density - Central Scheme for Dissipation for time = %1.3f' %t)

plt.figure(5)
plt.contourf(xp, tp, np.transpose(velc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
plt.title('Velocity - Central Scheme for Dissipation for time = %1.3f' %t)

plt.figure(6)
plt.contourf(xp, tp, np.transpose(prc), N, cmap = 'bwr')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
plt.title('Pressure - Central Scheme for Dissipation for time = %1.3f' %t)
'''
'''
QC = QI
rho = np.zeros(Nx)
vel = np.zeros(Nx)
pr = np.zeros(Nx)
rhoc = np.zeros((Nx, nt))
velc = np.zeros((Nx, nt))
prc = np.zeros((Nx, nt))

for it in range(0, nt):

    usol = solve_ivp(FOU, [it*dt, (it*dt)+dt], QC, args = [SW]) 
    for i in range(0,Nx):
        rho[i] = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[0]
        vel[i] = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[1]
        pr[i]  = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[2]
    rhoc[:,it] = rho
    velc[:,it] = vel
    prc[:,it] = pr
    
    if (it%5==0):
        plt.figure(7)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,rho);
        plt.xlabel(r'x');
        plt.ylabel(r'$\rho$');
        plt.title('Density - 1st order Steger Warming Scheme for time = %1.3f' %t)
        #plt.title('Density - 1st order Van Leer Scheme for time = %1.3f' %t)
        #plt.title('Density - 1st order Roe Scheme for time = %1.3f' %t)
        #plt.pause(0.2)
        
        plt.figure(8)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,vel);
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        plt.title('Velocity - 1st order Steger Warming Scheme for time = %1.3f' %t)
        #plt.title('Velocity - 1st order Van Leer Scheme for time = %1.3f' %t)
        #plt.title('Velocity - 1st order Roe Scheme for time = %1.3f' %t)
        #plt.pause(0.2)
        
        plt.figure(9)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,pr);
        plt.xlabel(r'x');
        plt.ylabel(r'p');
        plt.title('Pressure - 1st order Steger Warming Scheme for time = %1.3f' %t)
        #plt.title('Pressure - 1st order Van Leer Scheme for time = %1.3f' %t)
        #plt.title('Pressure - 1st order Roe Scheme for time = %1.3f' %t)
        #plt.pause(0.2)
    
    QC = usol.y[:,-1]
     
plt.show()

plt.figure(10)
plt.contourf(xp, tp, np.transpose(rhoc), N, cmap = 'bwr', extend = 'both')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
plt.title('Density - 1st order Steger Warming Scheme for time = %1.3f' %t)
#plt.title('Density - 1st order Van Leer Scheme for time = %1.3f' %t)
#plt.title('Density - 1st order Roe Scheme for time = %1.3f' %t)

plt.figure(11)
plt.contourf(xp, tp, np.transpose(velc), N, cmap = 'bwr', extend = 'both')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
plt.title('Velocity - 1st order Steger Warming Scheme for time = %1.3f' %t)
#plt.title('Velocity - 1st order Van Leer Scheme for time = %1.3f' %t)
#plt.title('Velocity - 1st order Roe Scheme for time = %1.3f' %t)

plt.figure(12)
CS=plt.contourf(xp, tp, np.transpose(prc), N, cmap = 'bwr', extend = 'both')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
plt.title('Pressure - 1st order Steger Warming Scheme for time = %1.3f' %t)
#plt.title('Pressure - 1st order Van Leer Scheme for time = %1.3f' %t)
#plt.title('Pressure - 1st order Roe Scheme for time = %1.3f' %t)
'''

QC = QI
rhoc = np.zeros((Nx, nt))
velc = np.zeros((Nx, nt))
prc = np.zeros((Nx, nt))
rho = np.zeros(Nx)
vel = np.zeros(Nx)
pr = np.zeros(Nx)

for it in range(0, nt):

    usol = solve_ivp(SOU, [it*dt, (it*dt)+dt], QC, args = [FDS, vanleer]) 
    for i in range(0,Nx):
        rho[i] = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[0]
        vel[i] = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[1]
        pr[i]  = QtoU([usol.y[i,-1],usol.y[Nx+i,-1],usol.y[(2*Nx)+i,-1]])[2]
    rhoc[:,it] = rho
    velc[:,it] = vel
    prc[:,it] = pr
    if (it%5==0):
        plt.figure(13)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,rho);
        plt.xlabel(r'x');
        plt.ylabel(r'$\rho$');
        #plt.title('Density - 2nd order Steger Warming for time = %1.3f' %t)
        #plt.title('Density - 2nd order Van Leer Scheme for time = %1.3f' %t)
        plt.title('Density - 2nd order Roe Scheme for time = %1.3f' %t)
        #plt.pause(0.2)
        
        plt.figure(14)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,vel);
        plt.xlabel(r'x');
        plt.ylabel(r'u');
        #plt.title('Velocity - 2nd order Steger Warming for time = %1.3f' %t)
        #plt.title('Velocity - 2nd order Van Leer Scheme for time = %1.3f' %t)
        plt.title('Velocity - 2nd order Roe Scheme for time = %1.3f' %t)
        #plt.pause(0.2)
        
        plt.figure(15)
        #plt.clf()
        plt.grid('true')
        plt.plot(x,pr);
        plt.xlabel(r'x');
        plt.ylabel(r'p');
        #plt.title('Pressure - 2nd order Steger Warming for time = %1.3f' %t)
        #plt.title('Pressure - 2nd order Van Leer Scheme for time = %1.3f' %t)
        plt.title('Pressure - 2nd order Roe Scheme for time = %1.3f' %t)
        #plt.pause(0.2)
    
    QC = usol.y[:,-1]
     
plt.show()

plt.figure(16)
plt.contourf(xp, tp, np.transpose(rhoc), N, cmap = 'bwr', extend  = 'both')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
#plt.title('Density - 2nd order Steger Warming Scheme for time = %1.3f' %t)
#plt.title('Density - 2nd order Van Leer Scheme for time = %1.3f' %t)
plt.title('Density - 2nd order Roe Scheme for time = %1.3f' %t)

plt.figure(17)
plt.contourf(xp, tp, np.transpose(velc), N, cmap = 'bwr', extend = 'both')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
#plt.title('Velocity - 2nd order Steger Warming Scheme for time = %1.3f' %t)
#plt.title('Velocity - 2nd order Van Leer Scheme for time = %1.3f' %t)
plt.title('Velocity - 2nd order Roe Scheme for time = %1.3f' %t)

plt.figure(18)
plt.contourf(xp, tp, np.transpose(prc), N, cmap = 'bwr', extend = 'both')
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r't');
#plt.title('Pressure - 2nd order Steger Warming Scheme for time = %1.3f' %t)
#plt.title('Velocity - 2nd order Van Leer Scheme for time = %1.3f' %t)
plt.title('Velocity - 2nd order Roe Scheme for time = %1.3f' %t)

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
