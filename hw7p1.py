import numpy as np
import matplotlib.pyplot as plt
L = 1
Nx = 100
dx = L/Nx
x = np.linspace(dx/2,L-dx/2,Nx)
gamma = 1
D = gamma/dx
P = 10
Pi = P*dx
d = np.zeros(Nx)
phiw = 1
phie = 0
phi0 = phiw
phiL = phie

phi_a = phi0 + (phiL - phi0)*((np.exp(P*x)-1)/(np.exp(P)-1))

def TDMA(n,a,b,c,d):
    y = np.zeros(Nx)
    for i in range(1,n):
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]
    y[-1] = d[-1]/b[-1]
    for i in range(n-2,-1,-1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]
    return y

def CD(P):
    return 1-(0.5*P)

def up(P):
    return 1
    
def hyb(P):
    return max(0, 1-(0.5*P))

def power(P):
    return max(0, (1-(0.1*P))**5)

def exp(P):
    return P/(np.exp(P)-1)

aw = D*(power(abs(Pi)) + max(Pi,0))*np.ones(Nx)    
ae = D*(power(abs(Pi)) + max(-Pi,0))*np.ones(Nx)
ap  = ae + aw
d[0] = -aw[0]*phiw
d[-1] = -ae[-1]*phie
phi = TDMA(Nx,aw,-ap,ae,d)

plt.figure(1)
plt.grid(True)
plt.plot(x,phi,x,phi_a,'r')
plt.legend(['Numerical','Analytical'])
plt.show()