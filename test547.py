import numpy as np
import matplotlib.pyplot as plt
L = 2
Nx = 100
Ny = 100
dx = L/Nx
dy = L/Ny
x = np.linspace(dx/2,L-dx/2,Nx)
y = np.linspace(dy/2,L-dy/2,Ny)
t = 5
rho = np.zeros((Nx,Ny))
rhof = list(map(float,open("rho.txt").read().split()))
for i in range(0,Nx):
    for j in range(0,Ny):
        rho[i,j] = rhof[(Ny*i)+j]
        
velx = np.zeros((Nx,Ny))
velxf = list(map(float,open("velx.txt").read().split()))
for i in range(0,Nx):
    for j in range(0,Ny):
        velx[i,j] = velxf[(Ny*i)+j]
        
vely = np.zeros((Nx,Ny))
velyf = list(map(float,open("vely.txt").read().split()))
for i in range(0,Nx):
    for j in range(0,Ny):
        vely[i,j] = velyf[(Ny*i)+j]
        
pr = np.zeros((Nx,Ny))
prf = list(map(float,open("pr.txt").read().split()))
for i in range(0,Nx):
    for j in range(0,Ny):
        pr[i,j] = prf[(Ny*i)+j]

plt.figure(1)
plt.pcolor(x,y,rho);
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r'$\rho$');
plt.title('Density - 1st order Van Leer Scheme for time = %1.3f' %t)

plt.figure(2)
plt.pcolor(x,y,velx);
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r'$u_x$');
plt.title('Velocity - 1st order Van Leer Scheme for time = %1.3f' %t)

plt.figure(3)
plt.pcolor(x,y,vely);
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r'$u_y$');
plt.title('Velocity - 1st order Van Leer Scheme for time = %1.3f' %t)

plt.figure(4)
plt.pcolor(x,y,pr);
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r'p');
plt.title('Pressure - 1st order Van Leer Scheme for time = %1.3f' %t)

plt.figure(5)
plt.contourf(x,y,rho);
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r'$\rho$');
plt.title('Density - 1st order Van Leer Scheme for time = %1.3f' %t)

plt.figure(6)
plt.contourf(x,y,velx);
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r'$u_x$');
plt.title('Velocity - 1st order Van Leer Scheme for time = %1.3f' %t)

plt.figure(7)
plt.contourf(x,y,vely);
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r'$u_y$');
plt.title('Velocity - 1st order Van Leer Scheme for time = %1.3f' %t)

plt.figure(8)
plt.contourf(x,y,pr);
plt.colorbar()
plt.xlabel(r'x');
plt.ylabel(r'p');
plt.title('Pressure - 1st order Van Leer Scheme for time = %1.3f' %t)
