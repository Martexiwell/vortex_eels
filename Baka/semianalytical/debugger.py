import numpy as np
import scipy as scp
import scipy.constants as const
from scipy.integrate import quad
from scipy.interpolate import interp2d
import sympy as sp
#import pickle
import matplotlib.pyplot as plt

import king
import spectrum
#import spherical_particle as sphp
#import psi_perp
#%%
K = king.setup([0,0,0], 1e6)
K = king.setpolarizability(K, np.eye(3), np.eye(3), np.eye(3))
K = king.setomega(K, 20)

x,y,z,x_,y_,z_ = sp.symbols("x y z x' y' z'")
gv1x, gv1y, gv1z, gv2x, gv2y, gv2z = sp.symbols("gv1x gv1y gv1z gv2x gv2y gv2z")
kfun = sp.lambdify((x,y,z,x_,y_,z_, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z ), K[0,0], modules="scipy")

#%%
xrange = np.linspace(-10,10,15)
X,Y,X_,Y_ = np.meshgrid(xrange,xrange,xrange,xrange)

Qin = spectrum.QueenFun(1,X,Y,X_,Y_,1)

#%%

omegas = np.linspace(4,9, 5)
SP = SpherePolarizability()
epsilons = epsilon(omegas, 8, 2)
polarizabilities = PolarizabilityMatrices(omegas, 1e-3, epsilons)

#%%
qz = 1e6
v = 1e6
ppos = [0,0,0]
psiint1 = np.sinc(np.sqrt(X**2 + Y**2))
psiint2 = np.sinc(np.sqrt(X_**2 + Y_**2))
lfin = 1
qc = 1

#%%

dx = X[1,0,0,0] - X[0,0,0,0]
dy = Y[0,1,0,0] - Y[0,0,0,0]

print("Calculating the Queen ... ", end="")
theQueen = QueenFun(lfin, X,Y,X_,Y_, qc)
print("DONE")


gradpsi1, gradpsi2 = GradsOfPsi(psiint1, psiint2, qz, dx, dy)


print("Preparing the Green's tensor ...",end="")
Kgeo = king.setup(ppos, v)
print("DONE")

omegai = 0
omega = omegas[omegai]
aEE, aMM, aEM = polarizabilities[omegai]

Komega = king.setpolarizability(Kgeo, aEE, aMM, aEM)
Komega = king.setomega(Komega, omega)
KFun =  sp.lambdify((x,y,z,x_,y_,z_, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z ),
                    Komega[0,0]) 

v1x = gradpsi1[0]
v1y = gradpsi1[1]
v1z = gradpsi1[2]
v2x = gradpsi2[0]
v2y = gradpsi2[1]
v2z = gradpsi2[2]

#%%

vKFun = np.vectorize(KFun)
#%%
finalKing = KFun(X,Y,0*X, X_,Y_,0*X_, v1x,v1y,v1z,v2x,v2y,v2z)
#%%
finalKing = KFun(X,Y,0*X, X_,Y_,0*X_, 1e-3*v1x,1e-3*v1y,1e-3*v1z,1e-3*v2x,1e-3*v2y,1e-3*v2z)
#%%
finalKing = KFun(1e-6*X,1e-6*Y,0*1e-6*X, 1e-6*X_,1e-6*Y_,0*X_, v1x,v1y,v1z,v2x,v2y,v2z)
v1x#%%
finalKing = KFun(X,X,X,X,X,X,X,X,X,X,X,X)
#%%
finalKing = KFun(xrange, xrange, xrange, xrange, xrange, xrange, xrange, xrange, xrange, xrange, xrange, xrange )

#%%
testsp = sp.lambdify((x,y), x+y**2)

#%%
finalKing = KFun(1,2,3,4,5,6,7,8,9,10,11,12)