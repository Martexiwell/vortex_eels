# -*- coding: utf-8 -*-
# Polarizability of spherical particle

# importing stuff
import numpy as np
# import scipy as scp
import scipy.constants as const
import sympy as sp
# from scipy.special import jv 
import matplotlib.pyplot as plt

import spectrum as spectrum
import king_im as king
import psi_perp


#%% polarizability
def SpherePolarizability():
    x = sp.Symbol("x")
    omega, a, epsilon, c = sp.symbols("omega a epsilon c")
    
    j1 = sp.sin(x) / x**2 - sp.cos(x) / x
    h1 = ( 1 / x**2 - sp.I / x ) * sp.exp( sp.I * x )
        
    ja = sp.diff(x * j1, x)
    ju = sp.diff(x * h1, x)
    
    rho0 = omega * a / c
    rho1 = rho0 * sp.sqrt(epsilon)
    
    j10 = j1.subs(x, rho0)
    j11 = j1.subs(x, rho1)
    h10 = h1.subs(x, rho0)
    
    ja0 = ja.subs(x, rho0)
    ja1 = ja.subs(x, rho1)
    ha0 = ju.subs(x, rho0)
    
    alpha = 3/2 * c**3 / omega**3 *     (-j10 * ja1 + epsilon * ja0 *j11) / (h10 * ja1 - epsilon * ha0 * j11) 
    
    alpha = alpha.subs(c, const.c)
    
    return sp.lambdify([omega, a, epsilon], alpha)

def epsilon(omega, omega_p, gamma):
    return 1 - omega_p**2 / (omega**2 + 1j * gamma * omega)

#%% Polarizability Matrices

def PolarizabilityMatrix(omega, a, epsilon):
    SP = SpherePolarizability()
    aEE = np.eye(3) * SP(omega, a, epsilon) 
    aMM = np.zeros((3,3))
    aEM = np.zeros((3,3))
    return [aEE, aMM, aEM]

def PolarizabilityMatrices(omega, a, epsilon):
    output = []
    for i in range(len(omega)):
        output.append(PolarizabilityMatrix(omega[i], a, epsilon[i]))
    return output

#%% omegas and alphas

print("Preparing the omegas and alphas...", end="")
Omegas = np.linspace(5,13,20) * const.eV/const.hbar
omega_p = 9.1 * const.eV/const.hbar
gamma_p = 0.15 * const.eV/const.hbar
epsilonko = epsilon(Omegas, omega_p, gamma_p)
Alphas = PolarizabilityMatrices(Omegas, 5e-9, epsilonko)
print("DONE")

#%% arrays
qxm = 1e-6
qym = qxm


print("Preparing input arrays ... ", end="")
numofxpoints = 50
numofypoints = 50
xrange = [-qxm,qxm]
yrange = [-qym,qym]
dx = (xrange[1] - xrange[0])/(numofxpoints-1)
dy = (yrange[1] - yrange[0])/(numofypoints-1)

xx = np.linspace(xrange[0], xrange[1], numofxpoints)
yy = np.linspace(yrange[0], yrange[1], numofypoints)

X, Y, X_, Y_   = np.meshgrid(xx, yy, xx, yy, indexing="ij")
print("DONE")
#%% parameters
print("Preparing parameters ... ", end="")
ppos = [0.28e-6,0,0]
qz = np.sqrt(2 * const.m_e * 60e3 * const.eV) / const.h
v = 0.446 * const.c

qr = qz * 1e-4      # cutoff q for queen (detector)
qc = qz * 1e-4      # cutoff q for initial psi (aperture)

lf=1
li=1
print("DONE")
#%% initial psis
print("Preparing the initial psis ... ", end="")
#psifun = psi_perp.psi_perp


R       = np.sqrt(X**2 + Y**2)
Phi     = np.arctan2(Y,X)
R_      = np.sqrt(X_**2 + Y_**2)
Phi_    = np.arctan2(Y_,X_)
#PSI1 = jv(li, qr * R) * np.exp(1j * li * Phi) #psifun(li, R, Phi, 1)
#PSI2 = jv(li, qr * R_) * np.exp(1j * li * Phi_)#psifun(li, R_, Phi_, 1)
PSI1 = psi_perp.psiperp(li, X,Y,qc)
PSI2 = psi_perp.psiperp(li, X_,Y_,qc)
print("DONE")
qc = 1


#%% king prep
print("Preparing the King ... ",end="")
K = king.setup(ppos, qz, v)
print("DONE)")

#%%
# The following part is for debugging and control purposes and before evaluation should be commented
#%% initial psis plot

fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.imshow(np.real(PSI1)[:,:, 0,0])
axim.imshow(np.imag(PSI1)[:,:, 0,0])
axabs.imshow(np.abs(PSI1)[:,:, 0,0])
axang.imshow(np.angle(PSI1)[:,:, 0,0])

#%% test of king and queen - prep
omega = Omegas[0]
alfamat = Alphas[0]

KK = king.setpolarizability(K, alfamat[0], alfamat[1], alfamat[2])
KK = king.setomega(KK, omega)
gradpsi1, gradpsi2 = spectrum.GradsOfPsi(PSI1, PSI2, qz, dx, dy)

#%% test of king and queen - plot of gradients
fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.imshow(np.real(gradpsi1[0])[:,:, 0,0])
axim.imshow(np.imag(gradpsi1[0])[:,:, 0,0])
axabs.imshow(np.abs(gradpsi1[0])[:,:, 0,0])
axang.imshow(np.angle(gradpsi1[0])[:,:, 0,0])
#%% test of king and queen - queen
qarg = spectrum.QueenFun(lf, X, Y, X_, Y_, qr)
#%% test of king and queen - plot of queen
fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.imshow(np.real(qarg)[:,:,20,20])
axim.imshow(np.imag(qarg)[:,:,20,20])
axabs.imshow(np.abs(qarg)[:,:,20,20])
axang.imshow(np.angle(qarg)[:,:,20,20])

#%% test of king and queen - king
karg = spectrum.KingFun(X, Y, X_, Y_, KK, gradpsi1, gradpsi2) 

#%% test of king and queen - plot of king
fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.imshow(np.real(karg)[:,:,25,25])
axim.imshow(np.imag(karg)[:,:,25,25])
axabs.imshow(np.abs(karg)[:,:,25,25])
axang.imshow(np.angle(karg)[:,:,25,25])

#%% test of king and queen - plot of argument
arg = karg*qarg

fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.imshow(np.real(arg)[:,:,10,10])
axim.imshow(np.imag(arg)[:,:,10,10])
axabs.imshow(np.abs(arg)[:,:,10,10])
axang.imshow(np.angle(arg)[:,:,10,10])

#%% Final calculation of spectrum
print("Initializing calculation of the spectrum...")
spec = spectrum.Spectrum(X, Y, X_, Y_, qz, v, ppos, PSI1, PSI2, K, lf, qc, Omegas, Alphas, filename="AgSphere5nm_0_0_0")
print(spec)

#%%
# import matplotlib.pyplot as plt

# omegas = np.linspace(4,7,100) * const.eV/const.hbar
# epsilons = epsilon(omegas, omega_p,gamma_p)
# SP = SpherePolarizability()
# a_radius = 5e-9
# alphas = SP(omegas, a_radius, epsilons)

# fig,ax = plt.subplots()
# ax.plot(omegas,alphas,)
# ax.grid()
# ax.set(xlabel = r"$\omega$",
#        ylabel = r"$\alpha(\omega)$",
#        title = f"Polarizability of Ag sphere of radius {a_radius} m")
# fig.savefig("Ag_polarizability.png",dpi=150)
