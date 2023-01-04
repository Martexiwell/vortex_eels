# -*- coding: utf-8 -*-
# Polarizability of spherical particle

# importing stuff
import numpy as np
import scipy as scp
import scipy.constants as const
import sympy as sp

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
    aEE = np.eye(3) * np.imag(SP(omega, a, epsilon) )
    return [aEE, np.zeros((3,3)), np.zeros((3,3))]

def PolarizabilityMatrices(omega, a, epsilon):
    output = []
    for i in range(len(omega)):
        output.append(PolarizabilityMatrix(omega[i], a, epsilon[i]))
    return output

#%%

print("Preparing the omegas and alphas...", end="")
Omegas = np.linspace(5,13,8) * const.eV/const.hbar
omega_p = 9.1 * const.eV/const.hbar
gamma_p = 0.15 * const.eV/const.hbar
epsilonko = epsilon(Omegas, omega_p, gamma_p)
Alfas = PolarizabilityMatrices(Omegas, 5e-9, epsilonko)
print("DONE")

#%%
import spectrum
import green
import psi_perp
from scipy.special import jv 

print("preparing input arrays ... ", end="")
numofxpoints = 20
numofypoints = 20
xrange = [-1e-6,1e-6]
yrange = [-1e-6,1e-6]
dx = (xrange[1] - xrange[0])/(numofxpoints-1)
dy = (yrange[1] - yrange[0])/(numofypoints-1)

xx = np.linspace(xrange[0], xrange[1], numofxpoints)
yy = np.linspace(yrange[0], yrange[1], numofypoints)

XX, YY, XX_, YY_   = np.meshgrid(xx, yy, xx, yy, indexing="ij")

ppos = [0,0,0]
qz = np.sqrt(2 * const.m_e * 60e3 * const.eV) / const.h
v = 0.446 * const.c

lf=1
li=1

print("DONE")
print("Preparing the initial psis...", end="")
psifun = psi_perp.psi_perp
R       = np.sqrt(XX**2 + YY**2)
Phi     = np.arctan2(YY,XX)
R_      = np.sqrt(XX_**2 + YY_**2)
Phi_    = np.arctan2(YY_,XX_)
PSI1 = jv(li, R) * np.exp(Phi) #psifun(li, R, Phi, 1)
PSI2 = jv(li, R_) * np.exp(Phi_)#psifun(li, R_, Phi_, 1)
print("DONE")
qc = 1

print("Preparing the Green's tensor...")
GG = green.setup(ppos, qz, v)

#%%
print("Initializing calculation of the spectrum...")
spec = spectrum.Spectrum(XX, YY, XX_, YY_, qz, v, ppos, PSI1, PSI2, GG, lf, qc, Omegas, Alfas, filename="Ag_5nm_sphere")
print(spec)

#%%
import matplotlib.pyplot as plt

omegas = np.linspace(4,7,100) * const.eV/const.hbar
epsilons = epsilon(omegas, omega_p,gamma_p)
SP = SpherePolarizability()
a_radius = 5e-9
alphas = SP(omegas, a_radius, epsilons)

fig,ax = plt.subplots()
ax.plot(omegas,alphas,)
ax.grid()
ax.set(xlabel = r"$\omega$",
       ylabel = r"$\alpha(\omega)$",
       title = f"Polarizability of Ag sphere of radius {a_radius} m")
fig.savefig("Ag_polarizability.png",dpi=150)
