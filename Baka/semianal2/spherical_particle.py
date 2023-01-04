# -*- coding: utf-8 -*-
# Polarizability of spherical particle

## importing stuff
import numpy as np
# import scipy as scp
import scipy.constants as const
import sympy as sp
# from scipy.special import jv 
from datetime import datetime
starttime = datetime.now()

## My >>toolboxes<<
import spectrum
#import green
import psi_perp


#%%
## Folowing don't currently work on egon & karl
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
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

def epsilon(omega, omega_p, gamma, epsilon_0=1):
    return epsilon_0 * ( 1 - omega_p**2 / (omega**2 + 1j * gamma * omega) )

#%% Polarizability Matrices

def PolarizabilityMatrix(omega, a, epsilon, multipliers=(1,0,0)):
    # multipliers are for choosing of nonzero/zero polarizability tensors
    SP = SpherePolarizability()
    aEE = multipliers[0] * np.eye(3) * SP(omega, a, epsilon) 
    aMM = multipliers[1] * np.eye(3) * SP(omega, a, epsilon) 
    aEM = multipliers[2] * np.eye(3) * SP(omega, a, epsilon) 
    return [aEE, aMM, aEM]

def PolarizabilityMatrices(omega, a, epsilon, multipliers=(1,0,0)):
    output = []
    for i in range(len(omega)):
        output.append(PolarizabilityMatrix(omega[i], a, epsilon[i], multipliers=multipliers))
    return output

#%% Omegas and alphas

print("Preparing the omegas and alphas...", end="")
sphrad = 5e-10               #radius of particle
polsmultipliers = (0,0,1)   #polarizability tensor multiplier to choose

Omegas = np.linspace(.6,1,50) *1e16 #np.linspace(3,9,50) * const.eV/const.hbar
omega_p = 9.1 * const.eV/const.hbar
gamma_p = 0.15 * const.eV/const.hbar
epsilonko = epsilon(Omegas, omega_p, gamma_p)
spols = SpherePolarizability()(Omegas, sphrad, epsilonko)
Alphas = PolarizabilityMatrices(Omegas, sphrad, epsilonko, polsmultipliers)
print("DONE")

#%% plot polarizability
fig,ax = plt.subplots()
ax.plot(Omegas,np.real(spols), "b", label="Re")
ax.plot(Omegas,np.imag(spols), "r", label="Im")
ax.set(xlabel = r"$\omega\, /\, \mathrm{rad}\cdot\mathrm{s}^{-1}$",
       ylabel = r"$\alpha (\omega)$")
ax.grid()
ax.legend()

del spols

#%% Arrays
xm = 1.0e-8
ym = xm

print("Preparing input arrays ... ", end="")
numofxpoints = 50
numofypoints = 50
xrange = [-xm,xm]
yrange = [-ym,ym]
dx = (xrange[1] - xrange[0])/(numofxpoints-1)
dy = (yrange[1] - yrange[0])/(numofypoints-1)

xx = np.linspace(xrange[0], xrange[1], numofxpoints)
yy = np.linspace(yrange[0], yrange[1], numofypoints)

X, Y, X_, Y_   = np.meshgrid(xx, yy, xx, yy, indexing="ij")
print("DONE")
#%% parameters
print("Preparing parameters ... ", end="")
ppos = [0.2e-8,0,0]
qz = np.sqrt(2 * const.m_e * 60e3 * const.eV) / const.h
v = 0.446 * const.c

qc = qz * 1e-2      # cutoff q for queen (detector)
qa = qz * 1e-2      # cutoff q for initial psi (aperture)

li=0
lf=1

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
PSI1 = psi_perp.psiperp(li, X,Y,qa)
PSI2 = psi_perp.psiperp(li, X_,Y_,qa)
print("DONE")


#%% Metadata
#filename = f"{starttime.strftime('5nmAgsphere_%Y%m%d_%H%M%S')}"

filename = f"5nmAgSph_EE{polsmultipliers[0]}MM{polsmultipliers[1]}EM{polsmultipliers[2]}_li{li}_lf{lf}_ppos{ppos}"

beamparameters = {"qz":qz,
                  "v":v,
                  "q_colector":qc,
                  "q_aperture":qa,
                  "lf":lf,
                  "li":li
                  }

specimenparameters = {"pos":ppos,
                      "polsmultipliers":polsmultipliers,
                      "shereradius":sphrad,
                      "omega_p":omega_p,
                      "gamma_p":gamma_p
                      }

calculattionparamters = {"xrange":xrange,
                         "yrange":yrange,
                         "numxpoints":numofxpoints,
                         "numypoints":numofypoints
                         }

metadata = {"filename":filename,
            "beam":beamparameters,
            "specimen":specimenparameters,
            "calculation":calculattionparamters,
            "starttime":starttime}

#%% Final calculation of spectrum

print("Initializing calculation of the spectrum...")
print(f"Spectrum will be saved to \n{filename}")
spec = spectrum.Spectrum(X, Y, X_, Y_, qz, v, ppos, 
                         PSI1, PSI2, lf, qc, 
                         Omegas, Alphas, 
                         #filename="AgSphere5nm_0_0_0"
                         )


#%% Save data
metadata["endtime"] = datetime.now()

np.savetxt(filename+".csv", spec , header=str(metadata)+"\nomega\tgamma")
np.save(filename, spec)
print(f"Spectrum was saved to {filename}")

#%%
# The following part is for debugging and control purposes and before evaluation should be commented
# X and Y for plots
Xplot = X[:,:,25,25] * 1e9
Yplot = Y[:,:,25,25] * 1e9
#%% initial psis plot

fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2,figsize=(5,5), sharex=True, sharey=True)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.pcolormesh(Xplot,Yplot, np.real(PSI1)[:,:, 25,25])
axim.pcolormesh(Xplot,Yplot, np.imag(PSI1)[:,:, 25,25])
axabs.pcolormesh(Xplot,Yplot, np.abs(PSI1)[:,:, 25,25])
axang.pcolormesh(Xplot,Yplot, np.angle(PSI1)[:,:, 25,25])

fig.subplots_adjust(0,0,1,1,0,0)
fig.savefig(f"mapy/map_psi_aEE{polsmultipliers[0]}aEM{polsmultipliers[2]}_li{li}_lf{lf}_xp{ppos[0]}.png",dpi=150)


#%% test of king and queen - prep
omega = Omegas[8]
aee, amm, aem = Alphas[8]

#G = spectrum.KingTensorFun(X,Y,X_,Y_, ppos, qz, v, omega, aee, amm, aem)
gradpsi1, gradpsi2 = spectrum.GradsOfPsi(PSI1, PSI2, qz, dx, dy)

#%% test of king and queen - plot of gradients
fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2,figsize=(5,5), sharex=True, sharey=True)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.pcolormesh(Xplot,Yplot, np.real(gradpsi1)[:,:, 25,25,0,0])
axim.pcolormesh(Xplot,Yplot, np.imag(gradpsi1)[:,:, 25,25,0,0])
axabs.pcolormesh(Xplot,Yplot, np.abs(gradpsi1)[:,:, 25,25,0,0])
axang.pcolormesh(Xplot,Yplot, np.angle(gradpsi1)[:,:, 25,25,0,0])

fig.subplots_adjust(0,0,1,1,0,0)
fig.savefig(f"mapy/map_gradpsi_aEE{polsmultipliers[0]}aEM{polsmultipliers[2]}_li{li}_lf{lf}_xp{ppos[0]}.png",dpi=150)

#%% test of king and queen - queen
qarg = spectrum.QueenFun(lf, X, Y, X_, Y_, qc)
#%% test of king and queen - plot of queen
fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2,figsize=(5,5), sharex=True, sharey=True)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.pcolormesh(Xplot,Yplot, np.real(qarg)[:,:,25,25])
axim.pcolormesh(Xplot,Yplot, np.imag(qarg)[:,:,25,25])
axabs.pcolormesh(Xplot,Yplot, np.abs(qarg)[:,:,25,25])
axang.pcolormesh(Xplot,Yplot, np.angle(qarg)[:,:,25,25])

fig.subplots_adjust(0,0,1,1,0,0)
fig.savefig(f"mapy/map_queen_aEE{polsmultipliers[0]}aEM{polsmultipliers[2]}_li{li}_lf{lf}_xp{ppos[0]}.png",dpi=150)

#%% test of king and queen - king manual
#karg = (gradpsi1 @ np.imag(G) @ gradpsi2 )[:,:,:,:,0,0]

#%% test of king and queen - king function
karg = spectrum.KingFun(X,Y,X_,Y_, ppos, qz, v, omega, aee, amm, aem, gradpsi1,gradpsi2)

#%% test of king and queen - plot of king
fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2,figsize=(5,5), sharex=True, sharey=True)

axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.pcolormesh(Xplot,Yplot, np.real(karg)[:,:,24,24])
axim.pcolormesh(Xplot,Yplot, np.imag(karg)[:,:,24,24])
axabs.pcolormesh(Xplot,Yplot, np.abs(karg)[:,:,24,24])
axang.pcolormesh(Xplot,Yplot, np.angle(karg)[:,:,24,24])
fig.tight_layout()

fig.subplots_adjust(0,0,1,1,0,0)
fig.savefig(f"mapy/map_king_aEE{polsmultipliers[0]}aEM{polsmultipliers[2]}_li{li}_lf{lf}_xp{ppos[0]}.png",dpi=150)

#%% test of king and queen - plot of argument
arg = karg*qarg

fig, ((axreal, axim), (axabs, axang)) = plt.subplots(2,2,figsize=(5,5), sharex=True, sharey=True)
axreal.set_aspect(1)
axim.set_aspect(1)
axabs.set_aspect(1)
axang.set_aspect(1)

axreal.pcolormesh(Xplot,Yplot, np.real(arg)[:,:,25,25])
axim.pcolormesh(Xplot,Yplot, np.imag(arg)[:,:,25,25])
axabs.pcolormesh(Xplot,Yplot, np.abs(arg)[:,:,25,25])
axang.pcolormesh(Xplot,Yplot, np.angle(arg)[:,:,25,25])

fig.subplots_adjust(0,0,1,1,0,0)
fig.savefig(f"mapy/map_integrand_aEE{polsmultipliers[0]}aEM{polsmultipliers[2]}_li{li}_lf{lf}_xp{ppos[0]}.png",dpi=150)



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
