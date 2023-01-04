# -*- coding: utf-8 -*-
"""
numericsprocessor

@author: MrMai
"""

#%% import

import numpy as np
from scipy.special import kn

import scipy.constants as const
# import sympy as sp
# from scipy.special import jv 
# from datetime import datetime
# starttime = datetime.now()

## My >>toolboxes<<
#import spectrum
#import green
import wave_functions
from tools import read_matlab_data

import matplotlib.pyplot as plt
import matplotlib.colors as colors
inchtocm = 2.54
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Computer Modern Roman'],
    #"figure.figsize" : [8/inchtocm, 8/inchtocm]    
    })
# %matplotlib auto       # interactive plots
# %matplotlib inline     # non-interactive plots


#%% read data

eigencharges = read_matlab_data("Data/eigencharges")
eigenpseudolambdas = np.diag(np.loadtxt("Data/eigenpseudolambdas.csv", delimiter=","))
pfacepos = read_matlab_data("Data/pfacepos")
pfacearea = read_matlab_data("Data/pfacearea")

numofpoints, numofstates  = np.shape(eigencharges) 

#%%

#%% Arrays
xm = 40e-9
ym = xm

print("Preparing input arrays ... ", end="")
numofxpoints = 70
numofypoints = 70
xrange = [-xm,xm]
yrange = [-ym,ym]
dx = (xrange[1] - xrange[0])/(numofxpoints-1)
dy = (yrange[1] - yrange[0])/(numofypoints-1)

xx = np.linspace(xrange[0], xrange[1], numofxpoints)
yy = np.linspace(yrange[0], yrange[1], numofypoints)

X, Y   = np.meshgrid(xx, yy, indexing="ij")
print("DONE")
#%% parameters
print("Preparing parameters ... ", end="")
qz = np.sqrt(2 * const.m_e * 60e3 * const.eV) / const.h
v = 0.446 * const.c



print("DONE")

#%%
XX = np.array([X] * numofpoints)
YY = np.array([Y] * numofpoints)
sx = np.moveaxis(np.array([[pfacepos[:,0]]]), 2, 0).astype(float) * 1e-9
sy = np.moveaxis(np.array([[pfacepos[:,1]]]), 2, 0).astype(float) * 1e-9
sz =  pfacepos[:,2] * 1e-9
#%%plot the particle
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(sx,sy,sz)

#%%
#%% P E R M I I V I T Y
def dielectricfun(omega, omega_p, gamma, epsilon_0=1):
    return epsilon_0 * ( 1 - omega_p**2 / (omega**2 + 1j * gamma * omega) )

#%% Omegas and alphas

omegas = np.linspace(.6,1,50) *1e16 #np.linspace(3,9,50) * const.eV/const.hbar
omega_p = 9.1 * const.eV/const.hbar
gamma_p = 0.15 * const.eV/const.hbar
epsilonko = dielectricfun(omegas, omega_p, gamma_p)

#%% P O T E N T I A L

omega = .33e16
potentials = [] 
for n in range(numofstates):
    potential = (
                2 * pfacearea * eigencharges[:,n] * np.exp(-1j * omega / v * sz) @
                np.moveaxis(kn(0, omega / v * np.sqrt((XX-sx)**2 + (YY-sy)**2) ), 0, 1)
                )
    potentials.append(potential)
    print(f"potential number {n}: DONE")
potentials = np.array(potentials)
#%% vykresleni vlastnich modu real

for i in range(numofstates-2):
    fig, ax = plt.subplots()
    ax.set_title(f"Re$[\,\Phi_{{{i+1}}} ( \omega = {omega/1e15} \cdot 10^{{15}} \, \mathrm{{rad\cdot s^{{-1}}}}  )]$")
    ax1 = ax.pcolormesh(X*1e9,Y*1e9,np.real(potentials[i+2]), norm=colors.CenteredNorm(), cmap="RdBu_r")
    ax.set_aspect(1)
    plt.colorbar(ax1)
    ax.set(xlabel = "$x \, / \, \mathrm{nm}$",
           ylabel = "$y \, / \, \mathrm{nm}$")
    fig.savefig(f"potentials/Phi_{i}_re.png", dpi=150)
    
#%% vykresleni vlastnich modu imag

for i in range(numofstates-2):
    fig, ax = plt.subplots()
    ax.set_title(f"Im$[\,\Phi_{{{i+1}}} ( \omega = {omega/1e15} \cdot 10^{{15}} \, \mathrm{{rad\cdot s^{{-1}}}}  )]$")
    ax1 = ax.pcolormesh(X*1e9,Y*1e9,np.imag(potentials[i+2]), norm=colors.CenteredNorm(), cmap="PiYG_r")
    ax.set_aspect(1)
    plt.colorbar(ax1)
    ax.set(xlabel = "$x \, / \, \mathrm{nm}$",
           ylabel = "$y \, / \, \mathrm{nm}$")
    fig.savefig(f"potentials/Phi_{i}_im.png", dpi=150)
    
#%% vykresleni vlastnich modu abs

for i in range(numofstates-2):
    fig, ax = plt.subplots()
    ax.set_title(f"$| \phi_{{{i+1}}} |$")
    ax1 = ax.pcolormesh(X,Y,np.abs(potentials[i+2]), cmap="inferno")
    ax.set_aspect(1)
    plt.colorbar(ax1)
    
#%% vykresleni vlastnich modu 
for i in range(numofstates-2):
    fig, (axre, axim) = plt.subplots(1,2)
    
    axre.set_title(f"Re$[\,\Phi_{{{i+1}}} ( \omega = {omega/1e15} \cdot 10^{{15}} \, \mathrm{{rad\cdot s^{{-1}}}}  )]$")
    ax1 = axre.pcolormesh(X*1e9,Y*1e9,np.real(potentials[i+2]), norm=colors.CenteredNorm(), cmap="RdBu_r")
    axre.set_aspect(1)
    plt.colorbar(ax1)
    axre.set(xlabel = "$x \, / \, \mathrm{nm}$",
           ylabel = "$y \, / \, \mathrm{nm}$")
    
    axim.set_title(f"Im$[\,\Phi_{{{i+1}}} ( \omega = {omega/1e15} \cdot 10^{{15}} \, \mathrm{{rad\cdot s^{{-1}}}}  )]$")
    ax2 = axim.pcolormesh(X*1e9,Y*1e9,np.imag(potentials[i+2]), norm=colors.CenteredNorm(), cmap="PiYG_r")
    axim.set_aspect(1)
    plt.colorbar(ax2)
    axim.set(xlabel = "$x \, / \, \mathrm{nm}$",
           ylabel = "$y \, / \, \mathrm{nm}$")
    

    fig.savefig(f"potentials/Phi_{i}.png", dpi=150)

#%% nalezeni gfactors

# epsilon = dielectricfun(omega, omega_p, gamma_p)
# eigenlambdas = eigenpseudolambdas / 2 / np.pi
# gfactors = np.imag(-2 / (epsilon * (1 + eigenlambdas) + (1 - eigenlambdas) ) )

#%% O M E G A S   A   G F A C T O R S
omegas = np.linspace(.3,0.55,151) *1e16
epsilons = dielectricfun(omegas, omega_p, gamma_p)
eigenlambdas = eigenpseudolambdas / 2 / np.pi
gfactorss = np.imag(-2 / (np.array([epsilons]).T @ np.array([(1 + eigenlambdas)]) + (1 - eigenlambdas) ) )

#%%
nmax = 2
preskoc=2
gfactors = gfactorss[:,preskoc:preskoc+nmax].T
#%%
labels = (np.array(["$n = "]*(nmax), dtype=np.object) 
          + (np.array(range(nmax+1)[1:])).astype(str).astype(np.object) 
          + np.array([" $"]*(nmax), dtype=np.object)
          )
fig, ax = plt.subplots()
ax.plot(omegas, gfactors.T, label = labels)
ax.legend(loc = "center left")
ax.grid()
ax.set_xlim(omegas[0], omegas[-1])
ax.set(xlabel = "$\omega \, / \, \mathrm{rad\cdot s^{-1}}$",
       ylabel = "Im$[-g_n(\omega)]$")

fig.savefig("gfactors.png", dpi=150)

#%% P O T E N T I A L S

potentials = [] 
for n in range(numofstates)[preskoc:preskoc+nmax]:
    potential=[]
    for omega in omegas:
        potentialek = (
                    2 * pfacearea * eigencharges[:,n] * np.exp(-1j * omega / v * sz) @
                    np.moveaxis(kn(0, omega / v * np.sqrt((XX-sx)**2 + (YY-sy)**2) ), 0, 1)
                    )
        potential.append(potentialek)
    potentials.append(potential)
    print(f"potential number {n-preskoc+1}: DONE")
potentials = np.array(potentials)

#%%
np.save("potentials.npy", potentials)
#%% W A V E F U N C T I O N S

xc = 0e-9
yc = 0e-9

qc = qz * 3e-3     # cutoff q for queen (detector)
qa = qz * 3e-3      # cutoff q for initial psi (aperture)

lf=+1
li=+1

print("Preparing the initial psis ... ", end="")
psii = np.array([[wave_functions.psiperp(li, X - xc, Y - yc, qa)]])
psif = np.array([[wave_functions.psiperp(li, X - xc, Y - yc, qc)]])
print("DONE")

#%%
fig, (axabs, axphase) = plt.subplots(1,2,sharey=True)
fig.set_size_inches(18/inchtocm, 10/inchtocm)

axabs.set_title("$\mathrm{Abs}\,( \psi )$")
ax1 = axabs.pcolormesh(X*1e9,Y*1e9,np.abs(psii[0,0,:,:]), cmap="viridis")
axabs.set_aspect(1)
plt.colorbar(ax1, ax=axabs, fraction=0.046, pad=0.035)
axabs.set(xlabel = "$x \, / \, \mathrm{nm}$",
       ylabel = "$y \, / \, \mathrm{nm}$")

axphase.set_title("$\mathrm{Arg}\,( \psi )$")
ax2 = axphase.pcolormesh(X*1e9,Y*1e9,np.angle(psii[0,0,:,:]), cmap="hsv")
axphase.set_aspect(1)
plt.colorbar(ax2, ax=axphase, fraction=0.046, pad=0.035)
axphase.set(xlabel = "$x \, / \, \mathrm{nm}$",)


fig.savefig("Psi.png", dpi=150)

#%% S P E C T R U M
integrandek = psii * psif  * potentials
plt.imshow(np.abs(integrandek[0,20,:,:]))
integralek = np.sum( integrandek, axis = (2,3) ) * dx * dy
gammas = np.sum(  gfactors * np.abs(integralek)**2   , axis=0)

spectrum = np.array([omegas,gammas]).T
#%% S A V E
metadata = {'li': li,
            'lf': lf,
            'xc': xc,
            'yc': yc
            }
np.savetxt(f"Data/spectrum_li{li}_lf{lf}_xc{xc}_yc{yc}.csv", spectrum, header=str(metadata)+"\nomega\tgamma")

#%%

def read_matlab_data( filename):
    csvpath =  filename
    data = np.loadtxt(csvpath, dtype = float)
    f = open(csvpath, "r")
    txt = f.readline()[2:].replace("'","\"")#.replace("datetime.datetime","")
    metadata = eval(txt)
    f.close()

    return data, metadata

#%%
fig,ax = plt.subplots()
figdich,axdich = plt.subplots()
#linestyles = ["-", (0,(5,1)), (0,(3,1)), (0,(3,1,1,1)), (0,(3,1,1,1,1,1)), (0,(3,1,1,1,1,1,1))]
xcs = ["-2e-08", "-1e-08", "0.0", "1e-08", "2e-08", "2.5e-08", "3e-08"]
markers = np.array(["$"]*7,dtype=object)+np.array(range(7)).astype(str).astype(object)+np.array(["$"]*7,dtype=object)

for i in range(7):
    xc = xcs[i]
    fname1 = f"Data/fullspectrum_li1_lf1_xc{xc}_yc{xc}.csv"
    data1, metadata1 = read_matlab_data(fname1)
    omegas1 = data1[:,0]
    gammas1 = data1[:,1]
    
    fname2 = f"Data/fullspectrum_li-1_lf-1_xc{xc}_yc{xc}.csv"
    data2, metadata2 = read_matlab_data(fname2)
    omegas2 = data2[:,0]
    gammas2 = data2[:,1]
    
    ax.plot(omegas1, gammas1, 
            color="red", 
            label=f"$x_c = y_c = {round(float(xc)*1e9)}$ nm",
            marker=markers[i],
            markevery=5,
            mec=(0,0,0))
    ax.plot(omegas2, gammas2, color="blue", 
            label=f"$x_c = y_c = {round(float(xc)*1e9)}$ nm",
            marker=markers[i],
            markevery=5,
            mec=(0,0,0))
    
    dichroism = (gammas1-gammas2)/(gammas1+gammas2)
    axdich.plot(omegas1, dichroism, 
                label=f"$x_c = y_c = {round(float(xc)*1e9)}$ nm")
        
ax.legend()
axdich.legend()
axdich.grid()
axdich.set_xlim(omegas1[0],omegas1[-1])





#%%
fig,ax = plt.subplots()
figdich,axdich = plt.subplots()
#linestyles = ["-", (0,(5,1)), (0,(3,1)), (0,(3,1,1,1)), (0,(3,1,1,1,1,1)), (0,(3,1,1,1,1,1,1))]
xcs = ["-2e-08", "-1e-08", "0.0", "1e-08", "2e-08", "2.5e-08", "3e-08"]
markers = np.array(["$"]*7,dtype=object)+np.array(range(7)).astype(str).astype(object)+np.array(["$"]*7,dtype=object)

for i in range(7):
    xc = xcs[i]
    fname1 = f"Data/fullspectrum_li1_lf1_xc{xc}_yc{xc}.csv"
    data1, metadata1 = read_matlab_data(fname1)
    omegas1 = data1[:,0]
    gammas1 = data1[:,1]
    
    fname2 = f"Data/fullspectrum_li-1_lf-1_xc{xc}_yc{xc}.csv"
    data2, metadata2 = read_matlab_data(fname2)
    omegas2 = data2[:,0]
    gammas2 = data2[:,1]
    
    ax.plot(omegas1, gammas1, 
            color="red", 
            label=f"$x_c = y_c = {round(float(xc)*1e9)}$ nm",
            marker=markers[i],
            markevery=5,
            mec=(0,0,0))
    ax.plot(omegas2, gammas2, color="blue", 
            label=f"$x_c = y_c = {round(float(xc)*1e9)}$ nm",
            marker=markers[i],
            markevery=5,
            mec=(0,0,0))
    
    dichroism = (gammas1-gammas2)/(gammas1+gammas2)
    axdich.plot(omegas1, dichroism, 
                label=f"$x_c = y_c = {round(float(xc)*1e9)}$ nm")
        
ax.legend()
axdich.legend()
axdich.grid()
axdich.set_xlim(omegas1[0],omegas1[-1])


#%%
xcs = ["-2e-08", "-1e-08", "0.0", "1e-08", "2e-08", "2.5e-08", "3e-08"]
markers = np.array(["$"]*7,dtype=object)+np.array(range(7)).astype(str).astype(object)+np.array(["$"]*7,dtype=object)
for i in range(7):
    xc = xcs[i]
    fname1 = f"Data/fullspectrum_li1_lf1_xc{xc}_yc{xc}.csv"
    data1, metadata1 = read_matlab_data(fname1)
    omegas1 = data1[:,0]
    gammas1 = data1[:,1]
    
    fname2 = f"Data/fullspectrum_li-1_lf-1_xc{xc}_yc{xc}.csv"
    data2, metadata2 = read_matlab_data(fname2)
    omegas2 = data2[:,0]
    gammas2 = data2[:,1]
    xlims = (omegas1[0],omegas1[-1])
    
    fig, (ax1) = plt.subplots(1,1)
    ax2 = ax1.twinx()
    
    ax1.plot(omegas1, gammas1, 
            color="red", 
            label="$l = +1$",
            #marker=markers[i],
            #markevery=5,
            #mec=(0,0,0)
            )
    ax1.plot(omegas2, gammas2, 
            color="blue", 
            label="$l = -1$",
            #marker=markers[i],
            #markevery=5,
            #mec=(0,0,0)
            )
    
    dichroism = (gammas1-gammas2)/(gammas1+gammas2) 
    ax2.plot(omegas1, dichroism, 
             color="k",
             #label=f"$x_c = y_c = {round(float(xc)*1e9)}$ nm"
             )
    
    ax1.grid()
    ax1.legend(loc="upper left")
    #ax2.legend(loc = "upper right")
    
    ax1.set_ylabel("$\Gamma(\omega)$ / arb. u.", color="purple")
    ax2.set_ylabel("$D(\omega)$", color="k")
    ax1.set_xlabel("$\omega \ / \mathrm{rad \cdot s^{-1}}$")
    ax1.tick_params(axis='y', labelcolor="purple")
    ax1.spines['left'].set_color('purple')
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.spines['left'].set_color('purple')
    ax1.set_xlim(xlims)
    
    
    ax1.set_title(f"$x_c = y_c = {round(float(xc)*1e9)}$ nm")
    fig.set_figwidth(17/inchtocm)
    fig.show()
    fig.savefig(f"spectra/EELSF_xc{xc}_w.png", 
                dpi = 150, 
                bbox_inches='tight', 
                #transparent=True
                )