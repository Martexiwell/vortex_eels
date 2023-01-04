# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:52:24 2022

@author: MrMai
"""

import numpy as np
import scipy as scp
import scipy.constants as const
from scipy.integrate import quad
from scipy.interpolate import interp2d
import sympy as sp
#import pickle
#import matplotlib.pyplot as plt


import green
import psi_perp



# %% Queen

# def QueenFunPrep(l, r1, r2, phi1, phi2, qc, numofpoints=100):
#     vyraz = lambda k : k * scp.special.jv(l, k*r1) * scp.special.jv(l, k*r2)
#     intvyraz = scp.integrate.quad(vyraz, 0, qc, limit=numofpoints)
#     return intvyraz[0] * np.exp(1j * l * (phi1-phi2))
#     print("The Queen has been succefully evaluated!"   )   

# QueenFun = np.vectorize(QueenFunPrep)

# def QueenFun(l, R1, R2, Phi1, Phi2, qc, numofpoints=100):
#     rmax = max(np.max(R1), np.max(R2))
#     rspace1 = np.linspace(0,rmax)
#     Rspace1,Rspace2 = np.meshgrid(rspace1,rspace1)
#     preeval 

def QueenFun(l, X1,Y1,X2,Y2, qc, numofqpoints=50, numofrpoints=50): # version 2
    R1       = np.sqrt(X1**2 + Y1**2)
    Phi1     = np.arctan2(Y1,X1)
    R2      = np.sqrt(X2**2 + Y2**2)
    Phi2    = np.arctan2(Y2,X2)
    
    rmax = np.max(R1)
    
    r1 = np.linspace(0,rmax, numofrpoints)
    rr1, rr2 = np.meshgrid(r1,r1)
    rr1 = rr1.flatten()
    rr2 = rr2.flatten()
    def integrator(rho1, rho2, qc=qc):
        vyraz = lambda k : k * scp.special.jv(l, k*rho1) * scp.special.jv(l, k*rho2)
        intvyraz = quad(vyraz, 0, qc, limit=numofqpoints)
        return intvyraz[0]
    vintegrator = np.vectorize(integrator)
    preeval = vintegrator(rr1,rr2)

    interpfun = np.vectorize(interp2d(rr1,rr2, preeval) )
    queen = interpfun(R1,R2)
    
    return queen * np.exp(1j * l * (Phi1 - Phi2)  )
    

#%%
# This section is for debugging the Queen

# R1 = np.linspace(0,50,1000)
# R2 = np.copy(R1)

# R1,R2 = np.meshgrid(R1,R2)

# Z = np.real(QueenFun(1, R1, R2, 0,0, 1.0))

# fig,ax = plt.subplots()
# ax.pcolor(R1,R2,Z)
# ax.set_aspect(1)

# %% King generation

def KingFun(X, Y, X_, Y_, G, gradpsi1, gradpsi2 ):    
    print("Evaluating the king...")
    
    # print(G.atoms())
    
    x,y,z,x_,y_,z_ = sp.symbols("x y z x' y' z'")
    GFun =  sp.lambdify((x,y,z,x_,y_,z_),G, modules="scipy") 
    
    theGreat = GFun(X,Y,0*X, X_,Y_,0*X)
    
    theKing =  np.sum ( np.moveaxis(gradpsi1, 0, -2) * (theGreat @ np.moveaxis(gradpsi2, 0, -2)), axis=-2 )
    
    # theKing = np.copy(X).astype(complex)
    #shape = np.shape(X)
    # for i in range(shape[0]):
    #     for j in range(shape[1]):
    #         for k in range(shape[2]):
    #             for l in range(shape[3]):
    #                 pos  = [X[i,j,k,l], Y[i,j,k,l], 0]
    #                 pos_ = [X_[i,j,k,l], Y_[i,j,k,l], 0]
                    
    #                 vec1 = np.array([gradpsi1[0][i,j,k,l],
    #                                  gradpsi1[1][i,j,k,l],
    #                                  gradpsi1[2][i,j,k,l]])
                    
    #                 vec2 = np.array([gradpsi2[0][i,j,k,l],
    #                                  gradpsi2[1][i,j,k,l],
    #                                  gradpsi2[2][i,j,k,l]])
                    
    #                 Gtensor = GFun(*pos, *pos_)
                    
                    
    #                 scalar = vec1 @ Gtensor @ vec2
    #                 theKing[i,j,k,l] = scalar
    #    print(f"{round((i+1)/shape[0]*100)} %", end="\r")
    print("DONE"   )             
    return theKing



#%%
prefactor = const.hbar * const.e**2 / ( 2 * np.pi**2 * const.m_e**2 )

def Gamma(omega, v, X, Y, X_, Y_, G, gradpsi1, gradpsi2, queen):
    dx = X[1,0,0,0] - X[0,0,0,0]
    dy = Y[0,1,0,0] - Y[0,0,0,0]
    theking = KingFun(X,Y,X_,Y_,G,gradpsi1,gradpsi2)
    return prefactor / (omega**2 * v) * np.nansum(theking*queen) * dx**2 *dy**2


# %% Spectrum

def Spectrum(X,Y,X_,Y_,
             qz, v, ppos, 
             psiinit1, psiinit2 ,
             greentensor,
             lfin, qc_forqueen, 
             omegas, polarizabilities, # iterable of 3-long iterables of form [aEE, aMM, aEM]
             SI = True, numofpoints_forqueen=100,
             filename=None):
    
    R       = np.sqrt(X**2 + Y**2)
    Phi     = np.arctan2(Y,X)
    R_      = np.sqrt(X_**2 + Y_**2)
    Phi_    = np.arctan2(Y_,X_)
    
    dx = X[1,0,0,0] - X[0,0,0,0]
    dy = Y[0,1,0,0] - Y[0,0,0,0]
    
    print("Calculating the Queen...")
    theQueen = QueenFun(lfin, R, R_, Phi, Phi_, qc_forqueen, numofqpoints=numofpoints_forqueen)
    
    print("Calculating gradients of inital psi...")
    Psi1 = np.conj( psiinit1 )
    gradPsi1 = np.gradient(Psi1, axis=(0,1))
    gradPsi1[0] = gradPsi1[0]/dx
    gradPsi1[1] = gradPsi1[1]/dy
    gradPsi1.append(-1j * qz * Psi1)
    print("1/2 done")
    Psi2 =  psiinit2
    gradPsi2 = np.gradient(Psi2, axis=(2,3))
    gradPsi2[0] = gradPsi2[0]/dx
    gradPsi2[1] = gradPsi2[1]/dy
    gradPsi2.append(1j * qz * Psi2)
    print("2/2 DONE")
    
    print("Preparing the Green's tensor")
    GreenGeo = green.setup(ppos, qz, v, SI=SI)
    
    print("Beginning of evaluation for each omega...")
    omegapoints = len(omegas)
    gammas=[]
    omegacounter = 0
    
    for omegai in range(omegapoints):
        omega = omegas[omegai]
        aEE, aMM, aEM = polarizabilities[omegai]
        
        GreenOmega = green.setpolarizability(GreenGeo, aEE, aMM, aEM)
        GreenOmega = green.setomega(GreenOmega, omega)
        gamma =  Gamma(omega, v, X, Y, X_, Y_, GreenOmega, gradPsi1, gradPsi2, theQueen)
        gammas.append(gamma)
        omegacounter+=1
        print(f"Evaluation of spectrum integral is done in {omegacounter} / {omegapoints} energy-points.")
    
    spectrum = np.array([omegas,gammas]).T
    
    if type(filename) == str: 
        try:
            import pickle
            f = open(f"{filename}.dat", "wb")
            pickle.dump(spectrum, f)
            f.close()
            print(f"Spectrum saved to \t {filename}.dat \t.")
        except: 
            print("Couldn't save the spectrum into file, thus priniting it: \n")
            print(spectrum)
            
    return spectrum


#%% testing
# numofxpoints = 20
# numofypoints = 20
# xrange = [-20,20]
# yrange = [-20,20]
# dx = (xrange[1] - xrange[0])/(numofxpoints-1)
# dy = (yrange[1] - yrange[0])/(numofypoints-1)

# xx = np.linspace(xrange[0], xrange[1], numofxpoints)
# yy = np.linspace(yrange[0], yrange[1], numofypoints)

# XX, YY, XX_, YY_   = np.meshgrid(xx, yy, xx, yy, indexing="ij")

ppos = [0,0,0]
qz= np.sqrt(2 * const.m_e * 60e3)/const.hbar
v=0,446 * const.c
omega = 0.1


# lf=1
# li=1

# psifun = psi_perp.psi_perp
# R       = np.sqrt(XX**2 + YY**2)
# Phi     = np.arctan2(YY,XX)
# R_      = np.sqrt(XX_**2 + YY_**2)
# Phi_    = np.arctan2(YY_,XX_)
# PSI1 = psifun(li, R, Phi, 1)
# PSI2 = psifun(li, R_, Phi_, 1)

# qc = 1

# GG = green.setup(ppos, qz, v)

# Omegas = np.linspace(1,3,10)
# Alfas=[]
# for i in range(len(Omegas)):
#     aEE = np.eye(3)*.3
#     aMM = np.eye(3)*0.1
#     aEM = np.array([[1,0, np.sin(Omegas[i])],
#                     [0,2, np.cos(Omegas[i])],
#                     [np.sin(Omegas[i]), np.cos(Omegas[i]), 3]])*0.05
#     Alfas.append([aEE, aMM, aEM])
    
    
# Spectrum(XX, YY, XX_, YY_, qz, v, ppos, PSI1, PSI2, GG, lf, qc, Omegas, Alfas, filename="prvnitest")
# %%



GreenGeo = green.setup(ppos, qz, v)

# GreenOmega = green.setpolarizability(GreenGeo, np.eye(3), 
#                                                 np.eye(3), 
#                                                 np.array([[1,2,3],
#                                                           [2,2,0],
#                                                           [3,0,3]])*1e-5 )

# GreenOmega = green.setomega(GreenOmega, omega)

# %%
###


# XX,YY,XX_,YY_ --> r, r_, phi, phi_r=R

# r_=R_
# phi=Phi
# phi_=Phi_
# omega =1
# k = 1e6
# v=1e6
# G = GreenOmega


# Psi1 = np.conj( psifun(l, R, Phi, qc ) )
# gradPsi1 = np.gradient(Psi1, axis=(0,1))/(dx*dy)
# gradPsi1.append(-1j * qz * Psi1)

# Psi2 =  psifun(l, r_, phi_, qc ) 
# gradPsi2 = np.gradient(Psi2, axis=(2,3))
# gradPsi2.append(1j * qz * Psi2)

# %%
# theKing = KingFun(XX, YY, XX_, YY_, GreenOmega, gradPsi1, gradPsi2)



                
