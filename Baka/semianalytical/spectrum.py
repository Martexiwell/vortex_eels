# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:07:00 2022

@author: Martin Ošmera

In this file, functions for calculation of EELspectrum will be stated
"""

#%% Import

import numpy as np
import scipy as scp
import scipy.constants as const
from scipy.integrate import quad
#from scipy.interpolate import interp2d
from scipy.interpolate import griddata
import sympy as sp
#import pickle
#import matplotlib.pyplot as plt

import king
#import psi_perp

#%% Queen

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
    rpairs = np.array([rr1,rr2]).T
    def integrator(rho1, rho2, qc=qc):
        vyraz = lambda k : k * scp.special.jv(l, k*rho1) * scp.special.jv(l, k*rho2)
        intvyraz = quad(vyraz, 0, qc, limit=numofqpoints)
        return intvyraz[0]
    vintegrator = np.vectorize(integrator)
    preeval = vintegrator(rr1,rr2)

    # interpfun = interp2d(rr1, rr2, preeval) # interpfun is a function for rr1 and rr2 as inputs with linear spline interpolation
    # queen = interpfun(R1,R2)
    
    queen = griddata(rpairs, preeval, (R1,R2), method="cubic")
    
    return queen * np.exp(1j * l * (Phi1 - Phi2)  )

#%% Gradient of psi

def GradsOfPsi(psi1, psi2, qz, dx, dy):
    print("Calculating gradients of inital psi ...", end="")
    Psi1 = np.conj( psi1)
    gradPsi1 = np.gradient(Psi1, axis=(0,1))
    gradPsi1[0] = gradPsi1[0]/dx
    gradPsi1[1] = gradPsi1[1]/dy
    gradPsi1.append(-1j * qz * Psi1)
    print("1/2 done ... ",end="")
    Psi2 =  psi2
    gradPsi2 = np.gradient(Psi2, axis=(2,3))
    gradPsi2[0] = gradPsi2[0]/dx
    gradPsi2[1] = gradPsi2[1]/dy
    gradPsi2.append(1j * qz * Psi2)
    print("2/2 DONE")
    
    return (gradPsi1, gradPsi2)
    

#%% King

def KingFun(X,Y,X_,Y_, theKing, grad1, grad2):
    x,y,z,x_,y_,z_ = sp.symbols("x y z x' y' z'")
    gv1x, gv1y, gv1z, gv2x, gv2y, gv2z = sp.symbols("gv1x gv1y gv1z gv2x gv2y gv2z")
    KFun =  sp.lambdify((x,y,z,x_,y_,z_, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z ),
                        theKing[0,0]) 
    v1x = grad1[0]
    v1y = grad1[1]
    v1z = grad1[2]
    v2x = grad2[0]
    v2y = grad2[1]
    v2z = grad2[2]
    
    finalKing = KFun(X,Y,0*X, X_,Y_,0*X_, v1x,v1y,v1z,v2x,v2y,v2z)
    
    return finalKing
    
#%% Gamma

def Gamma(omega, v, X, Y, X_, Y_, K, gradpsi1, gradpsi2, queen):
    prefactor = const.hbar * const.e**2 / ( 2 * np.pi**2 * const.m_e**2 )
    dx = X[1,0,0,0] - X[0,0,0,0]
    dy = Y[0,1,0,0] - Y[0,0,0,0]
    theking = KingFun(X,Y,X_,Y_,K,gradpsi1,gradpsi2)
    return prefactor / (omega**2 * v) * np.nansum(theking*queen) * dx**2 * dy**2    
    
#%% Spectrum

def Spectrum(X,Y,X_,Y_,
             qz, v, ppos, 
             psiinit1, psiinit2 ,
             greentensor,
             lfin, qc_forqueen, 
             omegas, polarizabilities, # iterable of 3-long iterables of form [aEE, aMM, aEM]
             SI = True, 
             numofqpoints_queen=100,
             filename=None):
    print("\nInitializing the calculation of spectrum")
    
    dx = X[1,0,0,0] - X[0,0,0,0]
    dy = Y[0,1,0,0] - Y[0,0,0,0]
    
    print("Calculating the Queen ... ", end="")
    theQueen = QueenFun(lfin, X,Y,X_,Y_, qc_forqueen, 
                        numofqpoints=numofqpoints_queen)
    print("DONE")
    
    gradpsi1, gradpsi2 = GradsOfPsi(psiinit1, psiinit2, qz, dx, dy)
    
    print("Preparing the Green's tensor ...",end="")
    Kgeo = king.setup(ppos, v)
    print("DONE")
    
    omegapoints = len(omegas)
    print(f"Beginning of evaluation for each of {omegapoints} omega points:\n",
          "omega \tgamma", sep="")
    gammas=[]
    
    for omegai in range(omegapoints):
        omega = omegas[omegai]
        aEE, aMM, aEM = polarizabilities[omegai]
        
        Komega = king.setpolarizability(Kgeo, aEE, aMM, aEM)
        Komega = king.setomega(Komega, omega)
        
        gamma = Gamma(omega, v, X, Y, X_, Y_, Komega, gradpsi1, gradpsi2, theQueen)
        gammas.append(gamma)
        
        print(f"{omega}\t{gamma}")
    
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