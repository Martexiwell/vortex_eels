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
#import sympy as sp
#import pickle
import green

from datetime import datetime

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
    
    
    v1 = np.array([[gradPsi1[0],
                    gradPsi1[1],
                    gradPsi1[2]]])
    v2 = np.array([[gradPsi2[0]],
                   [gradPsi2[1]],
                   [gradPsi2[2]]])
    
    v1 = np.moveaxis(v1, (0,1), (-2,-1))
    v2 = np.moveaxis(v2, (0,1), (-2,-1))
    
    return (v1,v2)

#%%

def KingTensorFun(X,Y,X_,Y_, ppos, k, v, omega, aee, amm, aem):
    
    xp,yp,zp = ppos
    
    Gee1    = green.gee(X, Y, 0*X, xp,yp,zp, omega, k, v)
    Gee2    = green.gee(xp,yp,zp, X_,Y_,0*X_, omega, k, v)
    Gem     = green.gem(X, Y, 0*X, xp,yp,zp, omega, k, v)
    Gme     = green.gme(xp,yp,zp, X_,Y_,0*X_, omega, k, v)
     
    # multiply the whole tensor
    ame = -aem.T
    G = (Gee1 @ aee @ Gee2 +
         Gee1 @ aem @ Gme +
         Gem @ ame @ Gee2 +
         Gem @ amm @ Gme)
    
    return G

def KingFun(X,Y,X_,Y_, ppos, k, v, omega, aee, amm, aem, grad1,grad2):
    G = KingTensorFun(X,Y,X_,Y_, ppos, k, v, omega, aee, amm, aem)
    K = (grad1 @ (G) @ grad2 )[:,:,:,:,0,0]
    return K
    
    
#%% Spectrum

def Spectrum(X,Y,X_,Y_,
             qz, v, 
             ppos, 
             psiinit1, psiinit2 ,
             lfin, qc_queen, 
             omegas, polarizabilities, # iterable of 3-long iterables of form [aEE, aMM, aEM]
             SI = True, 
             numofqpoints_queen=100,
             filename=None):
    print("\nInitializing the calculation of spectrum")
    
    dx = X[1,0,0,0] - X[0,0,0,0]
    dy = Y[0,1,0,0] - Y[0,0,0,0]
    #prefactor for integral (Jack)
    prefactor = const.hbar * const.e**2 / ( 2 * np.pi**2 * const.m_e**2 )
    
    
    print("Calculating the Queen ... ", end="")
    theQueen = QueenFun(lfin, X,Y,X_,Y_, qc_queen, 
                        numofqpoints=numofqpoints_queen)
    print("DONE")
    
    #Gradients
    
    grad1, grad2 = GradsOfPsi(psiinit1, psiinit2, qz, dx, dy)
    
    omegapoints = len(omegas)
    print(f"Beginning of evaluation for each of {omegapoints} omega points:\n",
          "date_and_time \tomega \tgamma", sep="")
    gammas=[]
    
    for omegai in range(omegapoints):
        omega = omegas[omegai]
        aee, amm, aem = polarizabilities[omegai]
        
        theKing = KingFun(X,Y,X_,Y_, ppos, qz, v, omega, aee, amm, aem, grad1,grad2)
        
        gamma = np.nansum(theKing * theQueen) * (dx**2 * dy**2)
        gamma = prefactor / (omega**2 * v) * gamma
        gamma = np.imag(gamma)      # imaginarni cast z celeho integralu
        gammas.append(gamma)
        
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}:\t{omega}\t{gamma}")
    
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