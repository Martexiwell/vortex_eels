# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:44:18 2022

@author: MrMai
"""

import numpy as np
import scipy as scp
import scipy.constants as const
import sympy as sp
import matplotlib.pyplot as plt
import spectrum
import king
import psi_perp
from scipy.special import jv 
#%%

#%%


print("Preparing input arrays ... ", end="")
numofxpoints = 100
numofypoints = 100
xrange = [-1e-8,1e-8]
yrange = [-1e-8,1e-8]
dx = (xrange[1] - xrange[0])/(numofxpoints-1)
dy = (yrange[1] - yrange[0])/(numofypoints-1)

xx = np.linspace(xrange[0], xrange[1], numofxpoints)
yy = np.linspace(yrange[0], yrange[1], numofypoints)

X, Y, X_, Y_   = np.meshgrid(xx, yy, xx, yy, indexing="ij")

ppos = [0,0,0]
qz = np.sqrt(2 * const.m_e * 60e3 * const.eV) / const.h
v = 0.446 * const.c

lf=1
li=1
print("DONE")

print("Preparing the initial psis ... ", end="")
#psifun = psi_perp.psi_perp
R       = np.sqrt(X**2 + Y**2)
Phi     = np.arctan2(Y,X)
R_      = np.sqrt(X_**2 + Y_**2)
Phi_    = np.arctan2(Y_,X_)
Psi1 = jv(li, qz * R) * np.exp(1j*li*Phi) #psifun(li, R, Phi, 1)
Psi2 = jv(li, qz * R_) * np.exp(1j*li*Phi_)#psifun(li, R_, Phi_, 1)
print("DONE")
qc = 1

#%%
grad1 = np.gradient(Psi1, axis=(0,1))
grad1[0] = grad1[0]/dx
grad1[1] = grad1[1]/dy
grad1.append(-1j * qz * Psi1)

grad2 = np.gradient(Psi2, axis=(2,3))
grad2[0] = grad2[0]/dx
grad2[1] = grad2[1]/dy
grad2.append(-1j * qz * Psi2)

#%%
plt.imshow(np.abs(grad1)[0][:,:,0,0])
