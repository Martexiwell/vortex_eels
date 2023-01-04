# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:07:00 2022

@author: Martin Ošmera

In this file, the Green response tensor will be generated in sympy. 
It will be multiplied by gradientvectors and dumped into file for further use.
"""

#%% Import
import sympy as sp
import pickle

#%% Constants and variables

c = sp.symbols('c')                     # speed of light
epsilon = sp.symbols(r'varepsilon')    # vacuum permititvity
x,y,z,x_,y_,z_,xp,yp,zp,omega,v, k = sp.symbols("x y z x' y' z' x_p y_p z_p omega v k")   
# position, primed position, position of specimen, freq. of photon, speed of electron, wavevenumber of photon
gamma = 1/sp.sqrt( 1 - (v/c)**2 )
gamma = sp.symbols(r'gamma_L')   # Lorentz factor

iu = sp.I       # imaginary unit

#%% Integrated vacuum green function
vacuumGreenInt = sp.exp(iu * omega * zp/v) * sp.besselk(0, omega * sp.sqrt( (x-xp)**2 + (y-yp)**2 ) / (v * gamma))

#%% GreenEE
GreenEE_xx = sp.diff(vacuumGreenInt, xp, xp)
GreenEE_xy = sp.diff(vacuumGreenInt, xp, yp)
GreenEE_xz = sp.diff(vacuumGreenInt, xp, zp)
GreenEE_yy = sp.diff(vacuumGreenInt, yp, yp)
GreenEE_yz = sp.diff(vacuumGreenInt, yp, zp)
GreenEE_zz = sp.diff(vacuumGreenInt, zp, zp)

matice = sp.Matrix([[GreenEE_xx, GreenEE_xy, GreenEE_xz],
                    [GreenEE_xy, GreenEE_yy, GreenEE_yz],
                    [GreenEE_xz, GreenEE_yz, GreenEE_zz]])

eye = sp.eye(3)

matice = matice + k**2 * eye * vacuumGreenInt

matice = 1/(2 * sp.pi * epsilon) * matice

GreenEE = matice

#%% GreenEM

def LeviCivita(i,j,k):
    if (i,j,k) in ((1,2,0),(2,0,1),(0,1,2)):
        output = 1
    elif (i,j,k) in ((1,0,2),(0,2,1),(2,1,0)):
        output = -1
    else:
        output = 0
    return output

xyzp = (xp,yp,zp)
ijk = range(3)

matice = sp.zeros(3)
for i in ijk:
    for l in ijk:
        for j in ijk:
            for k in ijk:
                matice[i,l] += LeviCivita(i,j,k) * sp.diff(GreenEE[k,l], xyzp[j])
 

GreenEM = 1/(iu * k * c) * matice
GreenEM = sp.simplify(GreenEM)

#%% SuperTensor
alpha_EE = sp.MatrixSymbol("alphaEE", 3, 3)
alpha_MM = sp.MatrixSymbol("alphaMM", 3, 3)
alpha_EM = sp.MatrixSymbol("alphaEM", 3, 3)
alpha_ME = - alpha_EM.T

superAlpha =    sp.BlockMatrix([[ alpha_EE, alpha_EM ],
                [ alpha_ME, alpha_MM ]]) 

superGreen = sp.BlockMatrix([[GreenEE, GreenEM],
              [-GreenEM, GreenEE]])

superGreen1 = superGreen
superGreen2 = superGreen.subs([(xp, x_),
                               (yp, y_),
                               (zp, z_)])
superGreen2 = superGreen2.subs([(x, xp),
                                (y, yp),
                                (z, zp)])

finalGreen = superGreen1 * superAlpha * superGreen2

finalGreen = superGreen1 * sp.im(superAlpha * superGreen2)

finalGreenEE =  finalGreen[0:3,0:3]

#finalGreenEE = sp.im(finalGreenEE)

#%% gradients

gv1x, gv1y, gv1z, gv2x, gv2y, gv2z = sp.symbols("gv1x gv1y gv1z gv2x gv2y gv2z")

gv1 = sp.Matrix([[gv1x, gv1y, gv1z]])
gv2 = sp.Matrix([[gv2x],
                 [gv2y],
                 [gv2z]])

#%% the King

king = gv1 * finalGreenEE * gv2
#king = king.simplify()

#%% pickle

f = open("king.dat","wb")
pickle.dump(king, f)
f.close()