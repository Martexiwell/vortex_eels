# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:07:00 2022

@author: Martin Ošmera

In this file, functions for the king evaluation will be declared
"""

#%% Import
import sympy as sp
import pickle
from scipy import constants as const

#%% Constants and variables

c_ = sp.symbols('c')                     # speed of light
epsilon_ = sp.symbols(r'varepsilon')    # vacuum permititvity
x,y,z,x_,y_,z_,xp,yp,zp, omega_,v_, k_ = sp.symbols("x y z x' y' z' x_p y_p z_p omega v k")   
# position, primed position, position of specimen, freq. of photon, speed of electron, wavevenumber of photon
#gamma = 1/sp.sqrt( 1 - (v/c)**2 )
gamma_ = sp.symbols(r'gamma_L')     # Lorentz factor

iu = sp.I       # imaginary unit

alpha_EE = sp.MatrixSymbol("alphaEE", 3, 3)
alpha_MM = sp.MatrixSymbol("alphaMM", 3, 3)
alpha_EM = sp.MatrixSymbol("alphaEM", 3, 3)
alpha_ME = - alpha_EM.T

gv1x, gv1y, gv1z, gv2x, gv2y, gv2z = sp.symbols("gv1x gv1y gv1z gv2x gv2y gv2z")

#%% load the king

f = open("king_im.dat", "rb")
theKing = pickle.load(f)
f.close() 

#%% Setup functions

def setconstants(G, c=1, epsilon=1, SI=False):
    gamma = 1/sp.sqrt( 1 - (v_/c_)**2 )
    if SI:
        c = const.c
        epsilon = const.epsilon_0
    G = G.subs(gamma_,gamma)
    G = G.subs(k_, c/omega_)
    
    return G.subs([(c_,c),
                  (epsilon_, epsilon)])

def setparticlepos(G, ppos): 
    xpos, ypos, zpos = tuple(ppos)
    return G.subs([(xp, xpos),
                   (yp, ypos),
                   (zp, zpos)])

def setpolarizability(G, aEE, aMM, aEM):
    return  G.subs([( alpha_EE, sp.Matrix(aEE) ),
                    ( alpha_MM, sp.Matrix(aMM) ),
                    ( alpha_EM, sp.Matrix(aEM) )])

def setelectronparameters(G, v=1):
    return G.subs([(v_,v)])

def setpos(G, pos): 
    xpos, ypos, zpos = tuple(pos)
    return G.subs([(x, xpos),
                   (y, ypos),
                   (z, zpos)]) 

def setpos_(G, pos): 
    xpos, ypos, zpos = tuple(pos)
    return G.subs([(x_, xpos),
                   (y_, ypos),
                   (z_, zpos)])
     
def setomega(G, omega):
    return G.subs(omega_, omega)

def setup(ppos, v, SI=False):
    G = setconstants(theKing, SI=SI)
    G = setelectronparameters(G, v)
    G = setparticlepos(G, ppos)
    return G

def lamb(K):
    return sp.lambdify((x,y,z,x_,y_,z_, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z ),
                        K[0,0]) 
    