# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:07:00 2022

@author: Martin Ošmera

In this file, the Green response tensor will be generated in sympy and 
functions for its transition into numpy will be stated
"""

#%% Import
import sympy as sp

#%% Constants and variables

c = sp.symbols('c')                     # speed of light
epsilon = sp.symbols(r'\varepsilon')    # vacuum permititvity
x,y,z,x_,y_,z_,xp,yp,zp,omega,v, k = sp.symbols("x y z x' y' z' x_p y_p z_p w v k")   
# position, primed position, position of specimen, freq. of photon, speed of electron, wavevector of photon
gamma = 1/sp.sqrt( 1 - (v/c)**2 )
gamma = sp.symbols(r'\gamma_{L}')   # Lorentz factor

iu = sp.I       # imaginary unit

#%%
a