# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 08:04:21 2022

@author: MrMai
"""

import sympy as sp
import pickle
import numpy as np


#%%

c = sp.symbols('c')                     # speed of light
epsilon = sp.symbols(r'varepsilon')    # vacuum permititvity
x,y,z,x_,y_,z_,xp,yp,zp,omega,v, k = sp.symbols("x y z x' y' z' x_p y_p z_p omega v k")   
# position, primed position, position of specimen, freq. of photon, speed of electron, wavevenumber of photon
gamma = 1/sp.sqrt( 1 - (v/c)**2 )
gamma = sp.symbols(r'gamma')   # Lorentz factor

iu = sp.I   

#%%

f = open("intgee.dat","rb")

GEE = pickle.load(f)

f.close()

#%%

Gee = GEE.subs([(c,3e8),
              (v,1e6),
              (k,1e9),
              (gamma,1),
              (epsilon,1)])

Gee1 = Gee.subs([(xp, 0),
                (yp, 0),
                (zp, 0)])

Geefun1 = sp.lambdify([omega,x,y,z], Gee1)

Gee2 = Gee.subs([(x, 0),
                (y, 0),
                (z, 0)])

Geefun2 = sp.lambdify([omega,xp,yp,zp], Gee2)

#%%
w = 1e12
X,Y,X_,Y_ = np.meshgrid(np.linspace(-1e-6,1e-6,6),np.linspace(-1e-6,1e-6,6),np.linspace(-1e-6,1e-6,6),np.linspace(-1e-6,1e-6,6),indexing="ij")

gs1 = Geefun1(w,X,Y,0*X)
gs1 = np.moveaxis(gs1, (0,1), (-2,-1))

gs2 = Geefun2(w,X_,Y_,0*X)
gs2 = np.moveaxis(gs2, (0,1), (-2,-1))

#%% 
alpha = np.random.rand(3,3)
gg = gs1 @ alpha @ gs2

def isalmostreal(arr, prec):
    return np.imag(arr) / np.real(arr) < prec

#%%
#%%
Gee1 = intgee

Gee2 = intgee
Gee2 = Gee2.subs([(xp, x_),
                  (yp, y_),
                  (zp, z_)])
Gee2 = Gee2.subs([(x, xp),
                  (y, yp),
                  (z, zp)])

Gem = intgem
Gme = -Gem
Gme = Gme.subs([(xp, x_),
                (yp, y_),
                (zp, z_)])
Gme = Gme.subs([(x, xp),
                (y, yp),
                (z, zp)])

#%%
# Substitute for the position of the polarizable particle
Gee1 = Gee1.subs([(xp, ppos[0]),
                  (yp, ppos[1]),
                  (zp, ppos[2])])
Gee2 = Gee2.subs([(xp, ppos[0]),
                  (yp, ppos[1]),
                  (zp, ppos[2])])
Gem = Gem.subs([(xp, ppos[0]),
                (yp, ppos[1]),
                (zp, ppos[2])])
Gme = Gme.subs([(xp, ppos[0]),
                (yp, ppos[1]),
                (zp, ppos[2])])

#%%
#Create functions
Gee1fun = sp.lambdify([x,y,z,x_,y_,z_], Gee1)
Gee2fun = sp.lambdify([x,y,z,x_,y_,z_], Gee2)
Gemfun = sp.lambdify([x,y,z,x_,y_,z_], Gem)
Gmefun = sp.lambdify([x,y,z,x_,y_,z_], Gme)

#%%
#Create numpy tensors

GEE1 = Gee1fun(X,Y,0*X,X_,Y_,0*X_)
GEE1 = np.moveaxis(GEE1, (0,1), (-2,-1))

#%%

GEE2 = Gee2fun(X,Y,0*X,X_,Y_,0*X_)
GEE2 = np.moveaxis(GEE2, (0,1), (-2,-1))

#%%
GEM = Gemfun(X,Y,0*X,X_,Y_,0*X_)
GEM = np.moveaxis(GEM, (0,1), (-2,-1))

GME = Gmefun(X,Y,0*X,X_,Y_,0*X_)
GME = np.moveaxis(GME, (0,1), (-2,-1))

#%%
# multiply the whole tensor
ame = -aem
G = (GEE1 @ aee @ GEE2 +
     GEE1 @ aem @ GME +
     GEM @ ame @ GEE2 +
     GEM @ amm @ GME)