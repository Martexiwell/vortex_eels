# -*- coding: utf-8 -*-
"""
Greens tensors from mathematica

@author: MrMai
"""

import numpy as np
#import scipy as scp
from scipy.special import kv as BesselK
import scipy.constants as const

Sqrt = lambda x : np.sqrt(x)
E = np.e
Pi = np.pi
I = 1j

def gee(x,y,z,x0,y0,z0,omega,k,v,c=const.c,epsilon=const.epsilon_0):
    gamma = 1 / np.sqrt(1-v**2/c**2)
    
    expression = np.array([[
                            (np.exp((I*omega*z0)/v)*
                                  (2*k**2*BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)) + 
                                    (2*omega*(x - x0)**2*
                                       BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                     (gamma*v*((x - x0)**2 + (y - y0)**2)**1.5) - 
                                    (2*omega*BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                     (gamma*v*Sqrt((x - x0)**2 + (y - y0)**2)) + 
                                    (omega**2*(x - x0)**2*
                                       (BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)) + 
                                         BesselK(2,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v))))/
                                     (gamma**2*v**2*(x**2 - 2*x*x0 + x0**2 + (y - y0)**2))))/
                                (4.*epsilon*Pi),
                                (np.exp((I*omega*z0)/v)*omega*(x - x0)*(y - y0)*
                                  (omega*Sqrt((x - x0)**2 + (y - y0)**2)*
                                     BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)) + 
                                    2*gamma*v*BesselK(1,
                                      (omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)) + 
                                    omega*Sqrt((x - x0)**2 + (y - y0)**2)*
                                     BesselK(2,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v))))/
                                (4.*epsilon*gamma**2*Pi*v**2*((x - x0)**2 + (y - y0)**2)**1.5),
                               (I/2*np.exp((I*omega*z0)/v)*omega**2*(x - x0)*
                                  BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (epsilon*gamma*Pi*v**2*Sqrt((x - x0)**2 + (y - y0)**2))
                            ],[    
                                (np.exp((I*omega*z0)/v)*omega*(x - x0)*(y - y0)*
                                  (omega*Sqrt((x - x0)**2 + (y - y0)**2)*
                                     BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)) + 
                                    2*gamma*v*BesselK(1,
                                      (omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)) + 
                                    omega*Sqrt((x - x0)**2 + (y - y0)**2)*
                                     BesselK(2,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v))))/
                                (4.*epsilon*gamma**2*Pi*v**2*((x - x0)**2 + (y - y0)**2)**1.5),
                               (np.exp((I*omega*z0)/v)*(2*k**2*
                                     BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)) - 
                                    (2*omega*BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                     (gamma*v*Sqrt((x - x0)**2 + (y - y0)**2)) + 
                                    (2*omega*(y - y0)**2*
                                       BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                     (gamma*v*((x - x0)**2 + (y - y0)**2)**1.5) + 
                                    (omega**2*(y - y0)**2*
                                       (BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)) + 
                                         BesselK(2,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v))))/
                                     (gamma**2*v**2*(x**2 - 2*x*x0 + x0**2 + (y - y0)**2))))/
                                (4.*epsilon*Pi),(I/2*np.exp((I*omega*z0)/v)*omega**2*(y - y0)*
                                  BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (epsilon*gamma*Pi*v**2*Sqrt((x - x0)**2 + (y - y0)**2))
                            ],[    
                                (I/2*np.exp((I*omega*z0)/v)*omega**2*(x - x0)*
                                  BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (epsilon*gamma*Pi*v**2*Sqrt((x - x0)**2 + (y - y0)**2)),
                               (I/2*np.exp((I*omega*z0)/v)*omega**2*(y - y0)*
                                  BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (epsilon*gamma*Pi*v**2*Sqrt((x - x0)**2 + (y - y0)**2)),
                               (np.exp((I*omega*z0)/v)*(-omega**2 + k**2*v**2)*
                                  BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (2.*epsilon*Pi*v**2)
                         ]])
    return np.moveaxis(expression, [0,1], [-2,-1]) 

def gem(x,y,z,x0,y0,z0,omega,k,v,c=const.c,epsilon=const.epsilon_0):
    gamma = 1 / np.sqrt(1-v**2/c**2)
    zero = np.zeros(np.shape(x))
    expression = np.array([[
                                zero,
                                -0.5*(np.exp((I*omega*z0)/v)*k*omega*
                                        BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*Pi*v),
                                (-I/2*np.exp((I*omega*z0)/v)*k*omega*(y - y0)*
                                                  BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*gamma*Pi*v*Sqrt((x - x0)**2 + (y - y0)**2))
                            ],[
                                (np.exp((I*omega*z0)/v)*k*omega*
                                 BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (2.*c*epsilon*Pi*v),
                                zero,
                                (I/2*np.exp((I*omega*z0)/v)*k*omega*(x - x0)*
                                                       BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*gamma*Pi*v*Sqrt((x - x0)**2 + (y - y0)**2))
                            ],[
                                (I/2*np.exp((I*omega*z0)/v)*k*omega*(y - y0)*
                                 BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*gamma*Pi*v*Sqrt((x - x0)**2 + (y - y0)**2)),
                                (-I/2*np.exp((I*omega*z0)/v)*k*omega*(x - x0)*
                                 BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*gamma*Pi*v*Sqrt((x - x0)**2 + (y - y0)**2)),
                                zero
                            ]])
    return np.moveaxis(expression, [0,1], [-2,-1]) 

def gme(x,y,z,x0,y0,z0,omega,k,v,c=const.c,epsilon=const.epsilon_0):
    gamma = 1 / np.sqrt(1-v**2/c**2)
    zero = np.zeros(np.shape(x0))
    expression = - np.array([[
                                zero,
                                -0.5*(np.exp((I*omega*z0)/v)*k*omega*
                                        BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*Pi*v),
                                (-I/2*np.exp((I*omega*z0)/v)*k*omega*(y - y0)*
                                                  BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*gamma*Pi*v*Sqrt((x - x0)**2 + (y - y0)**2))
                            ],[
                                (np.exp((I*omega*z0)/v)*k*omega*
                                 BesselK(0,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (2.*c*epsilon*Pi*v),
                                zero,
                                (I/2*np.exp((I*omega*z0)/v)*k*omega*(x - x0)*
                                                       BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*gamma*Pi*v*Sqrt((x - x0)**2 + (y - y0)**2))
                            ],[
                                (I/2*np.exp((I*omega*z0)/v)*k*omega*(y - y0)*
                                 BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*gamma*Pi*v*Sqrt((x - x0)**2 + (y - y0)**2)),
                                (-I/2*np.exp((I*omega*z0)/v)*k*omega*(x - x0)*
                                 BesselK(1,(omega*Sqrt((x - x0)**2 + (y - y0)**2))/(gamma*v)))/
                                (c*epsilon*gamma*Pi*v*Sqrt((x - x0)**2 + (y - y0)**2)),
                                zero
                            ]])
    return np.moveaxis(expression, [0,1], [-2,-1]) 