# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 17:33:38 2022

@author: MrMai
"""

import numpy as np
from scipy.special import jv
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

#%%

l = 2
q = 2*np.pi

x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
z = np.linspace(0,3,100)

X,Y,Z = np.meshgrid(x,y,z)

R = (X**2 + Y**2)**0.5
PHI = np.arctan2(Y,X)

wave = np.exp(1j * q * Z) * np.exp(1j*l*PHI) * jv(l, q*R)

wavereal = np.real(wave)
#wavephase = np.angle(wave)

df= pd.DataFrame({'x':x, 'y':y})

#%%
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=wavereal.flatten(),
    opacity = 0.6,
    isomin=.28,
    isomax=.6,
    surface_count = 5,
    caps=dict(x_show=False, y_show=False, z_show = False),
    colorscale = 'Reds'
    ))
fig.show()