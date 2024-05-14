# -*- coding: utf-8 -*-
"""
This file is for learning 04Solvers of the tyssue package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image


from tyssue import config, Sheet, SheetGeometry, History, EventManager
from tyssue.draw import sheet_view
from tyssue.generation import three_faces_sheet
from tyssue.draw.plt_draw import plot_forces

from tyssue.dynamics import PlanarModel

from tyssue.solvers.viscous import EulerSolver
from tyssue.draw.plt_draw import create_gif


geom  = SheetGeometry
model = PlanarModel

sheet = Sheet.planar_sheet_3d('planar', nx=6, ny=6, 
                             distx=1, disty=1)
sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)


'''
The history object

The HIstory class defines the object in charge of stroing the evolving epithelium
during the course of the simulation. It allows to access to different time points
of a simulation from 1 unique epithelium.

Most of the time, we use HistoryHDF5 class, that writes each time step to a file,
which can be useful for big files. It is also possible to read an hf5 file to analyze
a simulation later.

In the solver, we use the history.record method to store the epithelium.
In the create_grif function, we use the history.retrieve method to get back the epithelium
at a given time point.

'''

history = History(sheet, save_every=2, dt=1)

for i in range(10):
    
    geom.scale(sheet, 1.02, list('xy'))
    geom.update_all(sheet)
    # record only every `save_every` time 
    history.record()

create_gif(history, 'simple_growth.gif', num_frames=len(history))


Image('simple_growth.gif')

# retrieve function returns an epithelium of the same type as the original.
type(history.retrieve(5))

# Iterate over an history object yields a time and a sheet object:
for t, sheet in history:
    print(f"mean area at {t}: {sheet.face_df.area.mean():.3f}", )

# The vert_h, edge_h and face_h Dataframes hold the history:
history.vert_h.head()

'''
We can plot the evolution of a column over time!!
'''
fig, ax = plt.subplots()

ax.scatter(history.face_h['time'], history.face_h['area'], alpha=0.2, s=12)

history.face_h.groupby('time').area.mean().plot(ax=ax)


'''
Quasistatic solver

A common way to describe an epithelium is with the quasistatic approximation:
we assume

'''


































'''
This is the end of the file. :)
'''
