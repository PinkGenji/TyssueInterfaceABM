# -*- coding: utf-8 -*-
"""
This script where I implement any trial actions onto the bilayer structure.
"""


# =============================================================================
# First we need to surpress the version warnings from Pandas.
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning) 
# =============================================================================

# Load all required modules.

import numpy as np
import pandas as pd

import os
import json
import matplotlib as matplot
import matplotlib.pylab as plt
import ipyvolume as ipv

from tyssue import Sheet, config #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry

# For cell topology/configuration
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import collapse_edge, remove_face
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division


# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

#I/O
from tyssue.io import hdf5

'''
Action attempted: energy minimisation after setup the bilayer.
'''
# Generate a bilayer structure.
bilayer = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 30, ny = 4, distx = 2, disty = 2)

bilayer.sanitize(trim_borders=True, order_edges=True)
geom.update_all(bilayer)
print(f'Initially, we have {len(bilayer.face_df)} faces/cells in our sheet. \n')

# Have a look of the generated bilayer.
fig, ax = sheet_view(bilayer, mode = '2D')
fig.set_size_inches(30,30)

# =============================================================================
# """ Show index of faces:   """
#
# for face, data in bilayer.face_df.iterrows():
#     ax.text(data.x, data.y, face)
# =============================================================================


# =============================================================================
# """ Show index of vertices: """
#
# for vert, data in bilayer.vert_df.iterrows():
#     ax.text(data.x, data.y+0.1, vert)
# =============================================================================


#quick zoom in
fig, ax = sheet_view(bilayer, ['x', 'y'], mode="quick")

ax.set_xlim(3, 10)
ax.set_ylim(0, 6)


for face, data in bilayer.face_df.iterrows():
    ax.text(data.x, data.y, face)
for vert, data in bilayer.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)


# Update bilayer specs to add attributes for energy minimization.
nondim_specs = config.dynamics.quasistatic_plane_spec()
bilayer.update_specs(nondim_specs, reset = True)

# Perform energy minimization.
solver = QSSolver()
res = solver.find_energy_min(bilayer, geom, smodel)
fig, ax = sheet_view(bilayer)


""" Example of vertex manipulating """

bilayer.vert_df.loc[2,['y','x']]

bilayer.vert_df.loc[2,['y','x']] = 10,7
geom.update_all(bilayer)
sheet_view(bilayer)

res = solver.find_energy_min(bilayer, geom, smodel)
fig, ax = sheet_view(bilayer)
fig.set_size_inches(12, 5)


''' Show face index of bilayer '''
fig, ax = sheet_view(bilayer, mode = '2D')
fig.set_size_inches(30,30)

for face, data in bilayer.face_df.iterrows():
    ax.text(data.x, data.y, face)


''' Show force on vertices '''

grad_E = smodel.compute_gradient(bilayer)
grad_E.head()

gradients = smodel.compute_gradient(bilayer, components=True)    #return for each effector
gradients = {label: (srce, trgt) for label, (srce, trgt) in zip(smodel.labels, gradients)}
gradients['Line tension'][0].head()

from tyssue.draw import plot_forces
fig, ax = plot_forces(bilayer, geom, smodel, ['x', 'y'], scaling=1)
fig.set_size_inches(10, 12)


"""
Try manipulate the center dataset, 1) non uniform shape. 2) get boundary cells.
"""

'''
Perform cell division, without using event manager
'''

# Generate a daughter cell.
daughter = cell_division(bilayer, 7, geom, angle=np.pi/2)
geom.update_all(bilayer)
sheet_view(bilayer)

# Perform energy minimisation.
res = solver.find_energy_min(bilayer, geom, smodel)
fig, ax = sheet_view(bilayer)
fig.set_size_inches(12, 5)

for i in list(range(3,30,4)):
	sub = cell_division(bilayer,i, geom, angle=np.pi*np.random.rand(1).item())
	geom.update_all(bilayer)
	res=solver.find_energy_min(bilayer, geom, smodel)
fig, ax = sheet_view(bilayer)
fig.set_size_inches(12,5)


""" Perform cell divison by using event manager """

from tyssue.topology.sheet_topology import cell_division

# Write behavior function for division.
def division(sheet, manager, cell_id, crit_area, growth_rate=0.1, dt=1.):
    """Defines a division behavior.
    
    Parameters
    ----------
    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell
    crit_area: float
        the area at which 
    growth_rate: float
        increase in the area per unit time
        A_0(t + dt) = A0(t) * (1 + growth_rate * dt)
    """

    # if the cell area is larger than the crit_area, we let the cell divide.
    if sheet.face_df.loc[cell_id, "area"] > crit_area:
        # restore prefered_area
        sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, cell_id, geom)
        # update geometry
        geom.update_all(sheet)
        print(f"cell num: {daughter} is born")
    # if the cell area is less than the threshold, update the area by growth.
    else:
        sheet.face_df.loc[cell_id, "prefered_area"] *= (1 + dt * growth_rate)


from tyssue.behaviors import EventManager

# Initialisation of manager 
manager = EventManager("face")

from tyssue import History

t = 0
stop = 30

# The History object records all the time steps 
history = History(bilayer)

while manager.current and t < stop:
    manager.append(division, cell_id=t, crit_area=0.548)
    # Execute the event in the current list
    manager.execute(bilayer)
    t += 1
    
    # Find energy min
    #res = solver.find_energy_min(bilayer, geom, smodel)
    history.record()

    # Switch event list from the next list to the current list
    manager.update()

res = solver.find_energy_min(bilayer, geom, smodel)
fig, ax = sheet_view(bilayer, mode="2D")


""" Try remove_face """
from tyssue.topology.base_topology import remove_face

remove_face(bilayer, 1)
res = solver.find_energy_min(bilayer, geom, smodel)
sheet_view(bilayer, mode = '2D')


""" Try edge collapse """

"""
Collapses edge and merges it's vertices, creating (or increasing the rank of)
    a rosette structure.

    If `reindex` is `True` (the default), resets indexes and topology data.
    The edge is collapsed on the smaller of the srce, trgt indexes
    (to minimize reindexing impact)

    Returns the index of the collapsed edge's remaining vertex (its srce)
    
    In default, two-sided cells are not allowed.
 """

from tyssue.topology.base_topology import collapse_edge

collapse_edge(bilayer, 3)
res = solver.find_energy_min(bilayer, geom, smodel)
sheet_view(bilayer, mode = '2D')








"""
This is the end of the script.
"""
