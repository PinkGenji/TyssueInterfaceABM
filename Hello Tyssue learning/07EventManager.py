# -*- coding: utf-8 -*-
"""
This file is for learning 07EventManager of the tyssue package.
"""

'''
The event manager in tyssue is a system that manage the different behaviour of
a simulation.

It is compsoed of two lists:
    current: list of actions to be executed during the current simulation step.
    next: list of actions to be executed for the next simulation step, this list
    is fill during the current simulation step.

A behaviour is a function which describes/contains the actions requested to 
realize an event. For example, the behaviour which describe the cell division 
is composed of:
    -> growing cell area until it reaches a certain threshold
    -> dividing cell into two daughter cell

'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ipywidgets as widgets
from IPython import display
import ipyvolume as ipv

# Core object
from tyssue import Sheet
# Simple 2D geometry
from tyssue import PlanarGeometry as sgeom
# Visualisation
from tyssue.draw import (sheet_view, highlight_faces, create_gif, browse_history)
import numpy as np
import random
random.seed(42)  # Controls Python's random module (e.g. event shuffling)
np.random.seed(42) # Controls NumPy's RNG (e.g. vertex positions, topology)


'''
Generate a 2D mesh

'''
# Choose which solver to use
use_qs_solver = False  # Set to True for QSSolver, False for EulerSolver


sheet = Sheet.planar_sheet_2d(
    'basic2D', # a name or identifier for this sheet
    nx=6, # approximate number of cells on the x-axis
    ny=7, # approximate number of cells along the y-axis
    distx=1, # distance between 2 cells along x
    disty=1, # distance between 2 cells along y
    noise=0 # some position noise
)
sgeom.update_all(sheet)

# Give the tissue a nice haircut ;)
sheet.sanitize(trim_borders=True, order_edges=True)
sgeom.update_all(sheet)

# Visualisation of the tissue
fig, ax = sheet_view(sheet, mode="2D")
fig.set_size_inches(8, 8)
plt.show()

from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers import QSSolver
from tyssue.solvers.viscous import EulerSolver
from pprint import pprint

specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 0.12,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 1.0,
       'contractility': 0.04,
       'is_alive': 1,
       'prefered_area': 1.0},
   'settings': {
       'grad_norm_factor': 1.0,
       'nrj_norm_factor': 1.0
   },
   'vert': {
       'is_active': 1
   }
}


# Update the specs (adds / changes the values in the dataframes' columns)
sheet.update_specs(specs)

# Check the tissue is at its equilibrium
solver = QSSolver()
res = solver.find_energy_min(sheet, sgeom, smodel)
# Visualisation of the tissue
fig, ax = sheet_view(sheet, mode="2D")
plt.show()

'''
Write a behaviour function:

Behaviour function have 2 parts:
    signature part; which contains sheet and manager parameter.
    keywords part; which is specific to one behaviour function.    

To add a behaviour to the manager, append method has to be used, and it needs 
as a parameter the function name and the keyword part.

'''

from tyssue.topology.sheet_topology import cell_division

# Write the behaviour function:

def division(sheet, manager, cell_id=0, crit_area=2.0, growth_rate=0.1, dt=1.):
    """Defines a division behavior.

    Parameters
    ----------

    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell
    crit_area: float
        the area at which
    growth_rate: float
        increase in the prefered are per unit time
        A_0(t + dt) = A0(t) * (1 + growth_rate * dt)
    """

    if sheet.face_df.loc[cell_id, "area"] > crit_area:
        # restore prefered_area
        sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, cell_id, sgeom)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        sgeom.update_all(sheet)
        print(f"cell n°{daughter} is born")
        # Schedule division for both daughter cells
        manager.append(division, cell_id=cell_id)
        manager.append(division, cell_id=daughter)
    else:
        #
        sheet.face_df.loc[cell_id, "prefered_area"] *= (1 + dt * growth_rate)
        manager.append(division, cell_id=cell_id)

from tyssue.behaviors import EventManager

# Initialisation of manager 
manager = EventManager("face")

# Add action/event to the manager
# Add division behavior for all live cells
for cell_id in sheet.face_df.index:
    manager.append(division, cell_id=cell_id)


print('manager.current :')
print(manager.current)
print()
print('manager.next :')
print(manager.next)
from tyssue import History

t = 0
stop = 40

# The History object records all the time steps
history = History(sheet)

sheet.vert_df["viscosity"] = 1.0  # Required for EulerSolver

if use_qs_solver:
    print("Using QSSolver...")
    solver = QSSolver()
    while manager.current and t < stop:
        manager.execute(sheet)
        t += 1
        sheet.reset_index(order=True)
        res = solver.find_energy_min(sheet, sgeom, smodel)
        history.record()
        manager.update()
else:
    print("Using EulerSolver...")
    solver = EulerSolver(
        sheet, sgeom, smodel,
        history=history,
        manager=manager,
        bounds=(-sheet.edge_df.length.mean()/10, sheet.edge_df.length.mean()/10))
    dt = 1.0
    tf = 40
    solver.solve(tf=tf, dt=dt)
    history = solver.history



draw_specs = {
    "edge": {
        "color": lambda sheet: sheet.edge_df.length
    },
    "face": {
        "visible": True,
        "color": lambda sheet: sheet.face_df.area,
        "color_range": (0, 2)
    }
}

# Visualisation of the tissue
fig, ax = sheet_view(sheet, mode="2D")
fig.set_size_inches(8, 8)
plt.show()



'''
This is the end of the file.
'''
