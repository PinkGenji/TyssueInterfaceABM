# -*- coding: utf-8 -*-
"""
This script is to run a more automated simulation of cell division.
This simulation aims to provide better understanding on the data frame changes
during the process.
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

# Event manager
from tyssue.behaviors import EventManager

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells


""" start the project """
# Generate the cell sheet as three cells.
sheet = Sheet.planar_sheet_2d('face', nx = 3, ny=4, distx=2, disty=2)
sheet.sanitize(trim_borders=True)
geom.update_all(sheet)
sheet_view(sheet, mode = '2D')


# Add more mechanical properties, take four factors
# line tensions; edge length elasticity; face contractility and face area elasticity
new_specs = model_factory([effectors.LineTension, effectors.LengthElasticity, effectors.FaceContractility, effectors.FaceAreaElasticity])

sheet.update_specs(new_specs.specs, reset = True)
geom.update_all(sheet)

fig, ax = sheet_view(sheet, mode = '2D')

# Minimize the potential engery
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)
# Visualize the sheet.
fig, ax = sheet_view(sheet,  mode = '2D')

# Write a behaviour function.
def division(sheet, manager, cell_id=0, crit_area=2.0, growth_rate=0.001, dt=1.):
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
        sheet.face_df.loc[12, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, cell_id, geom)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        geom.update_all(sheet)
        print(f"cell n°{daughter} is born")
    else:
        # 
        sheet.face_df.loc[12, "prefered_area"] *= (1 + dt * growth_rate)
        manager.append(division, cell_id=cell_id)


# Initialise the manager, by default a wait function is set as current event.
# Any new event added to the manager are added to the 'next' list.

# Initialisation of manager 
manager = EventManager('face')

# Add action/event to the manager
manager.append(division, cell_id=2)

print('manager.current :')
print(manager.current)
print()
print('manager.next :')
print(manager.next)

from tyssue import History

t= 0
stop = 2

# initialise the History object.
sim_recorder = History(sheet)

while manager.current and t < stop:
	# Execute the event in the current list.
	manager.execute(sheet)
	t += 1
	sheet.reset_index(order = True)
	# Find energy min
	res = solver.find_energy_min(sheet, geom, smodel)
	sim_recorder.record()
	# Switch event list from the next list to the current list.
	manager.update()

# Visualisation of the tissue
fig, ax = sheet_view(sheet, mode="2D")
fig.set_size_inches(8, 8)

from IPython import display
from tyssue.draw import (
    sheet_view,
    highlight_faces,
    create_gif,
    browse_history
)
create_gif(sim_recorder, "growth.gif", num_frames=30, margin=5)
display.Image("growth.gif")



""" 
This is the end of the script.
"""
