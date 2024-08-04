# -*- coding: utf-8 -*-
"""
This script is used to investigate the mechanism of the Event Manager.
"""

# -*- coding: utf-8 -*-
"""
This script runs a simulation of cell divisions with 4 cells initially.
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

from tyssue import History


# Generate the cell sheet as three cells.
sheet = Sheet.planar_sheet_2d('face', nx = 3, ny=4, distx=2, disty=2)
sheet.sanitize(trim_borders=True)
geom.update_all(sheet)


# add mechanical properties.
nondim_specs = nondim_specs = config.dynamics.quasistatic_plane_spec()
sheet.update_specs(nondim_specs, reset = True)
geom.update_all(sheet)

# Minimize the potential engery
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)

# Visualize the sheet.
# fig, ax = sheet_view(sheet,  mode = '2D')
initial_cells_mean_area = np.mean(sheet.face_df['area'])
# Write a behaviour function.
def division(sheet, manager, cell_id, crit_area= initial_cells_mean_area , growth_rate=0.05, dt=1):
    """Defines a division behavior.
    
    Parameters
    ----------    
    sheet: a :class:`Sheet` object
    cell_id: int; the index of the dividing cell
    crit_area: float; the area at which 
    growth_rate: float; increase in the prefered are per unit time
        A_0(t + dt) = A0(t) * (1 + growth_rate * dt)
    """

    print(f'currently checking cell number: {cell_id}')
	
    if sheet.face_df.loc[cell_id, "area"] > crit_area:
        # restore prefered_area
        sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, cell_id, geom)
        # Update the topology
        #sheet.reset_index(order=True)
        # update geometry
        geom.update_all(sheet)
        print(f"cell # {daughter} is born from {cell_id}")
    else:
        # 
        sheet.face_df.loc[cell_id, "area"] *= (sheet.face_df.loc[cell_id, "area"] + dt * growth_rate)
        geom.update_all(sheet)    

manager = EventManager('face')

""" What is the order of manger.append? """
print('\n Manager current is: ')
print(manager.current)

print("\n Manager next is: ")
print(manager.next)

# Now let's append one event to the manager.
manager.append(division, cell_id = 2)
print("\n After appending one, manager current is: ")
print(manager.current)
print("\n After appending one, manager next is: ")
print(manager.next)

# Without execution, we add another event and inspect the manager.
manager.append(division, cell_id = 1)
print("\n After appending another one, manager current is: ")
print(manager.current)
# From the printed 'manger.next', we see the new event is added side the list.
print("\n After appending another one, manager next is: ")
print(manager.next)

""" How does manager.update() work?  """

# perform update once and see what happens.
manager.update()
print("\n After one update, manager current is: ")
print(manager.current)   # We see that 'current deque' is replaced by 'next deque'.
print("\n After one update, manager next is: ")
print(manager.next)


""" What is the order of manager.execute? """

# Now perform one execution, see what happens.

manager.execute(sheet)
print('\n After the first execution, the manager current is: ')
print(manager.current)
print('\n After the first execution, the manager next is: ')
print(manager.next)

print()
print("From our experiment, we can see the execution works as a 'list flush'.")
print()

print("The work-flow of event manger hence is: \n")
print("(1) initialise the event manager. \n")
print("(2) append the list of events to the event manager. \n")
print("(3) use manger.update() to update the working deque. \n")
print("(4) do manger.execute(obj), which is done in a 'C++ flush way.' \n")



"""
This is the end of the script.
"""
