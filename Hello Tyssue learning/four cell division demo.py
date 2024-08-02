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


""" start the project """
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

# Initialise the manager, by default a wait function is set as current event.
# Any new event added to the manager are added to the 'next' list.

# Initialisation of manager 
manager = EventManager('face')
print(manager.current)
print(manager.next)
manager.execute(sheet)
print(manager.current)

from tyssue import History

t= 0
stop = 1

# initialise the History object.
sim_recorder = History(sheet)

# =============================================================================
# print('manager.current :')
# print(manager.current)
# 
# print('\n manager.next :')
# print(manager.next)
# 
# for i in list(sheet.face_df.index):
#     print(f'\n cell id {i} to be appended: ')
#     manager.append(division, cell_id = i)
#     print(f'\n cell id {i} has been appended.')
# =============================================================================




while manager.next and t < stop:
    print(f'\n Time step {t} starts now: ')
    # Execute the event in the current list.

    for i in list(sheet.face_df.index):	
		
        print(f'\n current manager index is {i}')
        # print('manager current:')
        # print(manager.current)

        manager.append(division, cell_id = i)
        # print('\n manager next:')
        # print(manager.next)
        manager.execute(sheet)
        # print('\n after execution, manager current is: ')
        # print(manager.current)

        # Find energy min.
        res = solver.find_energy_min(sheet, geom, smodel)
    # Record the step.
        sim_recorder.record()
	# Switch event list from the next list to the current list.
        manager.update()
    print(f'\n Finished time step {t} simulation.')    
    t += 1    # move into the next time step.
    

# Visualisation of the tissue
fig, ax = sheet_view(sheet, mode="2D")


from IPython import display
from tyssue.draw import (
    sheet_view,
    highlight_faces,
    create_gif,
    browse_history
)

# Createa gif image, set margin = -1 to let the draw function decide.
create_gif(sim_recorder, "growth.gif", num_frames=100, margin=-1)
display.Image("growth.gif")



""" 
This is the end of the script.
"""
