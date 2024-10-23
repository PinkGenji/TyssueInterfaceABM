# -*- coding: utf-8 -*-
"""
This script contains an example of vertex modelling for petri dish like tissue 
cells.
"""
# =============================================================================
# First we need to surpress the warnings about deprecation or future.
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
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
from tyssue.topology.base_topology import collapse_edge, remove_face, add_vert
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division, face_division

# Event manager
from tyssue.behaviors import EventManager

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

from my_headers import delete_face, xprod_2d, put_vert


""" start the project. """
# Generate the cell sheet as three cells.
np.random.seed(70)
num_x = 4
num_y = 4

sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=2, disty=2)

geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
for i in list(range(num_x, num_y*(num_x+1), 2*(num_x+1) )):
    delete_face(sheet, i)
    delete_face(sheet, i+1)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
# add mechanical properties.
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
sheet.update_specs(specs, reset = True)
geom.update_all(sheet)

# Minimize the potential engery
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)

# Visualize the sheet.
fig, ax = sheet_view(sheet,  mode = '2D')

""" Grow first, then cells divide. """

# Write behavior function for division_1.
def division_1(sheet, manager, cell_id, crit_area, growth_rate=0.8, dt=1):
    """The cells keep growing, when the area exceeds a critical area, then
    the cell divides.
    
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
    np.random.seed(70)

    # if the cell area is larger than the crit_area, we let the cell divide.
    if sheet.face_df.loc[cell_id, "area"] > crit_area:
        # Do division
        edge_in_cell = sheet.edge_df[sheet.edge_df.loc[:,'face'] == cell_id]
        edge_in_cell_ind_list = list(edge_in_cell.index)
        chosen_index = int(np.random.choice(edge_in_cell_ind_list , 1))
        
        # add a vertex in the middle of the chosen edge.
        new_mid_index = add_vert(sheet, edge = chosen_index)[0]
        
        # We need to determine which edge is the opposite edge
        c0x = float(edge_in_cell.loc[chosen_index,'fx'])
        c0y = float(edge_in_cell.loc[chosen_index,'fy'])
        c0 = [c0x, c0y]
        
        sheet.vert_df = sheet.vert_df.append({'y': c0y, 'is_active': 1, 'x': c0x}, ignore_index = True)
        
        # Extract for source vertex coordinates
        p0x = float(edge_in_cell.loc[chosen_index ,'sx'])
        p0y = float(edge_in_cell.loc[chosen_index ,'sy'])


        # Extract the directional vector.
        rx = float(edge_in_cell.loc[chosen_index ,'rx'])
        ry = float(edge_in_cell.loc[chosen_index ,'ry'])
        r  = [-rx, -ry]   # use the line in opposite direction.
        
        # We need to use iterrows to iterate over rows in pandas df
        # The iteration has the form of (index, series)
        # The series can be sliced.
        for index, row in edge_in_cell.iterrows():
            s0x = row['sx']
            s0y = row['sy']
            t0x = row['tx']
            t0y = row['ty']
            v1 = [s0x-p0x,s0y-p0y]
            v2 = [t0x-p0x,t0y-p0y]
            # if the xprod_2d returns negative, then line intersects the line segment.
            if xprod_2d(r, v1)*xprod_2d(r, v2) < 0:
                #print(f'The edge that is intersecting is: {index}')
                dx = row['dx']
                dy = row['dy']
                c1 = (dx*ry/rx)-dy
                c2 = s0y-p0y - (s0x*ry/rx) + (p0x*ry/rx)
                k=c2/c1
                intersection = [s0x+k*dx, s0y+k*dy]
                oppo_index = int(put_vert(sheet, index, intersection)[0])
        new_face_index = face_division(sheet, mother = cell_id, vert_a = new_mid_index , vert_b = oppo_index )
        # Put a vertex at the centroid, on the newly formed edge (last row in df).
        put_vert(sheet, edge = sheet.edge_df.index[-1], coord_put = c0)
        sheet.update_num_sides()
        
        # update geometry
        #geom.update_all(sheet)
        print(f"cell num: {new_face_index} is born ")
        print(f'{chosen_index} is chosen ')
        return new_face_index
    # if the cell area is less than the threshold, update the area by growth.
    else:
        sheet.face_df.loc[cell_id, "prefered_area"] *= (1 + dt * growth_rate)


# Initialisation of manager 
manager = EventManager("face")

from tyssue import History

t = 0
stop = 3

# The History object records all the time steps 
history = History(sheet)

# Minimize the potential engery
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)

# Visualize the sheet.
cell_ave = sheet.face_df.loc[:,'area'].mean()
fig, ax = sheet_view(sheet,  mode = '2D')
ax.title.set_text('Initial setup')
ax.text(0.05, 0.95, f'Mean cell area = {cell_ave:.4f}', transform=ax.transAxes, fontsize=8, va='top', ha='left')


while manager.current and t <= stop:
    
    manager.execute(sheet)
    t += 1
    sheet.reset_index(order=True)
    
    for i in sheet.face_df.index:
        print(f'we are at time step {t}, cell {i} is being checked')
        manager.append(division_1, cell_id=i, crit_area=1.5)
    # Find energy min
    res = solver.find_energy_min(sheet, geom, smodel)
    history.record()
    fig, ax = sheet_view(sheet, mode = 'quick')
    # Switch event list from the next list to the current list
    manager.update()

min_sides = sheet.face_df.loc[:,'num_sides'].min()
print(f'The min number of edges within this configuration is: {min_sides}. ')

# Colour the vertices
from tyssue.config.draw import sheet_spec as draw_specs
draw_specs = draw_specs()

sheet_view(sheet)

# =============================================================================
#     # Execute the event in the current list
#         manager.execute(sheet)
#     # Find energy min
#     #res = solver.find_energy_min(bilayer, geom, smodel)
#         history.record()
# 
#     # Switch event list from the next list to the current list
#         manager.update()
# =============================================================================
        
# =============================================================================
#     res = solver.find_energy_min(sheet, geom, smodel)
#     geom.update_all(sheet)
#     cell_ave = sheet.face_df.loc[:,'area'].mean()
#     
#     fig, ax = sheet_view(sheet, mode="2D")
#     ax.title.set_text(f'Snapshot at the starting of t = {t}')
#     ax.text(0.05, 0.95, f'Mean cell area = {cell_ave:.4f}', transform=ax.transAxes, fontsize=8, va='top', ha='left')
#     t += 1
# =============================================================================




# Double check the energy minimization process at t=0 and t = 1

# Need to quantify the evolution of different algorithms
# by looking at some parameters from literature. 
#e.g. cell size in the cell packing paper.




""" Divide first, then daughter cells expand. """





"""
This is the end of the script.
"""
