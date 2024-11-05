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

from tyssue import Sheet, config, History #import core object
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
from tyssue.topology.sheet_topology import face_division
from tyssue.solvers.viscous import EulerSolver

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import plot_forces

from my_headers import delete_face, xprod_2d, put_vert


""" start the project. """
# Set global RNG seed

rng = np.random.default_rng(70)


# Generate the cell sheet as three cells.
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


# Visualize the sheet.
fig, ax = sheet_view(sheet,  mode = '2D')


""" Implement the Euler simple forward solver. """
geom.update_all(sheet)
sheet.settings['threshold_length'] = 1e-3

sheet.update_specs(config.dynamics.quasistatic_plane_spec())
sheet.face_df["prefered_area"] = sheet.face_df["area"].mean()
history = History(sheet) #, extra_cols={"edge":["dx", "dy", "sx", "sy", "tx", "ty"]})

sheet.vert_df['viscosity'] = 1.0
sheet.edge_df.loc[[0, 17],  'line_tension'] *= 4
sheet.face_df.loc[1,  'prefered_area'] *= 1.2

fig, ax = plot_forces(sheet, geom, smodel, ['x', 'y'], 1)


# Solver instanciation: contrary to the quasistatic solver, this sovler needs 
# the sheet, goemetry and model at instanciation time.
solver = EulerSolver(
    sheet,
    geom,
    smodel,
    history=history,
    auto_reconnect=True)

'''
The solver's solve method accepts a on_topo_change function as argument.
This function is executed each time a topology change occurs.
Here, we reset the line tension to its original value.

'''
def on_topo_change(sheet):
    print('Topology changed!\n')
    print("reseting tension")
    sheet.edge_df["line_tension"] = sheet.specs["edge"]["line_tension"]

# Solving from t = 0 to t = 15.

res = solver.solve(tf=15, dt=0.05, on_topo_change=on_topo_change,
                   topo_change_args=(solver.eptm,))

# Showing the result via picture.
sheet_view(sheet)

""" Grow first, then cells divide. """
# Write behavior function for division_1.
def division_1(sheet, cell_id, crit_area=1, growth_rate=0.5, dt=1):
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

    # if the cell area is larger than the crit_area, we let the cell divide.
    if sheet.face_df.loc[cell_id, "area"] > crit_area:
        # Do division, pikc number 2 cell for example.
        condition = sheet.edge_df.loc[:,'face'] == cell_id
        edge_in_cell = sheet.edge_df[condition]
        # We need to randomly choose one of the edges in cell 2.
        chosen_index = int(np.random.choice(list(edge_in_cell.index) , 1))
        # Extract and store the centroid coordinate.
        c0x = float(centre_data.loc[centre_data['face']==cell_id, ['fx']].values[0])
        c0y = float(centre_data.loc[centre_data['face']==cell_id, ['fy']].values[0])
        c0 = [c0x, c0y]

        # Add a vertex in the middle of the chosen edge.
        new_mid_index = add_vert(sheet, edge = chosen_index)[0]
        # Extract for source vertex coordinates of the newly added vertex.
        p0x = sheet.vert_df.loc[new_mid_index,'x']
        p0y = sheet.vert_df.loc[new_mid_index,'y']
        p0 = [p0x, p0y]

        # Compute the directional vector from new_mid_point to centroid.
        rx = c0x - p0x
        ry = c0y - p0y
        r  = [rx, ry]   # use the line in opposite direction.
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
            if xprod_2d(r, v1)*xprod_2d(r, v2) < 0 and index !=chosen_index :
                dx = row['dx']
                dy = row['dy']
                c1 = (dx*ry/rx)-dy
                c2 = s0y-p0y - (s0x*ry/rx) + (p0x*ry/rx)
                k=c2/c1
                intersection = [s0x+k*dx, s0y+k*dy]
                oppo_index = put_vert(sheet, index, intersection)[0]
            else:
                continue
        # Split the cell with a line.
        new_face_index = face_division(sheet, mother = cell_id, vert_a = new_mid_index , vert_b = oppo_index )
        # Put a vertex at the centroid, on the newly formed edge (last row in df).
        cent_index = put_vert(sheet, edge = sheet.edge_df.index[-1], coord_put = c0)[0]
        # update geometry
        geom.update_all(sheet)
        return new_face_index
    # if the cell area is less than the threshold, update the area by growth.
    else:
        sheet.face_df.loc[cell_id, "prefered_area"] *= (1 + dt * growth_rate)






from tyssue.config.draw import sheet_spec as default_spec
draw_specs = default_spec()


t = 0
stop = 1
while t < stop:
    # Store the centroid before iteration of cells.
    unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
    centre_data = unique_edges_df.loc[:,['face','fx','fy']]
    # Loop over all the faces.
    all_cells = sheet.face_df.index
    for i in all_cells:
        #print(f'We are in time step {t}, checking cell {i}.')
        division_1(sheet, cell_id = i)
    #res = solver.find_energy_min(sheet, geom, smodel)
    geom.update_all(sheet)
    # Plot with highlighted vertices
    sheet.vert_df['rand'] = np.linspace(0.0, 1.0, num=sheet.vert_df.shape[0])
    cmap = plt.cm.get_cmap('viridis')
    color_cmap = cmap(sheet.vert_df.rand)
    draw_specs['vert']['visible'] = True
    draw_specs['edge']['head_width'] = 0.1
    draw_specs['vert']['color'] = color_cmap
    draw_specs['vert']['alpha'] = 0.5
    draw_specs['vert']['s'] = 20
    coords = ['x', 'y']
    fig, ax = sheet_view(sheet, coords, **draw_specs)
    ax.title.set_text(f'time = {t}')
    fig.set_size_inches((8, 8))
    # Check the min edges.
    min_sides = sheet.face_df.loc[:,'num_sides'].min()
    print(f'We are at time step {t}, min_side of current configuration is {min_sides}.')
    # update the time step.
    t +=1


""" The following lines highlights the centroids of each cell. """
original = sheet.vert_df

unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
centre = []
for index, row in unique_edges_df.iterrows():
    centre_x = row['fx']
    centre_y = row['fy']
    centre.append([centre_x, centre_y])
    sheet.vert_df = sheet.vert_df.append({'y': centre_y, 'is_active': 1, 'x': centre_x}, ignore_index = True)

## Let's add a column to sheet.vert_df
sheet.vert_df['rand'] = np.linspace(0.0, 1.0, num=sheet.vert_df.shape[0])

cmap = plt.cm.get_cmap('viridis')
color_cmap = cmap(sheet.vert_df.rand)
draw_specs['vert']['visible'] = True
draw_specs['edge']['head_width'] = 0.1

draw_specs['vert']['color'] = color_cmap
draw_specs['vert']['alpha'] = 0.5
draw_specs['vert']['s'] = 20

coords = ['x', 'y']

fig, ax = sheet_view(sheet, coords, **draw_specs)
ax.title.set_text("Not loop, just plot centroid")
fig.set_size_inches((8, 8))

# Remove added rows by restore to variable original.
sheet.vert_df = original


# Need to quantify the evolution of different algorithms
# by looking at some parameters from literature. 
#e.g. cell size in the cell packing paper.




""" Divide first, then daughter cells expand. """



"""
This is the end of the script.
"""
