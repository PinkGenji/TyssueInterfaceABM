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
from tyssue.topology.base_topology import remove_face, add_vert
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.topology.sheet_topology import face_division
from tyssue.solvers.viscous import EulerSolver
from tyssue.dynamics import model_factory, effectors

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import plot_forces

from my_headers import delete_face, xprod_2d, put_vert, T1_check, my_ode, type1_transition_custom


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




""" Grow first, then cells divide. """
# Write behavior function for division_1.
def division_1(sheet, cent_data, cell_id, crit_area, growth_rate, dt):
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
        chosen_index = rng.choice(list(edge_in_cell.index))
        # Extract and store the centroid coordinate.
        c0x = float(cent_data.loc[cent_data['face']==cell_id, ['fx']].values[0])
        c0y = float(cent_data.loc[cent_data['face']==cell_id, ['fy']].values[0])
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
                c1 = dx*ry-dy*rx
                c2 = s0y*rx-p0y*rx - s0x*ry + p0x*ry
                k=c2/c1
                intersection = [s0x+k*dx, s0y+k*dy]
                oppo_index = put_vert(sheet, index, intersection)[0]
                # Split the cell with a line.
                new_face_index = face_division(sheet, mother = cell_id, vert_a = new_mid_index , vert_b = oppo_index )
                # Put a vertex at the centroid, on the newly formed edge (last row in df).
                cent_index = put_vert(sheet, edge = sheet.edge_df.index[-1], coord_put = c0)[0]
                return new_face_index
            else:
                continue
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
        division_1(sheet, cent_data = centre_data, cell_id = i)
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


""" Implement the Euler simple forward solver. """
# Generate the cell sheet as three cells.
num_x = 4
num_y = 4
sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=2, disty=2)
geom.update_all(sheet)
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())  
for i in list(range(num_x, num_y*(num_x+1), 2*(num_x+1) )):
    delete_face(sheet, i)
    delete_face(sheet, i+1)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
sheet_view(sheet)
sheet.get_extra_indices()
# Visualize the sheet.
fig, ax = sheet_view(sheet,  mode = '2D')
# First, we need a way to compute the energy, then use gradient descent.
model = model_factory([
    effectors.LineTension, # This defines cell cell adhesion force.
    effectors.FaceContractility, # This defines deformation energy coefficient
    effectors.FaceAreaElasticity # This defines memrbane surface energy coefficient.
    ])
sheet.vert_df['viscosity'] = 1.0
sheet.update_specs(model.specs, reset=True)
geom.update_all(sheet)
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] /= 2
    else:
        continue
geom.update_all(sheet)

# We need set the all the threshold value first.
t1_threshold = sheet.edge_df.loc[:,'length'].mean()/10
t2_threshold = sheet.face_df.loc[:,'area'].mean()/5
area_threshold = sheet.face_df.loc[:,'area'].mean()*1.1
growth_speed = sheet.face_df.loc[:,'area'].mean()/2

# Now assume we want to go from t = 0 to t= 0.2, dt = 0.1
t0 = 0
t_end = 0.12
dt = 0.001
time_points = np.linspace(t0, t_end, int((t_end - t0) / dt) + 1)
print(f'time points are: {time_points}.')
ls=[]


for t in time_points:
    print(f'start at t= {round(t, 5)}.')
    
    # Mesh restructure check
    # T1 transition, edge rearrangment check
    while True:
    # Check for any edge below the threshold, starting from index 0 upwards
        edge_to_process = None
        for index in sheet.edge_df.index:
            if sheet.edge_df.loc[index, 'length'] < t1_threshold:
                edge_to_process = index
                break  
        # Exit the loop if no edges are below the threshold
        if edge_to_process is None:
            break
    
        # Process the identified edge with T1 transition
        print(f'Edge {edge_to_process} is shorter than the t1 threshold value.')
        ls.append(sheet.edge_df.loc[edge_to_process,])
        type1_transition(sheet, edge_to_process,remove_tri_faces=False, multiplier=1.5)
        geom.update_all(sheet)
        fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
        for face, data in sheet.vert_df.iterrows():
            ax.text(data.x, data.y, face)
        sheet.reset_index()
        
        # Update sheet.edge_df to reflect changes and re-check edges in the next loop iteration
    geom.update_all(sheet)
    sheet.reset_index()

    
    # Force computing and updating positions.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # Compute the moving direction.
    dot_r = my_ode(sheet)
    new_pos = pos + dot_r*dt
    # Save the new positions back to `vert_df`
    sheet.vert_df.loc[valid_active_verts , sheet.coords] = new_pos
    geom.update_all(sheet)
    
    # Plot with title contain time.
    if t in time_points[::3]:
        fig, ax = sheet_view(sheet)
        ax.title.set_text(f'time = {round(t, 5)}')
    
    
    # T2 transition check.
    tri_faces =sheet.face_df[sheet.face_df['num_sides']<4].index
    for i in tri_faces:
        if sheet.face_df.loc[i,'area'] < area_threshold:
            remove_face(sheet, tri_faces[0])
        else:
            continue
    sheet.reset_index(order = True)
    geom.update_all(sheet)
    # Cell division.
    # Store the centroid before iteration of cells.
    unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
    centre_data = unique_edges_df.loc[:,['face','fx','fy']]
    # Loop over all the faces.
    all_cells = sheet.face_df.index
    for i in all_cells:
        #print(f'We are in time step {t}, checking cell {i}.')
        division_1(sheet, cent_data= centre_data, cell_id = i, crit_area=area_threshold, growth_rate= growth_speed, dt=dt)
    sheet.reset_index(order = True)
    geom.update_all(sheet)
    
    
    # Plot with title contain time.
    if t in time_points[::3]:
        fig, ax = sheet_view(sheet)
        ax.title.set_text(f'time = {round(t, 5)}')


""" Divide first, then daughter cells expand. """



"""
This is the end of the script.
"""
