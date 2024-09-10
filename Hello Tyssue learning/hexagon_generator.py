# -*- coding: utf-8 -*-
"""
This script does the boundary shape change.
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
from tyssue.topology.sheet_topology import type1_transition, cell_division
from tyssue.topology.base_topology import collapse_edge, remove_face, merge_border_edges
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology import condition_4i, condition_4ii




# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

#I/O
from tyssue.io import hdf5

# Draw voronoi diagram
from scipy.spatial import Voronoi, voronoi_plot_2d
from tyssue.generation import hexa_grid2d, from_2d_voronoi

nx = 3
ny=2
distx=1
disty = 1
noise = 0

grid = hexa_grid2d(nx, ny, distx, disty, noise)
#grid = np.flip(grid,1)
datasets = from_2d_voronoi(Voronoi(grid))

vor = Voronoi(grid)

fig = voronoi_plot_2d(vor)
plt.show()


""" Try: without using trim, just delete the outside faces """
bilayer=Sheet.planar_sheet_2d(identifier='bilayer', nx = 3, ny = 2, distx = 1, disty = 1)
geom.update_all(bilayer)

# Find the outside-edges and faces associated with them, and get rid off them.
invalid_edge = bilayer.get_invalid()   
bilayer.remove(invalid_edge)  

fig, ax = sheet_view(bilayer, edge = {'head_width':0.1})
for face, data in bilayer.face_df.iterrows():
    ax.text(data.x, data.y, face) 


def delete_face(sheet_obj, face_deleting):
    """

    
    Parameters
    ----------
    sheet_obj : Epithelium
        An Epithelium 'Sheet' object from Tyssue.
    face_deleting : Int
        The index of the face to be deleted.

    Returns
    -------
    A Pandas Data Frame that deletes the face, with border edges are single
    arrowed, without index resetting.

    """
    # Compute all edges associated with the face, then drop these edges in df.
    associated_edges = sheet_obj.edge_df[sheet_obj.edge_df['face'] == face_deleting]
    sheet_obj.edge_df.drop(associated_edges.index, inplace = True)
    
    # All associated edges are removed, now remove the 'empty' face and reindex.
    sheet_obj.face_df.drop(face_deleting , inplace =True)


# Use the defined delet_face function to delete the face.
delete_face(bilayer, 2)
delete_face(bilayer,3)


# reset the indices.
bilayer.reset_index()

# Plot the figure to check.
fig, ax = sheet_view(bilayer, edge = {'head_width':0.1})
for face, data in bilayer.face_df.iterrows():
    ax.text(data.x, data.y, face)




""" This is the end of the script """
