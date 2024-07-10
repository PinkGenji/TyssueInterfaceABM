# -*- coding: utf-8 -*-
"""
Let us investigate the relationships between different data structures
that deals with different components of the model (edge, face, vertices).
"""
# load core object
from tyssue import Sheet
# for simple 2D shape
from tyssue import PlanarGeometry as geom
# for visualisation
from tyssue.draw import sheet_view


'''
The following section is to investigate the upcasting and downcasting.
'''


sheet = Sheet.planar_sheet_2d(
    'basic2D', # a name or identifier for this sheet
    nx=6, # approximate number of cells on the x axis
    ny=7, # approximate number of cells along the y axis
    distx=1, # distance between 2 cells along x
    disty=1 # distance between 2 cells along y
)
geom.update_all(sheet)

# Give the tissue a nice hear cut ;)
sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)

fig, ax = sheet_view(sheet, mode = '2D')
fig.set_size_inches(10,10)

# Inspect the data structures first.
# All the data associated with the cell mesh is 
for element, data in sheet.datasets.items():
    print(f"{element} table has shape {data.shape}")

sheet.datasets['edge'].head()
sheet.face_df
sheet.vert_df.head()

# upcasting: copy the area of each face to each of its edges.

print('Faces associated with the first edges:')
print(sheet.edge_df['face'].head())
print('\n')

# First edge associated face, also called upcasting
face = sheet.edge_df.loc[0, 'face']

print('Area of cell # {}:'.format(int(face)))
print(sheet.face_df.loc[face, 'area'])

print('\n')
print('Upcasted areas over the edges:')
print(sheet.upcast_face(sheet.face_df['area']).head())


#The oppsite of upcasting is downcasting.

#Function sum_srce() sums the values of the edge-indexed dataframe `df` grouped
# by the values of `self.edge_df["srce"]
print(sheet.sum_srce(sheet.edge_df['trgt']).head())


'''
Next, we try the Tyssue visualisation tools to understand the indexing order.
'''







'''
Now we explore how cells are drawn. We need to figure out how polygons are computed.

'''

from tyssue.generation import hexa_grid2d, from_2d_voronoi
help(hexa_grid2d) #Creates a hexagonal shape.
help(from_2d_voronoi)     #Creates 2D datasets from a voronoi tessellation.

'''
Explore the algorithm of hexa_grid in 2D, with an aim to use other polygons.

Code below

def hexa_grid2d(nx, ny, distx, disty, noise=None):
    """Creates an hexagonal grid of points"""
    cy, cx = np.mgrid[0:ny, 0:nx]
    cx = cx.astype(float)
    cy = cy.astype(float)
    cx[::2, :] += 0.5

    centers = np.vstack([cx.flatten(), cy.flatten()]).astype(float).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
    return centers

Algorithm explained below;

'''

import numpy as np

#set up constants.
nx = 4
ny =3
distx = 0.5
disty=0.7

#create a multi-dimensional meshgrid of ny and nx array.
cy, cx = np.mgrid[0:ny, 0:nx]   

#convert types
cx = cx.astype(float)
cy = cy.astype(float)

#move the x-axis by 0.5
cx[::2, :] += 0.5

#use vstack to stack arrays in sequence vertically (row wise).
#use flatten() function to get a copy of the array collapsed into 1D.
centers = np.vstack([cx.flatten(), cy.flatten()]).astype(float).T
centers[:, 0] *= distx
centers[:, 1] *= disty

# The next step is 2D voronoi tessellation.

from scipy.spatial import Voronoi, voronoi_plot_2d
from tyssue.generation import from_2d_voronoi

help(Voronoi)
'''
Voronoi() function returns a voronoi diagram, we can view the diagram via 
matplotlib.pyplot and we can also get the voronoice verices as an array.
'''
help(from_2d_voronoi)

import matplotlib.pyplot as plt

#centers[1]=0.88     # uncomment this line to see how we can get non-hexagons.
v_center = Voronoi(centers)

#show plot
fig = voronoi_plot_2d(v_center)
plt.show()

v_center.vertices 	      #show vertcies array
v_center.regions 	      #show voronoi regions
v_center.ridge_vertices   #voronoi ridge
v_center.ridge_point 	  #The ridges are perpendicular between lines drawn between the following input points.

# We can see that the hexagon is inherited from the voronoi_plot_2d function.


'''
This is the end of the script.
'''
