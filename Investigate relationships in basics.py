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
Next, we shall explore the specification.
'''
# First use the default specification for the dyanmics of a sheet vertex model.
# This way, we can assign properties such as line_tension into the edge_df.
from tyssue.solvers.quasistatic import QSSolver
from tyssue import Sheet, config
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue import PlanarGeometry as geom
from tyssue.draw import sheet_view


solver = QSSolver()
sheet1 = Sheet.planar_sheet_2d('division', 6, 6, 1, 1)
sheet1.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet1)

nondim_specs = config.dynamics.quasistatic_plane_spec()

# No differences between two specs.
# We can use either specs.

# udpate the new specs (contain line_tension, etc) into the cell data.
sheet1.update_specs(nondim_specs, reset=True)

# Show number of cells, edges and vertices of the sheet.
print("Number of cells: {}\n"
      "          edges: {}\n"
      "          vertices: {}\n".format(sheet1.Nf, sheet1.Ne, sheet1.Nv))

# ## Minimize energy
res = solver.find_energy_min(sheet1, geom, smodel)

# ## View the result
draw_specs = config.draw.sheet_spec()
draw_specs['vert']['visible'] = False
draw_specs['edge']['head_width'] = 0  # values other than 0 gives error.
fig, ax = sheet_view(sheet1, **draw_specs)
fig.set_size_inches(12, 5)

# Check terms in the spec.
sheet1.specs

# Make sure we print all, set both to None to print all.
import pandas as pd
pd.set_option('display.max_rows', 5, 'display.max_columns', 10)

# Inspect column names and the entries that are relevant.
print(sheet1.edge_df.keys())
print(sheet1.edge_df['sx'])

# Try change the line_tension in certain rows.
sum(sheet1.edge_df['line_tension'])    # Keep a record of the sum.
sheet1.edge_df.loc[sheet1.edge_df['sx']< 2, 'line_tension'] = 0.5
sum(sheet1.edge_df['line_tension'])    # Check if the sum changes.


# Now we try to add another property attribute into the edge dictionary.
sheet1.update_specs({'edge':{'test111':1.0}})
sheet1.edge_df.keys()
#try another way
sheet1.specs['edge']['test222'] = 0.1
sheet1.edge_df.keys() # Nothing changed in the spec dictionary.
'''
From the above, we can see that the we must use update_specs() to add attribute,
but we can change the value for certain cells by using the dataframe structure.
'''

"""
Now, we explore the energy related parts in Tyssue.
"""

# We can compute the energy of a given configuration.
from tyssue.dynamics import effectors, model_factory
model_example = model_factory([effectors.LineTension, effectors.FaceContractility, effectors.FaceAreaElasticity])
model_example.specs
sheet1.update_specs(model_example.specs, reset = True)  # Reset the model.
geom.update_all(sheet1)    # Update sheet1

from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
energy = smodel.compute_energy(sheet1)
print(f'Total energy: {energy: .3f}')
Et, Ec, Ea = smodel.compute_energy(sheet1, full_output=True)
Et.head()
Ec.head()
Ea.head()

fig, ax = sheet_view(sheet1, coords=list('xy'), face={"visible": True, "color": Ec, "colormap": "gray"}, edge={"color": Et},)


""" Given that we can compute energy with given configuration, we then need a 
solver to find the minimum energy state. """

smodel.compute_energy(sheet1) 	# Total energy.
smodel.compute_gradient(sheet1).head() 	# Min energy is found via gradient descent.
solver = QSSolver 	#Quasi-Static solver
res = solver.find_energy_min(sheet1, geom, smodel)
print("Successfull gradient descent? ", res['success'])

# Plot the new figure.
fig, ax = sheet_view(sheet1)
fig.set_size_inches(10, 10)
ax.set_aspect('equal')

# Print the gradient error during approximation.
print('Total gradient error: ')
solver.check_grad(sheet1, geom, smodel)

fig, ax = sheet_view(sheet1, mode="quick")
for f, (x, y) in sheet1.face_df[["x", "y"]].iterrows():
    ax.text(x, y, f)

''' Try with merging two vertices. '''

from tyssue.topology.base_topology import collapse_edge



centre_edge = sheet1.edge_df.eval("sx**2 + sy**2").idxmin()
collapse_edge(sheet1, centre_edge)
sheet1.update_rank()
geom.update_all(sheet1)

fig, ax = sheet_view(sheet1, mode="quick")

solver = QSSolver 	#Quasi-Static solver
res = solver.find_energy_min(sheet1, geom, smodel)
print("Successfull gradient descent? ", res['success'])

# Plot the new figure.
fig, ax = sheet_view(sheet1)
fig.set_size_inches(10, 10)
ax.set_aspect('equal')

print("Maximum vertex rank: ", sheet1.vert_df['rank'].max())

sheet1.validate()












'''
This is the end of the script.
'''
