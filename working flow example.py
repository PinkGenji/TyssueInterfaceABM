# -*- coding: utf-8 -*-
"""
This script is an example of the flow of drawing -> energy relaxation process.

"""

# First we need to created a cell mesh. 

# load core object
from tyssue import Sheet
# for simple 2D shape
from tyssue import PlanarGeometry as geom
# for visualisation
from tyssue.draw import sheet_view

# For solvers
from tyssue.solvers.quasistatic import QSSolver
from tyssue import config
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel

from tyssue.dynamics import effectors, model_factory

sheet1 = Sheet.planar_sheet_2d('division', 6, 6, 1, 1)
sheet1.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet1)
fig, ax = sheet_view(sheet1, mode= 'quick')   

# prepare for energy relaxation.
solver = QSSolver()

nondim_specs = config.dynamics.quasistatic_plane_spec()

# udpate the new specs (contain line_tension, etc) into the cell data.
sheet1.update_specs(nondim_specs, reset=True)

# Show number of cells, edges and vertices of the sheet.
print("Number of cells: {}\n"
      "          edges: {}\n"
      "          vertices: {}\n".format(sheet1.Nf, sheet1.Ne, sheet1.Nv))

# ## Minimize energy
res = solver.find_energy_min(sheet1, geom, smodel)

# ## View the result
# =============================================================================
# draw_specs = config.draw.sheet_spec()
# draw_specs['vert']['visible'] = False
# draw_specs['edge']['head_width'] = 0  # values other than 0 gives error.
# =============================================================================
fig, ax = sheet_view(sheet1, mode = "quick")

sheet2 = sheet1


# Now, perform an arrangement/event, we use vertex merging.

from tyssue.topology.base_topology import collapse_edge

centre_edge = sheet1.edge_df.eval("sx**2 + sy**2").idxmin()
collapse_edge(sheet1, centre_edge)
sheet1.update_rank()
geom.update_all(sheet1)

fig, ax = sheet_view(sheet1, mode="quick")


# Now perform energy minimisation after vertex merging.
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers.quasistatic import QSSolver

solver = QSSolver() 	#Quasi-Static solver
res = solver.find_energy_min(sheet1, geom, smodel)
print("Successfull gradient descent? ", res['success'])

# Plot the new figure.
fig, ax = sheet_view(sheet1, mode = "quick")


''' '''





"""
This is the end of the script
"""
