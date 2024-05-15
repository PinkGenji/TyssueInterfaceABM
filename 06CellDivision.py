# -*- coding: utf-8 -*-
"""
This file is for learning 06CellDivision of the tyssue package.
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pylab as plt
import ipyvolume as ipv


from tyssue import Sheet, config


from tyssue.geometry.planar_geometry import PlanarGeometry as geom
from tyssue.solvers.quasistatic import QSSolver
from tyssue.dynamics.planar_vertex_model import PlanarModel as model

from tyssue.draw import sheet_view
from tyssue.stores import load_datasets

from tyssue.topology.sheet_topology import remove_face, cell_division

'''
On a 2D mesh:

'''

solver = QSSolver()
sheet = Sheet.planar_sheet_2d('division', 6, 6, 1, 1)
sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)

sheet.get_opposite()

# ## Set up the model
nondim_specs = config.dynamics.quasistatic_plane_spec()
dim_model_specs = model.dimensionalize(nondim_specs)
sheet.update_specs(dim_model_specs, reset=True)

print("Number of cells: {}\n"
      "          edges: {}\n"
      "          vertices: {}\n".format(sheet.Nf, sheet.Ne, sheet.Nv))

# ## Minimize energy
res = solver.find_energy_min(sheet, geom, model)

# ## View the result
draw_specs = config.draw.sheet_spec()
draw_specs['vert']['visible'] = False
draw_specs['edge']['head_width'] = 0.1
fig, ax = sheet_view(sheet, **draw_specs)
fig.set_size_inches(12, 5)

# Generate the daughter cell.

daughter = cell_division(sheet, 7, geom, angle=np.pi/2)

res = solver.find_energy_min(sheet, geom, model)
print(res['success'])

fig, ax = sheet_view(sheet, **draw_specs)
fig.set_size_inches(12, 5)


# =============================================================================
# Since the 3D drawing of Tyssue is not well written, we shall skip 3D part!
# =============================================================================









































'''
This is the end of the file.
'''
